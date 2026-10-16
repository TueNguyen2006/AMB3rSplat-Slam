"""MoGe-3 cache access and robust metric alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .masking import safe_masked_mean


@dataclass
class MoGeFrame:
    depth: Tensor
    points: Tensor
    normal: Tensor
    valid: Tensor
    K_normalized: Tensor


class MoGeCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, frame_id: int) -> Path:
        return self.root / f"{int(frame_id):06d}.npz"

    def load(self, frame_id: int, device: str | torch.device = "cpu") -> MoGeFrame:
        data = np.load(self.path(frame_id), allow_pickle=False)
        return MoGeFrame(*[
            torch.from_numpy(data[key]).to(device)
            for key in ("depth", "points", "normal", "valid", "K_normalized")
        ])


@torch.no_grad()
def robust_global_scale(slam_depths: list[Tensor], moge_depths: list[Tensor],
                        weights: list[Tensor], min_pixels: int = 2048,
                        max_log_spread: float = 0.18) -> tuple[float | None, dict]:
    """One sequence scale from per-frame weighted log-depth medians."""
    per_frame = []
    support = []
    for slam, moge, weight in zip(slam_depths, moge_depths, weights):
        valid = ((slam > 1e-6) & (moge > 1e-6) & torch.isfinite(slam)
                 & torch.isfinite(moge) & torch.isfinite(weight) & (weight > 0))
        if int(valid.sum()) < min_pixels:
            continue
        ratio = torch.log(moge[valid] / slam[valid])
        qlo, qhi = torch.quantile(ratio, torch.tensor([0.1, 0.9], device=ratio.device))
        ratio = ratio[(ratio >= qlo) & (ratio <= qhi)]
        per_frame.append(float(ratio.median()))
        support.append(int(valid.sum()))
    if len(per_frame) < 3:
        return None, {"accepted": False, "reason": "fewer than 3 supported frames"}
    values = np.asarray(per_frame)
    center = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - center)))
    scale = float(np.exp(center))
    accepted = bool(np.isfinite(scale) and scale > 0 and spread <= max_log_spread)
    return (scale if accepted else None), {
        "accepted": accepted,
        "scale": scale,
        "log_mad": spread,
        "frames": len(values),
        "median_pixels": int(np.median(support)),
        "reason": "ok" if accepted else f"log MAD {spread:.4f} exceeds {max_log_spread}",
    }


def depth_normal_loss(render_depth: Tensor, prior: MoGeFrame, static: Tensor,
                      depth_weight: float = 0.1, normal_weight: float = 0.02) -> tuple[Tensor, dict]:
    valid = static & prior.valid.bool() & torch.isfinite(render_depth) & (render_depth > 0)
    log_res = torch.log(render_depth.clamp_min(1e-6)) - torch.log(prior.depth.clamp_min(1e-6))
    depth = safe_masked_mean(torch.sqrt(log_res.square() + 1e-6), valid.float())
    dzdx = render_depth[:, 1:] - render_depth[:, :-1]
    dzdy = render_depth[1:, :] - render_depth[:-1, :]
    normal_proxy = torch.stack((
        -torch.nn.functional.pad(dzdx, (0, 1)),
        -torch.nn.functional.pad(dzdy, (0, 0, 0, 1)),
        torch.ones_like(render_depth),
    ), dim=-1)
    normal_proxy = torch.nn.functional.normalize(normal_proxy, dim=-1)
    cosine = 1.0 - (normal_proxy * prior.normal).sum(-1).abs().clamp(0, 1)
    normal = safe_masked_mean(cosine, valid.float())
    return depth_weight * depth + normal_weight * normal, {
        "depth": float(depth.detach()), "normal": float(normal.detach()),
        "valid_pixels": int(valid.sum()),
    }
