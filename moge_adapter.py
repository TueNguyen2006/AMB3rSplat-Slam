"""Use cached MoGe-3 metric predictions inside AMB3R submap construction."""

from __future__ import annotations

import numpy as np
import torch

from .priors import MoGeCache


class MoGeGeometryPrior:
    def __init__(self, cache: MoGeCache, min_pixels: int = 2048,
                 min_frames: int = 3, max_log_spread: float = 0.18):
        self.cache = cache
        self.min_pixels = int(min_pixels)
        self.min_frames = int(min_frames)
        self.max_log_spread = float(max_log_spread)
        self.accepted_scales: list[float] = []
        self.global_scale: float | None = None
        self.global_log_uncertainty: float | None = None
        self.rejected = 0

    @torch.no_grad()
    def align_window(self, points, poses, confidence, frames, valid_static=None):
        device = points.device
        R, center = poses[:, :3, :3], poses[:, :3, 3]
        slam_depth = torch.einsum(
            "fij,fhwj->fhwi", R.transpose(1, 2), points - center[:, None, None, :]
        )[..., 2]
        per_frame, support = [], []
        for row, frame_id in enumerate(frames):
            try:
                prior = self.cache.load(int(frame_id), device)
            except FileNotFoundError:
                continue
            static = torch.ones_like(prior.valid, dtype=torch.bool)
            if valid_static is not None:
                static &= valid_static[row].bool()
            conf = confidence[row].float()
            positive_conf = conf[conf > 0]
            threshold = (0.1 * positive_conf.median()) if positive_conf.numel() else torch.inf
            valid = (static & prior.valid.bool() & (prior.depth > 1e-6)
                     & (slam_depth[row] > 1e-6) & torch.isfinite(slam_depth[row])
                     & torch.isfinite(prior.depth) & (conf > threshold))
            if int(valid.sum()) < self.min_pixels:
                continue
            ratio = torch.log(prior.depth[valid] / slam_depth[row][valid])
            lo, hi = torch.quantile(ratio, torch.tensor([0.1, 0.9], device=device))
            ratio = ratio[(ratio >= lo) & (ratio <= hi)]
            per_frame.append(float(ratio.median()))
            support.append(int(valid.sum()))
        if len(per_frame) < self.min_frames:
            self.rejected += 1
            return points, poses, {"accepted": False, "reason": "insufficient frames",
                                   "frames": len(per_frame)}
        logs = np.asarray(per_frame)
        center_log = float(np.median(logs))
        spread = float(1.4826 * np.median(np.abs(logs - center_log)))
        scale = float(np.exp(center_log))
        if not np.isfinite(scale) or scale <= 0 or spread > self.max_log_spread:
            self.rejected += 1
            return points, poses, {"accepted": False, "reason": "scale dispersion",
                                   "scale": scale, "log_mad": spread,
                                   "frames": len(per_frame)}
        if self.global_scale is None:
            # Monocular metric gauge is fixed once, from the first sufficiently
            # supported multi-frame window. Later windows validate this gauge but
            # never introduce an independent per-camera or per-submap scale.
            self.global_scale = scale
            self.global_log_uncertainty = spread
        disagreement = abs(float(np.log(scale / self.global_scale)))
        if disagreement > max(3.0 * self.max_log_spread, 0.35):
            self.rejected += 1
            return points, poses, {
                "accepted": False, "reason": "global scale disagreement",
                "window_scale": scale, "global_scale": self.global_scale,
                "log_disagreement": disagreement, "log_mad": spread,
                "frames": len(per_frame),
            }
        self.accepted_scales.append(scale)
        applied_scale = self.global_scale
        poses = poses.clone()
        poses[:, :3, 3] *= applied_scale
        return points * applied_scale, poses, {
            "accepted": True, "scale": applied_scale, "window_scale": scale,
            "global_scale_locked": True, "log_mad": spread,
            "global_log_uncertainty": self.global_log_uncertainty,
            "frames": len(per_frame), "median_pixels": int(np.median(support)),
            "sequence_scale_median": float(np.median(self.accepted_scales)),
            "sequence_scale_log_std": float(np.std(np.log(self.accepted_scales))),
        }
