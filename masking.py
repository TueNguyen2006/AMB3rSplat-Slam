"""Static-observation masks and safe residual weighting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class StaticMaskProvider:
    """Loads cached human masks where True means a dynamic human pixel."""

    def __init__(self, cache_dir: str | Path | None, frame_count: int, dilation: int = 3):
        self.root = None if cache_dir is None else Path(cache_dir)
        self.frame_count = int(frame_count)
        self.dilation = max(0, int(dilation))
        self._cache: dict[int, torch.Tensor] = {}
        self.removed_queries = 0
        self.removed_pixels = 0

    def human_mask(self, frame_id: int, hw: tuple[int, int]) -> torch.Tensor:
        H, W = map(int, hw)
        if frame_id not in self._cache:
            path = None if self.root is None else self.root / f"{frame_id:06d}.png"
            if path is None or not path.is_file():
                mask = torch.zeros(H, W, dtype=torch.bool)
            else:
                arr = np.asarray(Image.open(path).convert("L")) > 0
                mask = torch.from_numpy(arr.copy())
            self._cache[frame_id] = mask
        mask = self._cache[frame_id]
        if mask.shape != (H, W):
            mask = F.interpolate(mask[None, None].float(), (H, W), mode="nearest")[0, 0].bool()
        if self.dilation:
            k = 2 * self.dilation + 1
            mask = F.max_pool2d(mask[None, None].float(), k, stride=1,
                                padding=self.dilation)[0, 0].bool()
        return mask

    def __call__(self, frames, hw, device=None) -> torch.Tensor:
        masks = torch.stack([~self.human_mask(int(f), hw) for f in frames])
        self.removed_queries += len(frames)
        self.removed_pixels += int((~masks).sum())
        return masks.to(device=device or "cpu")

    def effective_weight(self, weight: torch.Tensor, frame_id: int) -> torch.Tensor:
        static = ~self.human_mask(frame_id, weight.shape[-2:]).to(weight.device)
        return torch.where(static, weight, torch.zeros((), dtype=weight.dtype, device=weight.device))


def safe_masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(value) & torch.isfinite(weight) & (weight > 0)
    numerator = torch.where(valid, value * weight, torch.zeros_like(value)).sum()
    denominator = torch.where(valid, weight, torch.zeros_like(weight)).sum()
    return numerator / denominator.clamp_min(1e-8)
