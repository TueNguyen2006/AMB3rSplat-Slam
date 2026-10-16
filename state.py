"""Versioned shared state spanning AMB3R, priors and Gaussian mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .geometry import invert_pose, validate_pose


@dataclass
class FrameState:
    frame_id: int
    timestamp: float
    rgb_path: str
    K: Tensor
    c2w: Tensor
    slam_confidence: Tensor | None = None
    human_mask_path: str | None = None
    moge_path: str | None = None
    pose_version: int = 0

    @property
    def w2c(self) -> Tensor:
        return invert_pose(self.c2w)


@dataclass
class SubmapState:
    submap_id: int
    frame_ids: list[int]
    sim3: Tensor
    version: int = 0


@dataclass
class SharedReconstructionState:
    output_dir: Path
    frames: dict[int, FrameState] = field(default_factory=dict)
    submaps: dict[int, SubmapState] = field(default_factory=dict)
    pose_version: int = 0
    map_version: int = 0
    metric_scale: float = 1.0
    metric_scale_version: int = 0
    gaussian_params: dict[str, Tensor] | None = None
    gaussian_owner: Tensor | None = None
    gaussian_source_frame: Tensor | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)

    def add_frame(self, frame: FrameState) -> None:
        validate_pose(frame.c2w)
        if frame.K.shape != (3, 3) or not torch.isfinite(frame.K).all():
            raise ValueError("K must be finite (3, 3)")
        self.frames[frame.frame_id] = frame

    def add_submap(self, submap_id: int, frame_ids: list[int], sim3: Tensor) -> None:
        if sim3.shape != (8,) or not torch.isfinite(sim3).all() or float(sim3[7]) <= 0:
            raise ValueError("submap Sim(3) must be finite with positive scale")
        self.submaps[int(submap_id)] = SubmapState(int(submap_id), list(frame_ids), sim3.detach().cpu())

    def commit_pose_update(self, submap_id: int, old: Tensor, new: Tensor,
                           reason: dict[str, Any]) -> None:
        sm = self.submaps[int(submap_id)]
        sm.sim3 = new.detach().cpu()
        sm.version += 1
        self.pose_version += 1
        for frame_id in sm.frame_ids:
            if frame_id in self.frames:
                self.frames[frame_id].pose_version = self.pose_version
        self.logs.append({
            "event": "pose_commit",
            "submap": int(submap_id),
            "pose_version": self.pose_version,
            "old": old.detach().cpu().tolist(),
            "new": new.detach().cpu().tolist(),
            **reason,
        })

    def bind_gaussians(self, params: dict[str, Tensor], owner: Tensor,
                       source_frame: Tensor) -> None:
        n = len(params["means"])
        if len(owner) != n or len(source_frame) != n:
            raise ValueError("Gaussian ownership arrays must match means")
        self.gaussian_params = params
        self.gaussian_owner = owner
        self.gaussian_source_frame = source_frame
        self.map_version += 1

    def validate(self) -> dict[str, Any]:
        inverse_error = 0.0
        for frame in self.frames.values():
            validate_pose(frame.c2w)
            inverse_error = max(inverse_error, float((frame.c2w @ frame.w2c - torch.eye(4)).abs().max()))
        if self.gaussian_params is not None:
            n = len(self.gaussian_params["means"])
            assert self.gaussian_owner is not None and len(self.gaussian_owner) == n
            assert all(torch.isfinite(value).all() for value in self.gaussian_params.values())
        else:
            n = 0
        return {
            "frames": len(self.frames),
            "submaps": len(self.submaps),
            "gaussians": n,
            "pose_version": self.pose_version,
            "map_version": self.map_version,
            "metric_scale": self.metric_scale,
            "metric_scale_version": self.metric_scale_version,
            "max_c2w_w2c_error": inverse_error,
        }
