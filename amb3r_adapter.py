"""Hooks that couple official AMB3R streaming state to the Gaussian mapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .masking import StaticMaskProvider
from .state import FrameState, SharedReconstructionState


class JointMappingSink:
    needs_dense = True
    retain_dense = False

    def __init__(self, shared: SharedReconstructionState, mapper, masks: StaticMaskProvider,
                 Ks: torch.Tensor, fps: float, local_steps: int = 3,
                 feedback: bool = True, feedback_after_submap: int = 2,
                 feedback_iterations: int = 12, log_path: str | Path | None = None,
                 checkpoint_callback=None, max_new_per_keyframe: int = 5_000):
        self.shared = shared
        self.mapper = mapper
        self.masks = masks
        self.Ks = Ks.float()
        self.fps = float(fps)
        self.local_steps = int(local_steps)
        self.feedback = bool(feedback)
        self.feedback_after_submap = int(feedback_after_submap)
        self.feedback_iterations = int(feedback_iterations)
        self.log_path = None if log_path is None else Path(log_path)
        self.backend = None
        self.images = None
        self.seen_frames: set[int] = set()
        self.events: list[dict[str, Any]] = []
        self.moge_cache = None
        self.checkpoint_callback = checkpoint_callback
        self.max_new_per_keyframe = int(max_new_per_keyframe)

    def _log(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(event, default=float) + "\n")

    def on_begin(self, backend, images, frontend) -> None:
        self.backend, self.images = backend, images
        self._log({"event": "begin", "frames": int(images.shape[0]),
                   "height": int(images.shape[-2]), "width": int(images.shape[-1]),
                   "feedback": self.feedback})

    def _world_geometry(self, S, sm, row: int):
        from amb3r_slam.tools.geometry import quat_to_rmat, sim3_from_matrix, sim3_mul, sim3_to_se3_matrix
        local_pose = sm.poses[row].float()
        global_pose = sim3_to_se3_matrix(sim3_mul(S, sim3_from_matrix(local_pose.double()).float()))
        points = sm.dense[0][row].float()
        R, t, scale = quat_to_rmat(S[3:7]), S[:3], S[7]
        world = scale * points @ R.T + t
        depth = ((world - global_pose[:3, 3]) @ global_pose[:3, :3])[..., 2]
        return world, depth, global_pose

    @torch.no_grad()
    def on_submap(self, backend, submap_id: int, sm, replacing: bool = False) -> None:
        if sm.dense is None:
            raise RuntimeError("Joint mapper requires the dense AMB3R submap pass")
        S = backend._st.S[submap_id].detach()
        self.shared.add_submap(submap_id, sm.frames, S)
        if sm.metric and sm.metric_scale is not None:
            scale = float(sm.metric_scale)
            if abs(scale - self.shared.metric_scale) > 1e-9:
                self.shared.metric_scale = scale
                self.shared.metric_scale_version += 1
                self._log({"event": "metric_scale", "submap": int(submap_id),
                           "scale": scale,
                           "scale_version": self.shared.metric_scale_version})
        inserted = []
        rows = list(range(len(sm.frames)))
        if len(rows) > 5:
            rows = sorted(set([0, len(rows) // 4, len(rows) // 2,
                               3 * len(rows) // 4, len(rows) - 1]))
        for row in rows:
            frame_id = int(sm.frames[row])
            if frame_id in self.seen_frames and replacing:
                continue
            world, depth, c2w = self._world_geometry(S, sm, row)
            H, W = depth.shape
            static = self.masks([frame_id], (H, W), device=world.device)[0]
            rgb = ((self.images[frame_id].permute(1, 2, 0).to(world.device).float() + 1.0)
                   * 0.5).clamp(0, 1)
            K = self.Ks[frame_id].to(world.device)
            moge = None
            if self.moge_cache is not None:
                try:
                    moge = self.moge_cache.load(frame_id, world.device)
                except FileNotFoundError:
                    pass
            result = self.mapper.insert(
                submap_id, frame_id, world, rgb, sm.dense[1][row].float(), static,
                K, c2w, depth, max_new=self.max_new_per_keyframe, moge=moge,
            )
            inserted.append(result)
            self.seen_frames.add(frame_id)
            if frame_id not in self.shared.frames:
                self.shared.add_frame(FrameState(
                    frame_id=frame_id, timestamp=frame_id / self.fps,
                    rgb_path="", K=K.detach().cpu(), c2w=c2w.detach().cpu(),
                    slam_confidence=sm.dense[1][row].detach().cpu(),
                    human_mask_path=None if self.masks.root is None else str(
                        self.masks.root / f"{frame_id:06d}.png"),
                    moge_path=None if self.moge_cache is None else str(
                        self.moge_cache.path(frame_id)),
                ))
            else:
                frame = self.shared.frames[frame_id]
                frame.K = K.detach().cpu()
                frame.c2w = c2w.detach().cpu()
                frame.slam_confidence = sm.dense[1][row].detach().cpu()
                frame.human_mask_path = None if self.masks.root is None else str(
                    self.masks.root / f"{frame_id:06d}.png")
                frame.moge_path = None if self.moge_cache is None else str(
                    self.moge_cache.path(frame_id))

        active = sorted(self.mapper.keyframes)[-6:]
        losses = []
        torch.set_grad_enabled(True)
        try:
            for _ in range(self.local_steps):
                losses.append(self.mapper.mapping_step(active))
        finally:
            torch.set_grad_enabled(False)
        event = {"event": "submap", "submap": int(submap_id), "replacing": replacing,
                 "frames": list(map(int, sm.frames)), "insertions": inserted,
                 "mapping": losses, "pose_version": self.shared.pose_version}
        self._log(event)
        if self.feedback and submap_id >= self.feedback_after_submap and active:
            self._feedback(backend, submap_id, sm, active)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(submap_id)

    def _feedback(self, backend, submap_id: int, sm, active: list[int]) -> None:
        from amb3r_slam.tools.geometry import sim3_from_matrix, sim3_inv, sim3_mul, sim3_to_se3_matrix
        target = int(sm.frames[len(sm.frames) // 2])
        if target not in self.mapper.keyframes:
            return
        torch.set_grad_enabled(True)
        try:
            proposal = self.mapper.refine_pose(target, iterations=self.feedback_iterations)
        finally:
            torch.set_grad_enabled(False)
        if not proposal["accepted"]:
            self._log({"event": "feedback_rejected", "submap": submap_id,
                       "frame": target, **{k: v for k, v in proposal.items()
                                            if k != "candidate_c2w"}})
            return

        S_old = backend._st.S[submap_id].detach()
        row = sm.frames.index(target)
        current = sim3_to_se3_matrix(sim3_mul(
            S_old, sim3_from_matrix(sm.poses[row].double()).float()))
        delta_T = proposal["candidate_c2w"] @ torch.linalg.inv(current)
        delta = sim3_from_matrix(delta_T.double(), scale=1.0).float()
        S_new = sim3_mul(delta, S_old)

        old_scores, new_scores = [], []
        for frame_id in [f for f in active if f in self.mapper.keyframes][-4:]:
            item = self.mapper.keyframes[frame_id]
            if int(item["submap"]) != submap_id:
                continue
            old_scores.append(self.mapper.score_pose(frame_id, item["c2w"]))
            new_scores.append(self.mapper.score_pose(frame_id, delta_T @ item["c2w"]))
        if old_scores:
            old_geom = sum(x["geometry"] for x in old_scores) / len(old_scores)
            new_geom = sum(x["geometry"] for x in new_scores) / len(new_scores)
            old_photo = sum(x["photo"] for x in old_scores) / len(old_scores)
            new_photo = sum(x["photo"] for x in new_scores) / len(new_scores)
            multi_ok = new_geom <= old_geom * 1.01 and new_photo < old_photo
        else:
            old_geom = new_geom = old_photo = new_photo = float("nan")
            multi_ok = False
        info = {k: v for k, v in proposal.items() if k != "candidate_c2w"}
        info.update({"multi_geometry_before": old_geom, "multi_geometry_after": new_geom,
                     "multi_photo_before": old_photo, "multi_photo_after": new_photo})
        if not multi_ok:
            info["reason"] = "multi-view gate rejected"
            self._log({"event": "feedback_rejected", "submap": submap_id,
                       "frame": target, **info})
            return
        backend.apply_external_submap_pose(submap_id, S_new, weight=0.25, info=info)
        self._log({"event": "feedback_accepted", "submap": submap_id,
                   "frame": target, **info})

    def on_pose_commit(self, submap_id: int, old, new, sm,
                       source: str = "unknown", info: dict[str, Any] | None = None) -> None:
        from amb3r_slam.tools.geometry import quat_to_rmat, sim3_inv, sim3_mul
        delta = sim3_mul(new, sim3_inv(old))
        moved = self.mapper.transform_owned(submap_id, delta)
        rotation, translation, scale = quat_to_rmat(delta[3:7]), delta[:3], delta[7]
        for frame_id, item in self.mapper.keyframes.items():
            if int(item["submap"]) != submap_id:
                continue
            pose = item["c2w"]
            corrected = pose.clone()
            corrected[:3, :3] = rotation @ pose[:3, :3]
            corrected[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
            item["c2w"] = corrected
            if frame_id in self.shared.frames:
                self.shared.frames[frame_id].c2w = corrected.detach().cpu()
        details = dict(info or {})
        self.shared.commit_pose_update(submap_id, old, new,
                                       {"source": source, "gaussians_moved": moved, **details})
        self._log({"event": "map_synchronized", "submap": submap_id,
                   "source": source, "gaussians_moved": moved,
                   "pose_version": self.shared.pose_version,
                   "map_version": self.shared.map_version})

    def on_finish(self, backend, result) -> None:
        self._log({"event": "finish", "submaps": len(result.submaps),
                   "gaussians": 0 if self.mapper.params is None else len(self.mapper.params["means"]),
                   "pose_version": self.shared.pose_version,
                   "map_version": self.shared.map_version})
