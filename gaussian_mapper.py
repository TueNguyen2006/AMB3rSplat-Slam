"""Incremental gsplat mapper with coverage insertion and pose refinement."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import invert_pose, rotation_angle, se3_exp
from .masking import safe_masked_mean

C0 = 0.28209479177387814


def _depth_normal(depth: Tensor, K: Tensor) -> Tensor:
    dzdx = F.pad(depth[:, 1:] - depth[:, :-1], (0, 1))
    dzdy = F.pad(depth[1:, :] - depth[:-1, :], (0, 0, 0, 1))
    nx = -dzdx * K[0, 0]
    ny = -dzdy * K[1, 1]
    nz = torch.ones_like(depth)
    return F.normalize(torch.stack((nx, ny, nz), dim=-1), dim=-1)


def _load_gsplat(gsplat_root: str | Path):
    root = str(Path(gsplat_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from gsplat import rasterization
    from gsplat.scene.components.gaussian_scene import GaussianScene
    from gsplat.strategy import MCMCStrategy
    return rasterization, GaussianScene, MCMCStrategy


class IncrementalGaussianMapper:
    def __init__(self, gsplat_root: str | Path, state, device: str = "cuda",
                 max_gaussians: int = 600_000, sh_degree: int = 0,
                 coverage_alpha: float = 0.55):
        self.rasterization, GaussianScene, MCMCStrategy = _load_gsplat(gsplat_root)
        self.GaussianScene = GaussianScene
        self.state = state
        self.device = torch.device(device)
        self.max_gaussians = int(max_gaussians)
        self.sh_degree = int(sh_degree)
        self.coverage_alpha = float(coverage_alpha)
        self.scene = None
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.strategy = MCMCStrategy(cap_max=self.max_gaussians, refine_start_iter=80,
                                     refine_stop_iter=20_000, refine_every=100)
        self.strategy_state = self.strategy.initialize_state()
        self.step = 0
        self.keyframes: dict[int, dict[str, Tensor]] = {}
        self.stats = {"inserted": 0, "human_rejected": 0, "coverage_rejected": 0,
                      "local_steps": 0, "global_steps": 0}

    def state_dict(self) -> dict[str, Any]:
        return {
            "scene": None if self.scene is None else self.scene.state_dict(),
            "optimizers": {key: value.state_dict() for key, value in self.optimizers.items()},
            "strategy_state": self.strategy_state,
            "sh_degree": self.sh_degree,
            "step": self.step,
            "stats": self.stats,
            "keyframes": {
                frame_id: {
                    key: value.detach().cpu() if isinstance(value, Tensor) else value
                    for key, value in item.items()
                }
                for frame_id, item in self.keyframes.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        scene_state = payload.get("scene")
        if scene_state is None:
            return
        scene_state["splats"] = {
            key: value.to(self.device) for key, value in scene_state["splats"].items()
        }
        scene_state["signal"] = {
            key: value.to(self.device) for key, value in scene_state.get("signal", {}).items()
        }
        scene_state["component_index"] = scene_state["component_index"].to(self.device)
        self.scene = self.GaussianScene.from_state_dict(scene_state)
        self.sh_degree = int(payload.get("sh_degree", self.sh_degree))
        self._rebuild_optimizers()
        for key, state in payload.get("optimizers", {}).items():
            if key in self.optimizers:
                self.optimizers[key].load_state_dict(state)
        self.strategy_state = payload.get("strategy_state", self.strategy.initialize_state())
        self.step = int(payload.get("step", 0))
        self.stats.update(payload.get("stats", {}))
        self.keyframes = {
            int(frame_id): {
                key: value.to(self.device) if isinstance(value, Tensor) else value
                for key, value in item.items()
            }
            for frame_id, item in payload.get("keyframes", {}).items()
        }
        self.state.bind_gaussians(self.params, self.scene.component_index,
                                  self.scene.signal["source_frame"])

    @property
    def params(self):
        return None if self.scene is None else self.scene.splats

    def _new_component(self, means: Tensor, rgb: Tensor, scale: Tensor):
        n = len(means)
        coeffs = torch.zeros(n, (self.sh_degree + 1) ** 2, 3, device=self.device)
        coeffs[:, 0] = (rgb - 0.5) / C0
        return torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.log(scale.clamp_min(1e-5)).repeat(1, 3)),
            "quats": torch.nn.Parameter(F.pad(torch.zeros(n, 3, device=self.device), (1, 0), value=1.0)),
            "opacities": torch.nn.Parameter(torch.full((n,), torch.logit(torch.tensor(0.12)).item(), device=self.device)),
            "sh0": torch.nn.Parameter(coeffs[:, :1]),
            "shN": torch.nn.Parameter(coeffs[:, 1:]),
        })

    def _optimizer_state_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for key, optimizer in self.optimizers.items():
            parameter = optimizer.param_groups[0]["params"][0]
            snapshot[key] = {
                name: value.detach().clone() if isinstance(value, Tensor) else value
                for name, value in optimizer.state.get(parameter, {}).items()
            }
        return snapshot

    @staticmethod
    def _migrate_optimizer_tensor(old: Tensor, parameter: Tensor) -> Tensor:
        if old.ndim == 0:
            return old.to(parameter.device)
        migrated = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if old.ndim != parameter.ndim:
            return migrated
        overlap = tuple(slice(0, min(a, b)) for a, b in zip(old.shape, parameter.shape))
        migrated[overlap] = old.to(device=parameter.device, dtype=parameter.dtype)[overlap]
        return migrated

    def _rebuild_optimizers(self, previous: dict[str, dict[str, Any]] | None = None) -> None:
        lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
               "opacities": 5e-2, "sh0": 2.5e-3, "shN": 1.25e-4}
        self.optimizers = {
            key: torch.optim.Adam([{"params": value, "lr": lrs[key], "name": key}],
                                  eps=1e-15)
            for key, value in self.params.items()
        }
        for key, old_state in (previous or {}).items():
            if key not in self.optimizers or not old_state:
                continue
            optimizer = self.optimizers[key]
            parameter = optimizer.param_groups[0]["params"][0]
            optimizer.state[parameter] = {
                name: self._migrate_optimizer_tensor(value, parameter)
                if isinstance(value, Tensor) else value
                for name, value in old_state.items()
            }
        self.strategy.check_sanity(self.params, self.optimizers)

    @torch.no_grad()
    def promote_sh_degree(self, degree: int) -> bool:
        """Expand view-dependent SH only after SLAM/Sim(3) synchronization is final."""
        degree = int(degree)
        if self.params is None or degree <= self.sh_degree:
            return False
        previous = self._optimizer_state_snapshot()
        target = (degree + 1) ** 2 - 1
        current = self.params["shN"]
        expanded = torch.zeros(
            current.shape[0], target, 3, device=current.device, dtype=current.dtype
        )
        expanded[:, : current.shape[1]] = current.detach()
        self.scene.splats["shN"] = torch.nn.Parameter(expanded)
        self.sh_degree = degree
        self._rebuild_optimizers(previous)
        return True

    def render(self, c2w: Tensor, K: Tensor, hw: tuple[int, int], sh_degree: int | None = None):
        H, W = map(int, hw)
        p = self.params
        if p is None or len(p["means"]) == 0:
            return (torch.zeros(H, W, 3, device=self.device),
                    torch.zeros(H, W, 1, device=self.device),
                    torch.zeros(H, W, device=self.device), {})
        colors = torch.cat((p["sh0"], p["shN"]), dim=1)
        renders, alpha, info = self.rasterization(
            means=p["means"], quats=F.normalize(p["quats"], dim=-1),
            scales=torch.exp(p["scales"]), opacities=torch.sigmoid(p["opacities"]),
            colors=colors, viewmats=invert_pose(c2w)[None], Ks=K[None],
            width=W, height=H, packed=False, render_mode="RGB+ED",
            sh_degree=self.sh_degree if sh_degree is None else sh_degree,
        )
        return renders[0, ..., :3], alpha[0], renders[0, ..., 3], info

    @torch.no_grad()
    def insert(self, submap_id: int, frame_id: int, means_world: Tensor, rgb: Tensor,
               confidence: Tensor, valid_static: Tensor, K: Tensor, c2w: Tensor,
               depth: Tensor, max_new: int = 20_000, moge=None) -> dict[str, Any]:
        means_world = means_world.to(self.device).reshape(-1, 3)
        rgb = rgb.to(self.device).reshape(-1, 3)
        conf = confidence.to(self.device).reshape(-1)
        static = valid_static.to(self.device).reshape(-1).bool()
        finite = torch.isfinite(means_world).all(-1) & torch.isfinite(conf)
        positive = depth.to(self.device).reshape(-1) > 1e-6
        threshold = torch.quantile(conf[finite], 0.35) if bool(finite.any()) else torch.inf
        keep = finite & positive & static & (conf >= threshold)
        self.stats["human_rejected"] += int((finite & positive & ~static).sum())

        H, W = valid_static.shape
        self.keyframes[int(frame_id)] = {
            "rgb": rgb.reshape(H, W, 3).detach(), "K": K.to(self.device).detach(),
            "c2w": c2w.to(self.device).detach(), "depth": depth.to(self.device).detach(),
            "static": valid_static.to(self.device).detach(), "submap": torch.tensor(submap_id),
        }
        if moge is not None:
            self.keyframes[int(frame_id)]["moge_depth"] = moge.depth.to(self.device).detach()
            self.keyframes[int(frame_id)]["moge_normal"] = moge.normal.to(self.device).detach()
            self.keyframes[int(frame_id)]["moge_valid"] = moge.valid.to(self.device).detach()
        if self.params is not None:
            _, alpha, rendered_depth, _ = self.render(c2w.to(self.device), K.to(self.device), (H, W))
            covered = alpha[..., 0].reshape(-1) >= self.coverage_alpha
            consistent = (rendered_depth.reshape(-1) - depth.to(self.device).reshape(-1)).abs() \
                <= 0.08 * depth.to(self.device).reshape(-1).abs().clamp_min(1e-4)
            coverage_reject = keep & covered & consistent
            self.stats["coverage_rejected"] += int(coverage_reject.sum())
            keep &= ~(covered & consistent)

        ids = torch.nonzero(keep).squeeze(1)
        if len(ids) > max_new:
            order = torch.argsort(conf[ids], descending=True)
            ids = ids[order[:max_new]]
        budget = self.max_gaussians - (0 if self.params is None else len(self.params["means"]))
        ids = ids[:max(0, budget)]
        if not len(ids):
            return {"inserted": 0, "frame": int(frame_id), "submap": int(submap_id)}

        z = depth.to(self.device).reshape(-1)[ids]
        focal = 0.5 * (K[0, 0] + K[1, 1]).to(self.device)
        scale = (1.5 * z / focal.clamp_min(1.0)).clamp_min(1e-5)[:, None]
        component = self._new_component(means_world[ids], rgb[ids], scale)
        previous = self._optimizer_state_snapshot()
        if self.scene is None:
            self.scene = self.GaussianScene.from_splats(
                component, id="amb3r", signal={
                    "source_frame": torch.full((len(ids),), int(frame_id), device=self.device, dtype=torch.long),
                })
            self.scene.component_names[0] = f"submap_{submap_id}"
        else:
            old_n = len(self.params["means"])
            self.scene.put(f"submap_{submap_id}_{frame_id}", component)
            self.scene.signal["source_frame"][old_n:] = int(frame_id)
        self._rebuild_optimizers(previous)
        self.stats["inserted"] += len(ids)
        self.state.bind_gaussians(self.params, self.scene.component_index,
                                  self.scene.signal["source_frame"])
        return {"inserted": len(ids), "frame": int(frame_id), "submap": int(submap_id),
                "total": len(self.params["means"])}

    def mapping_step(self, frame_ids: list[int], depth_weight: float = 0.05) -> dict[str, float]:
        if self.params is None:
            return {"loss": float("nan")}
        total = torch.zeros((), device=self.device)
        used = 0
        last_info = None
        for frame_id in frame_ids:
            item = self.keyframes.get(int(frame_id))
            if item is None:
                continue
            rgb, alpha, depth, info = self.render(item["c2w"], item["K"], item["rgb"].shape[:2])
            static = item["static"].bool()
            photo = safe_masked_mean((rgb - item["rgb"]).abs().mean(-1), static.float())
            depth_valid = static & (item["depth"] > 1e-6) & (alpha[..., 0] > 0.05)
            log_depth = (torch.log(depth.clamp_min(1e-6))
                         - torch.log(item["depth"].clamp_min(1e-6))).abs()
            total = total + photo + depth_weight * safe_masked_mean(log_depth, depth_valid.float())
            if "moge_depth" in item:
                mv = static & item["moge_valid"].bool() & (item["moge_depth"] > 1e-6) \
                    & (alpha[..., 0] > 0.05)
                moge_res = (torch.log(depth.clamp_min(1e-6))
                            - torch.log(item["moge_depth"].clamp_min(1e-6))).abs()
                total = total + 0.04 * safe_masked_mean(moge_res, mv.float())
                render_normal = _depth_normal(depth, item["K"])
                normal_res = 1.0 - (render_normal * item["moge_normal"]).sum(-1).abs().clamp(0, 1)
                total = total + 0.01 * safe_masked_mean(normal_res, mv.float())
            last_info, used = info, used + 1
        if not used:
            return {"loss": float("nan")}
        loss = total / used
        self.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state,
                                        self.step, last_info)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.params.values()), 1.0)
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state,
                                         self.step, last_info, lr=1.6e-4, scene=self.scene)
        self.step += 1
        self.stats["local_steps"] += 1
        self.state.bind_gaussians(self.params, self.scene.component_index,
                                  self.scene.signal["source_frame"])
        return {"loss": float(loss.detach()), "frames": used,
                "gaussians": len(self.params["means"])}

    def refine_pose(self, frame_id: int, iterations: int = 12,
                    translation_limit: float = 0.03, rotation_limit_deg: float = 2.0) -> dict[str, Any]:
        item = self.keyframes[int(frame_id)]
        base = item["c2w"].detach()

        def objective(c2w):
            rgb, alpha, depth, _ = self.render(c2w, item["K"], item["rgb"].shape[:2])
            static = item["static"].bool() & (alpha[..., 0] > 0.05)
            photo = safe_masked_mean((rgb - item["rgb"]).abs().mean(-1), static.float())
            geom_valid = static & (item["depth"] > 1e-6)
            geom = safe_masked_mean((torch.log(depth.clamp_min(1e-6))
                                     - torch.log(item["depth"].clamp_min(1e-6))).abs(),
                                    geom_valid.float())
            if "moge_depth" in item:
                mv = static & item["moge_valid"].bool() & (item["moge_depth"] > 1e-6)
                moge_geom = safe_masked_mean((torch.log(depth.clamp_min(1e-6))
                                              - torch.log(item["moge_depth"].clamp_min(1e-6))).abs(),
                                             mv.float())
                geom = 0.7 * geom + 0.3 * moge_geom
                render_normal = _depth_normal(depth, item["K"])
                normal = 1.0 - (render_normal * item["moge_normal"]).sum(-1).abs().clamp(0, 1)
                geom = geom + 0.05 * safe_masked_mean(normal, mv.float())
            return photo + 0.1 * geom, photo, geom, int(geom_valid.sum())

        with torch.no_grad():
            before, before_p, before_g, support = objective(base)
        delta = torch.zeros(6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=2e-3)
        for _ in range(int(iterations)):
            c2w = base @ se3_exp(delta)
            loss, _, _, _ = objective(c2w)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], 0.05)
            optimizer.step()
            with torch.no_grad():
                delta[:3].clamp_(-translation_limit, translation_limit)
                rlim = math.radians(rotation_limit_deg)
                if delta[3:].norm() > rlim:
                    delta[3:].mul_(rlim / delta[3:].norm())
        candidate = (base @ se3_exp(delta.detach())).detach()
        with torch.no_grad():
            after, after_p, after_g, _ = objective(candidate)
        trans = float(delta[:3].norm().detach())
        rot = float(torch.rad2deg(delta[3:].norm()).detach())
        accepted = bool(
            support >= 512 and torch.isfinite(after) and after < before * 0.995
            and after_g <= before_g * 1.01 and trans <= translation_limit
            and rot <= rotation_limit_deg
        )
        return {
            "accepted": accepted, "candidate_c2w": candidate,
            "before": float(before), "after": float(after),
            "photo_before": float(before_p), "photo_after": float(after_p),
            "geometry_before": float(before_g), "geometry_after": float(after_g),
            "translation": trans, "rotation_deg": rot, "support": support,
            "reason": "accepted" if accepted else "trust/geometric gate rejected",
        }

    @torch.no_grad()
    def score_pose(self, frame_id: int, c2w: Tensor) -> dict[str, float]:
        item = self.keyframes[int(frame_id)]
        rgb, alpha, depth, _ = self.render(c2w, item["K"], item["rgb"].shape[:2])
        static = item["static"].bool() & (alpha[..., 0] > 0.05)
        photo = safe_masked_mean((rgb - item["rgb"]).abs().mean(-1), static.float())
        valid = static & (item["depth"] > 1e-6)
        geometry = safe_masked_mean((torch.log(depth.clamp_min(1e-6))
                                     - torch.log(item["depth"].clamp_min(1e-6))).abs(),
                                    valid.float())
        return {"photo": float(photo), "geometry": float(geometry),
                "support": int(valid.sum())}

    @torch.no_grad()
    def transform_owned(self, submap_id: int, delta_sim3: Tensor) -> int:
        if self.params is None:
            return 0
        owner = self.scene.component_index
        names = self.scene.component_names
        component_ids = [i for i, name in enumerate(names) if name.startswith(f"submap_{submap_id}")]
        mask = torch.zeros_like(owner, dtype=torch.bool)
        for component_id in component_ids:
            mask |= owner == component_id
        if not bool(mask.any()):
            return 0
        from amb3r_slam.tools.geometry import quat_to_rmat
        d = delta_sim3.to(device=self.device, dtype=self.params["means"].dtype)
        R, t, scale = quat_to_rmat(d[3:7]), d[:3], d[7]
        self.params["means"][mask] = scale * self.params["means"][mask] @ R.T + t
        self.params["scales"][mask] += torch.log(scale)
        # AMB3R Sim(3) uses xyzw while gsplat uses wxyz.
        a = torch.cat((d[6:7], d[3:6])).expand(int(mask.sum()), 4)
        b = F.normalize(self.params["quats"][mask], dim=-1)
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        rotated = torch.stack((
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ), dim=-1)
        self.params["quats"][mask] = F.normalize(rotated, dim=-1)
        self.state.map_version += 1
        return int(mask.sum())
