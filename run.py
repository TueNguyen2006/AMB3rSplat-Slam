#!/usr/bin/env python3
"""Run official AMB3R-SLAM with synchronous incremental gsplat mapping."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AMB3R_ROOT = ROOT / "thirdparty" / "amb3r-slam"
if str(AMB3R_ROOT) not in sys.path:
    sys.path.insert(0, str(AMB3R_ROOT))

from amb3r_gs_slam.amb3r_adapter import JointMappingSink
from amb3r_gs_slam.gaussian_mapper import IncrementalGaussianMapper, C0
from amb3r_gs_slam.masking import StaticMaskProvider
from amb3r_gs_slam.moge_adapter import MoGeGeometryPrior
from amb3r_gs_slam.priors import MoGeCache
from amb3r_gs_slam.state import FrameState, SharedReconstructionState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "outputs/amb3r_slam_omega_gsplat"))
    parser.add_argument("--output", default=str(ROOT / "outputs/amb3r_gs_slam"))
    parser.add_argument("--config", default=str(ROOT / "amb3r_gs_slam/configs/rtx4070.yaml"))
    parser.add_argument("--mask-cache", default=None)
    parser.add_argument("--moge-cache", default=None)
    parser.add_argument("--moge-scale-only", action="store_true",
                        help="Use MoGe only to lock the global SLAM scale, not mapping losses")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-steps", type=int, default=None)
    parser.add_argument("--global-steps", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_images(paths: list[Path], max_frames: int) -> torch.Tensor:
    if max_frames > 0:
        paths = paths[:max_frames]
    arrays = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy() for path in paths]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()
    return (tensor / 127.5 - 1.0)[None]


def write_ply(path: Path, mapper) -> int:
    from plyfile import PlyData, PlyElement
    p = mapper.params
    if p is None:
        return 0
    keep = torch.sigmoid(p["opacities"]) > 0.01
    xyz = p["means"][keep].detach().cpu().numpy().astype(np.float32)
    rgb = ((p["sh0"][keep, 0].detach().cpu().numpy() * C0 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    opacity = torch.sigmoid(p["opacities"][keep]).detach().cpu().numpy().astype(np.float32)
    owner = mapper.scene.component_index[keep].detach().cpu().numpy().astype(np.int32)
    frame = mapper.scene.signal["source_frame"][keep].detach().cpu().numpy().astype(np.int32)
    vertex = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                              ("opacity", "f4"), ("submap_id", "i4"),
                                              ("frame_id", "i4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    vertex["opacity"], vertex["submap_id"], vertex["frame_id"] = opacity, owner, frame
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    return len(vertex)


def write_standard_3dgs_ply(path: Path, mapper) -> int:
    """Write the canonical Inria/SIBR Gaussian property layout."""
    from plyfile import PlyData, PlyElement
    p = mapper.params
    if p is None:
        return 0
    keep = torch.sigmoid(p["opacities"]) > 0.01
    values = {
        "x": p["means"][keep, 0], "y": p["means"][keep, 1], "z": p["means"][keep, 2],
        "nx": torch.zeros_like(p["opacities"][keep]),
        "ny": torch.zeros_like(p["opacities"][keep]),
        "nz": torch.zeros_like(p["opacities"][keep]),
        "f_dc_0": p["sh0"][keep, 0, 0], "f_dc_1": p["sh0"][keep, 0, 1],
        "f_dc_2": p["sh0"][keep, 0, 2], "opacity": p["opacities"][keep],
    }
    rest = p["shN"][keep].transpose(1, 2).reshape(int(keep.sum()), -1)
    for index in range(rest.shape[1]):
        values[f"f_rest_{index}"] = rest[:, index]
    for index in range(3):
        values[f"scale_{index}"] = p["scales"][keep, index]
    quats = torch.nn.functional.normalize(p["quats"][keep], dim=-1)
    # gsplat stores wxyz; the canonical 3DGS layout also expects wxyz.
    for index in range(4):
        values[f"rot_{index}"] = quats[:, index]
    dtype = [(name, "f4") for name in values]
    vertex = np.empty(int(keep.sum()), dtype=dtype)
    for name, tensor in values.items():
        vertex[name] = tensor.detach().float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    return len(vertex)


def save_checkpoint(path: Path, mapper, shared, result=None) -> None:
    payload = {
        "mapper": mapper.state_dict(),
        "shared_validation": shared.validate(), "shared_logs": shared.logs,
    }
    if result is not None:
        payload["poses_c2w"] = result.poses
        payload["submap_poses"] = torch.stack(result.submap_poses)
        payload["keyframes"] = torch.as_tensor(result.keyframes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def save_visualizations(out: Path, mapper) -> dict[str, object]:
    render_dir = out / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(mapper.keyframes)
    videos = {"render": [], "comparison": []}
    per_frame = []
    for frame_id in frames:
        item = mapper.keyframes[frame_id]
        rendered, alpha, depth, _ = mapper.render(
            item["c2w"], item["K"], item["rgb"].shape[:2])
        render_u8 = (rendered.clamp(0, 1).mul(255).byte().cpu().numpy())
        gt_u8 = (item["rgb"].clamp(0, 1).mul(255).byte().cpu().numpy())
        valid = alpha[..., 0] > 0.02
        error = (rendered - item["rgb"]).abs().mean(-1)
        mae = float(safe_mean(error, valid))
        comparison = np.concatenate((gt_u8, render_u8), axis=1)
        imageio.imwrite(render_dir / f"{frame_id:06d}_render.png", render_u8)
        imageio.imwrite(render_dir / f"{frame_id:06d}_comparison.png", comparison)
        depth_np = depth.detach().cpu().numpy()
        depth_valid = depth_np[np.asarray(valid.cpu())]
        if depth_valid.size:
            lo, hi = np.quantile(depth_valid, [0.02, 0.98])
            normalized = np.clip((depth_np - lo) / max(hi - lo, 1e-6), 0, 1)
            imageio.imwrite(render_dir / f"{frame_id:06d}_depth.png",
                            (normalized * 255).astype(np.uint8))
        videos["render"].append(render_u8)
        videos["comparison"].append(comparison)
        per_frame.append({"frame_id": int(frame_id), "masked_l1": mae,
                          "visible_pixels": int(valid.sum())})
    for name, sequence in videos.items():
        if sequence:
            try:
                imageio.mimwrite(out / f"{name}.mp4", sequence, fps=5,
                                 codec="libx264", quality=8)
            except Exception as exc:
                (out / f"{name}_video_error.txt").write_text(str(exc) + "\n")
    return {"frames": len(frames), "per_frame": per_frame}


def safe_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid & torch.isfinite(value)
    return torch.where(valid, value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(1)


def main():
    total_started = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    camera_data = np.load(data / "cameras.npz", allow_pickle=False)
    paths = sorted((data / "images").glob("*.png"))
    n = min(len(paths), args.max_frames) if args.max_frames > 0 else len(paths)
    paths = paths[:n]
    Ks = torch.from_numpy(camera_data["K"][:n]).float()
    frame_ids = camera_data["frame_ids"][:n].astype(int)
    if not np.array_equal(frame_ids, np.arange(n)):
        raise ValueError("online runner currently requires contiguous zero-based frame ids")
    images = load_images(paths, 0)
    H, W = images.shape[-2:]
    if tuple(camera_data["K"].shape[1:]) != (3, 3):
        raise ValueError("invalid camera intrinsics")

    shared = SharedReconstructionState(out)
    for i, path in enumerate(paths):
        shared.frames[i] = FrameState(i, i / float(config["fps"]), str(path), Ks[i],
                                      torch.eye(4))
    masks = StaticMaskProvider(args.mask_cache, n, config["masking"]["dilation"])
    if config["masking"]["enabled"] and args.mask_cache is None:
        print("[warning] SAM 3 cache unavailable: using explicit empty-mask ablation")

    max_gaussians = args.max_gaussians or config["mapping"]["max_gaussians"]
    mapper = IncrementalGaussianMapper(
        config["mapping"]["gsplat_root"], shared, device=config["device"],
        max_gaussians=max_gaussians, sh_degree=config["mapping"]["sh_degree"],
        coverage_alpha=config["mapping"]["coverage_alpha"],
    )
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        mapper.load_state_dict(resume_payload["mapper"])
    if args.mapping_only and resume_payload is None:
        raise ValueError("--mapping-only requires --resume CHECKPOINT")
    if args.resume and not args.mapping_only:
        raise ValueError("resume is currently supported for --mapping-only refinement")
    feedback = config["joint"]["gs_to_slam_feedback"] if args.feedback is None else args.feedback
    sink = JointMappingSink(
        shared, mapper, masks, Ks, config["fps"],
        local_steps=args.local_steps if args.local_steps is not None else config["mapping"]["local_steps"],
        feedback=feedback, feedback_after_submap=config["joint"]["feedback_after_submap"],
        feedback_iterations=config["joint"]["feedback_iterations"],
        log_path=out / "joint_events.jsonl",
        checkpoint_callback=lambda _submap_id: save_checkpoint(
            out / "checkpoints/latest.pt", mapper, shared),
        max_new_per_keyframe=config["mapping"]["max_new_per_keyframe"],
    )
    moge_prior = None
    if args.moge_cache:
        cache = MoGeCache(args.moge_cache)
        sink.moge_cache = None if args.moge_scale_only else cache
        moge_prior = MoGeGeometryPrior(cache)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = None
    if not args.mapping_only:
        from amb3r_slam.model import load_model
        from amb3r_slam.pipeline import AMB3R_SLAM
        print("Loading official AMB3R-SLAM VGGT-Omega...")
        model = load_model("omega", config["slam"]["checkpoint"], config["device"], fp32=False).eval()
        pipeline = AMB3R_SLAM(
            model, fps=config["fps"], device=config["device"],
            overrides={"backend": {"frame_stride": config["slam"]["frame_stride"],
                                     "loop": {"pgo_mode": "online" if config["slam"]["online_pgo"] else "offline"}}},
        )
        pipeline.backend.validity_provider = masks
        pipeline.backend.geometry_prior = moge_prior
        pipeline.backend.event_sink = sink
        result = pipeline.run(images, verbose=True)
    slam_mapping_time = time.time() - started

    global_steps = args.global_steps if args.global_steps is not None else config["mapping"]["global_steps"]
    global_started = time.time()
    global_losses = []
    mapper.promote_sh_degree(config["mapping"].get("final_sh_degree", mapper.sh_degree))
    all_frames = sorted(mapper.keyframes)
    for step in range(global_steps):
        if not all_frames:
            break
        window = [all_frames[(step * 4 + j) % len(all_frames)] for j in range(min(8, len(all_frames)))]
        global_losses.append(mapper.mapping_step(window))
        mapper.stats["global_steps"] += 1
    global_time = time.time() - global_started

    export_started = time.time()
    if result is not None:
        trajectory = result.poses[:n].cpu().numpy()
        np.savez_compressed(out / "trajectory.npz", c2w=trajectory,
                            w2c=np.linalg.inv(trajectory), frame_ids=frame_ids,
                            K=Ks.cpu().numpy(), width=np.int32(W), height=np.int32(H),
                            keyframes=np.asarray(result.keyframes),
                            submap_sim3=torch.stack(result.submap_poses).cpu().numpy())
    save_checkpoint(out / "checkpoints/final.pt", mapper, shared, result)
    latest_checkpoint = out / "checkpoints/latest.pt"
    if latest_checkpoint.exists():
        latest_checkpoint.unlink()
    ply_count = write_ply(out / "gaussians/final_gaussians.ply", mapper)
    write_standard_3dgs_ply(out / "gaussians/final_gaussians_splat.ply", mapper)
    visualization = save_visualizations(out, mapper)
    export_time = time.time() - export_started
    report = {
        "status": "completed", "frames": n,
        "submaps": len(result.submaps) if result is not None else len(shared.submaps),
        "loops": int(result.stats.get("n_loops", 0)) if result is not None else None,
        "mapping_only": args.mapping_only, "feedback_enabled": feedback,
        "feedback_proposed": sum(e["event"].startswith("feedback_") for e in sink.events),
        "feedback_accepted": sum(e["event"] == "feedback_accepted" for e in sink.events),
        "runtime_s": time.time() - total_started,
        "runtime_stages_s": {"slam_and_incremental_mapping": slam_mapping_time,
                             "global_refinement": global_time,
                             "export_and_visualization": export_time},
        "fps": None if args.mapping_only else n / slam_mapping_time,
        "peak_vram_gb_torch": torch.cuda.max_memory_allocated() / 1e9,
        "gaussians": ply_count, "mapper_stats": mapper.stats,
        "mask_removed_pixels": masks.removed_pixels,
        "mask_mode": "sam3" if args.mask_cache else "empty-mask-ablation",
        "moge_mode": ("scale-only" if args.moge_scale_only else "scale+depth+normal")
            if args.moge_cache else "disabled",
        "moge_geometry_windows": [] if result is None else
            result.stats.get("geometry_prior_windows", []),
        "pose_graph_commits": [] if result is None else
            result.stats.get("pose_graph_commits", []),
        "external_pose_updates": [] if result is None else
            result.stats.get("external_pose_updates", []),
        "global_last": global_losses[-1] if global_losses else None,
        "visualization": visualization,
        "state": shared.validate(),
    }
    (out / "run_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
