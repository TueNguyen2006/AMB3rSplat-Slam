#!/usr/bin/env python3
"""Evaluate an incremental AMB3R-GS checkpoint on the established 18-view split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .gaussian_mapper import IncrementalGaussianMapper
from .masking import StaticMaskProvider
from .state import SharedReconstructionState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Completed amb3r_gs_slam run directory")
    parser.add_argument("--data", required=True, help="AMB3R direct dataset directory")
    parser.add_argument("--split", required=True)
    parser.add_argument("--mask-cache", default=None)
    parser.add_argument("--gsplat-root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def masked_psnr(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    squared = (prediction - target).square().mean(-1)
    mse = torch.where(valid, squared, torch.zeros_like(squared)).sum() / valid.sum().clamp_min(1)
    return float(-10.0 * torch.log10(mse.clamp_min(1e-10)))


@torch.no_grad()
def main():
    args = parse_args()
    run = Path(args.run)
    data = Path(args.data)
    output = Path(args.output) if args.output else run / "evaluation"
    render_dir = output / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    frame_ids = list(map(int, split["val_frame_ids"]))
    cameras = np.load(data / "cameras.npz", allow_pickle=False)
    trajectory = np.load(run / "trajectory.npz", allow_pickle=False)
    checkpoint = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    checkpoint["mapper"]["optimizers"] = {}
    shared = SharedReconstructionState(output)
    mapper = IncrementalGaussianMapper(args.gsplat_root, shared, max_gaussians=2_000_000)
    mapper.load_state_dict(checkpoint["mapper"])
    masks = StaticMaskProvider(args.mask_cache, len(cameras["frame_ids"]), dilation=3)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=True
    ).cuda()
    rows = []
    render_seconds = 0.0
    for frame_id in frame_ids:
        image_path = data / "images" / f"{frame_id:06d}.png"
        target = torch.from_numpy(
            np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).copy()
        ).cuda().float() / 255.0
        H, W = target.shape[:2]
        K = torch.from_numpy(cameras["K"][frame_id]).cuda().float()
        c2w = torch.from_numpy(trajectory["c2w"][frame_id]).cuda().float()
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction, alpha, depth, _ = mapper.render(c2w, K, (H, W))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        render_seconds += elapsed
        prediction = prediction.clamp(0, 1)
        full = torch.ones((H, W), dtype=torch.bool, device="cuda")
        static = masks([frame_id], (H, W), device="cuda")[0]
        pred_nchw = prediction.permute(2, 0, 1)[None]
        target_nchw = target.permute(2, 0, 1)[None]
        ssim_metric.reset()
        lpips_metric.reset()
        static_prediction = torch.where(static[..., None], prediction, target)
        static_nchw = static_prediction.permute(2, 0, 1)[None]
        ssim = float(ssim_metric(pred_nchw, target_nchw))
        lpips = float(lpips_metric(pred_nchw, target_nchw))
        ssim_metric.reset()
        lpips_metric.reset()
        row = {
            "frame_id": frame_id,
            "psnr": masked_psnr(prediction, target, full),
            "static_psnr": masked_psnr(prediction, target, static),
            "ssim": ssim,
            "lpips_vgg_normalized": lpips,
            "static_ssim_composited": float(ssim_metric(static_nchw, target_nchw)),
            "static_lpips_vgg_composited": float(lpips_metric(static_nchw, target_nchw)),
            "static_fraction": float(static.float().mean()),
            "visible_fraction": float((alpha[..., 0] > 0.02).float().mean()),
            "render_seconds": elapsed,
        }
        rows.append(row)
        pred_u8 = prediction.mul(255).byte().cpu().numpy()
        gt_u8 = target.mul(255).byte().cpu().numpy()
        imageio.imwrite(render_dir / f"{frame_id:06d}_comparison.png",
                        np.concatenate((gt_u8, pred_u8), axis=1))
        depth_np = depth.cpu().numpy()
        finite = np.isfinite(depth_np) & (depth_np > 0)
        if finite.any():
            lo, hi = np.quantile(depth_np[finite], [0.02, 0.98])
            depth_u8 = (np.clip((depth_np - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(render_dir / f"{frame_id:06d}_depth.png", depth_u8)
    keys = ("psnr", "static_psnr", "ssim", "lpips_vgg_normalized",
            "static_ssim_composited", "static_lpips_vgg_composited",
            "static_fraction", "visible_fraction")
    report = {
        "protocol": "transductive 18-view evaluation; AMB3R geometry observed the full sequence",
        "strict_unseen": False,
        "frame_ids": frame_ids,
        "metrics": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "render_fps": len(rows) / max(render_seconds, 1e-9),
        "gaussians": len(mapper.params["means"]),
        "per_frame": rows,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
