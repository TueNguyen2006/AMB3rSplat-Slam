#!/usr/bin/env python3
"""Precompute official MoGe-3 metric depth, points and normals."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    from moge.model.v3 import MoGeModel

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cameras = np.load(args.cameras, allow_pickle=False)
    Ks = cameras["K"].astype(np.float32)
    frame_ids = cameras["frame_ids"].astype(int)
    paths = sorted(Path(args.images).glob("*.png"))
    if len(paths) != len(frame_ids):
        raise ValueError(f"{len(paths)} images != {len(frame_ids)} cameras")
    if args.max_frames > 0:
        paths = paths[:args.max_frames]
        frame_ids = frame_ids[:args.max_frames]
        Ks = Ks[:args.max_frames]

    model = MoGeModel.from_pretrained(args.checkpoint).to(args.device).eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    timings, valid_ratios = [], []
    for index, path in enumerate(tqdm(paths, desc="MoGe-3")):
        target = out / f"{frame_ids[index]:06d}.npz"
        if target.is_file():
            continue
        image = torch.from_numpy(imageio.imread(path)[..., :3].copy()).to(args.device)
        image = image.permute(2, 0, 1).float() / 255.0
        H, W = image.shape[-2:]
        fov_x = math.degrees(2.0 * math.atan(W / (2.0 * float(Ks[index, 0, 0]))))
        tic = time.time()
        with torch.inference_mode():
            pred = model.infer(image, fov_x=fov_x, refine_steps=args.refine_steps,
                               force_projection=True, apply_mask=False, use_fp16=True)
        timings.append(time.time() - tic)
        valid = pred["mask"].bool() & torch.isfinite(pred["depth"]) & (pred["depth"] > 0)
        valid_ratios.append(float(valid.float().mean()))
        np.savez_compressed(
            target,
            depth=pred["depth"].float().cpu().numpy(),
            points=pred["points"].float().cpu().numpy(),
            normal=pred["normal"].float().cpu().numpy(),
            valid=valid.cpu().numpy(),
            K_normalized=pred["intrinsics"].float().cpu().numpy(),
            K_input=Ks[index], frame_id=np.int32(frame_ids[index]), fov_x=np.float32(fov_x),
        )
        del image, pred
        torch.cuda.empty_cache()
    report = {
        "model": args.checkpoint, "version": "MoGe-3", "frames": len(paths),
        "refine_steps": args.refine_steps, "mixed_precision": "official use_fp16",
        "runtime_s": time.time() - started,
        "mean_inference_s": float(np.mean(timings)) if timings else 0.0,
        "mean_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "camera_fov_source": "AMB3R K at training image resolution",
        "coordinate_system": "OpenCV camera: x right, y down, z forward; metric metres",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
