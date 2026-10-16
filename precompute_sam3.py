#!/usr/bin/env python3
"""Precompute official SAM 3 video masks with the exact default prompt `Human`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Human")
    parser.add_argument("--fallback-prompt", default=None,
                        help="Explicit ablation only, e.g. person; never enabled silently")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offload-state", action="store_true", default=True)
    return parser.parse_args()


def masks_from_output(output) -> np.ndarray:
    masks = output.get("out_binary_masks")
    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return masks


def run_prompt(predictor, video: str, prompt: str, threshold: float, offload: bool,
               max_frames: int = 0):
    response = predictor.handle_request({
        "type": "start_session", "resource_path": video,
        "offload_video_to_cpu": True, "offload_state_to_cpu": offload,
    })
    session = response["session_id"]
    seed = predictor.handle_request({
        "type": "add_prompt", "session_id": session, "frame_index": 0,
        "text": prompt, "output_prob_thresh": threshold,
    })
    outputs = {0: seed["outputs"]}
    for item in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session,
        "output_prob_thresh": threshold,
        "max_frame_num_to_track": None if max_frames <= 0 else max_frames,
    }):
        outputs[int(item["frame_index"])] = item["outputs"]
    predictor.handle_request({"type": "close_session", "session_id": session})
    return outputs


def main():
    args = parse_args()
    if args.prompt != "Human":
        print(f"[ablation] primary prompt changed from 'Human' to {args.prompt!r}")
    out = Path(args.output)
    mask_dir, overlay_dir = out / "masks", out / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_predictor
    predictor = build_sam3_predictor(checkpoint_path=args.checkpoint, version="sam3")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    outputs = run_prompt(predictor, args.video, args.prompt, args.threshold,
                         args.offload_state, args.max_frames)
    detected = sum(int(masks_from_output(item).any()) for item in outputs.values())
    used_prompt = args.prompt
    if detected == 0 and args.fallback_prompt:
        print(f"[ablation] {args.prompt!r} detected nothing; explicitly retrying {args.fallback_prompt!r}")
        outputs = run_prompt(predictor, args.video, args.fallback_prompt,
                             args.threshold, args.offload_state, args.max_frames)
        used_prompt = args.fallback_prompt

    cap = cv2.VideoCapture(args.video)
    frames = 0
    coverages = []
    overlay_video = cv2.VideoWriter(
        str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS) or 10.0,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    while True:
        if args.max_frames > 0 and frames >= args.max_frames:
            break
        ok, bgr = cap.read()
        if not ok:
            break
        H, W = bgr.shape[:2]
        masks = masks_from_output(outputs.get(frames, {}))
        union = np.zeros((H, W), dtype=bool)
        for mask in masks:
            union |= cv2.resize(mask.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        Image.fromarray(union.astype(np.uint8) * 255).save(mask_dir / f"{frames:06d}.png")
        overlay = bgr.copy()
        overlay[union] = (0.35 * overlay[union] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(overlay_dir / f"{frames:06d}.jpg"), overlay)
        overlay_video.write(overlay)
        coverages.append(float(union.mean()))
        frames += 1
    cap.release()
    overlay_video.release()
    report = {
        "model": "official facebookresearch/sam3", "prompt_requested": args.prompt,
        "prompt_used": used_prompt, "fallback_is_ablation": used_prompt != args.prompt,
        "frames": frames, "detected_frames": int(sum(x > 0 for x in coverages)),
        "mean_coverage": float(np.mean(coverages)), "max_coverage": float(np.max(coverages)),
        "runtime_s": time.time() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "mask_convention": "255/True is Human; empty PNG is no detection",
    }
    predictor.shutdown()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
