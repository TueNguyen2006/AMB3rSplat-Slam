"""Collect completed AMB3R-GS-SLAM ablations into one machine-readable file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    ("incremental_only", "ablation_incremental_only"),
    ("sam", "ablation_sam"),
    ("moge_scale", "ablation_moge_scale"),
    ("moge_depth_normal", "ablation_no_feedback"),
    ("full_feedback", "full_quality"),
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for name, directory in RUNS:
        run_dir = args.root / directory
        report = load_json(run_dir / "run_report.json")
        evaluation = load_json(run_dir / "evaluation" / "metrics.json")
        metrics = evaluation["metrics"]
        rows.append(
            {
                "name": name,
                "directory": directory,
                "frames": report["frames"],
                "submaps": report["submaps"],
                "loops": report["loops"],
                "runtime_s": report["runtime_s"],
                "peak_vram_gb_torch": report["peak_vram_gb_torch"],
                "gaussians_in_checkpoint": evaluation["gaussians"],
                "gaussians_exported": report["gaussians"],
                "mask_mode": report["mask_mode"],
                "moge_mode": report["moge_mode"],
                "metric_scale": report["state"]["metric_scale"],
                "feedback_enabled": report["feedback_enabled"],
                "feedback_proposed": report["feedback_proposed"],
                "feedback_accepted": report["feedback_accepted"],
                "psnr": metrics["psnr"],
                "static_psnr": metrics["static_psnr"],
                "ssim": metrics["ssim"],
                "lpips_vgg_normalized": metrics["lpips_vgg_normalized"],
                "visible_fraction": metrics["visible_fraction"],
                "render_fps": evaluation["render_fps"],
            }
        )

    summary = {
        "protocol": "transductive 18-view evaluation",
        "strict_unseen": False,
        "offline_reference": {
            "name": "offline_3dgs_mcmc_sh_30000",
            "psnr": 19.2681,
            "ssim": 0.636885,
            "lpips_vgg_normalized": 0.309315,
            "gaussians": 1_000_000,
        },
        "ablations": rows,
    }
    output = args.output or args.root / "ABLATION_SUMMARY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
