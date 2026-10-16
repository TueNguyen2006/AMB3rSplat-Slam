"""Build reproducible figures and a report from measured refinement artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    steps = (1000, 3000, 10000, 30000)
    results = {}
    for step in steps:
        path = root / f"step_{step:06d}/diagnostics/diagnostics.json"
        if path.exists():
            results[step] = json.loads(path.read_text())
    losses = [json.loads(line) for line in (root / "loss.jsonl").read_text().splitlines()]
    # Resume can repeat unsaved steps; prefer the last measured occurrence.
    losses = list({r["global_step"]:r for r in losses}.values())
    losses.sort(key=lambda r:r["global_step"])
    with (root / "benchmark.csv").open("w", newline="") as handle:
        fields = ["global_steps", "psnr", "ssim", "lpips", "coverage", "focus_psnr", "fixed_train_loss",
                  "train_loss_last500", "extra_training_seconds", "total_global_training_seconds", "iterations_per_second", "peak_allocated_gb",
                  "peak_reserved_gb", "nvidia_smi_peak_mib", "gaussians"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        table = []
        for step, data in results.items():
            path = root / f"step_{step:06d}/refinement_report.json"
            telemetry = json.loads(path.read_text()) if path.exists() else {}
            last = [r["loss"] for r in losses if step-500<r["global_step"]<=step]
            fixed_path=root / f"step_{step:06d}/fixed_training_loss.json"
            fixed_loss=json.loads(fixed_path.read_text())["average"]["total"] if fixed_path.exists() else None
            row = dict(global_steps=step, **{k:data["validation18"][k] for k in ("psnr","ssim","lpips","coverage")},
                       focus_psnr=data["focus169_188"]["psnr"],
                       fixed_train_loss=fixed_loss,
                       train_loss_last500=float(np.mean(last)) if last else None,
                       extra_training_seconds=telemetry.get("training_seconds", 0),
                       total_global_training_seconds=telemetry.get("training_seconds",0)+105.53,
                       iterations_per_second=(step-1000)/telemetry["training_seconds"] if telemetry.get("training_seconds") else 1000/105.53,
                       peak_allocated_gb=telemetry.get("peak_allocated_gb"),
                       peak_reserved_gb=telemetry.get("peak_reserved_gb"),
                       nvidia_smi_peak_mib=telemetry.get("nvidia_smi_peak_mib"),
                       gaussians=data["gaussians"])
            writer.writerow(row); table.append(row)
    (root / "benchmark.json").write_text(json.dumps(table, indent=2))
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot(list(results), [r["validation18"]["psnr"] for r in results.values()], "o-", label="18 original views")
    axes[0].plot(list(results), [r["focus169_188"]["psnr"] for r in results.values()], "s-", label="Frames 169-188")
    axes[0].axhline(19.268, color="gray", linestyle="--", label="Offline 30k reference")
    axes[0].set(xlabel="Total global steps", ylabel="PSNR (dB)"); axes[0].legend()
    if losses:
        x = np.array([r["global_step"] for r in losses]); y=np.array([r["loss"] for r in losses])
        axes[1].plot(x,y,color="silver",alpha=.35,linewidth=.5)
        window=min(500,len(y))
        axes[1].plot(x[window-1:], np.convolve(y,np.ones(window)/window,mode="valid"), label=f"{window}-step mean")
    fixed_rows=[r for r in table if r["fixed_train_loss"] is not None]
    if fixed_rows:
        axes[1].plot([r["global_steps"] for r in fixed_rows],[r["fixed_train_loss"] for r in fixed_rows],"o-",label="All 136 fixed keyframes")
    axes[1].set(xlabel="Total global steps",ylabel="Fixed RGB + depth + normal loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(root / "psnr_loss.png", dpi=180); plt.close(fig)
    fig, axes=plt.subplots(4,1,figsize=(15,13),sharex=True)
    for step,data in results.items():
        rows=data["per_frame"]
        for axis,key in zip(axes,("psnr","ssim","lpips","coverage")):
            axis.plot([r["frame_id"] for r in rows],[r[key] for r in rows],label=f"{step} steps",linewidth=1)
            axis.set_ylabel(key); axis.axvspan(169,188,color="gray",alpha=.04)
    axes[0].legend(ncol=4); axes[-1].set_xlabel("Frame ID")
    fig.tight_layout(); fig.savefig(root / "per_frame_metrics.png",dpi=180); plt.close(fig)
    lines=["# Mapping-only Refinement Analysis", "", "## Protocol", "",
           "Continued the original full_quality 1,000-step checkpoint with fixed RGB, K, camera poses, masks and depth/normal priors. No SLAM/SAM/MoGe inference or pose updates were performed.",
           "Gaussian means/scales/rotations/opacity/SH continue to train with the original MCMC strategy. Global steps are absolute, not additional steps. The MCMC counter includes 87 earlier local steps.",
           "", "The original 18-frame evaluation is preserved. It is not a held-out novel-view benchmark: 11 of those frames occur directly among the 136 mapping keyframes. Offline 30k has a different training exposure and optimization budget; use its 19.268 dB as a reference, not a controlled method ranking.",
           "", "## Benchmark", "", "| Global steps | PSNR | SSIM | LPIPS VGG | Coverage | Focus PSNR | Fixed train loss | Last-500 loss | Extra train seconds |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        loss="n/a" if row["train_loss_last500"] is None else f'{row["train_loss_last500"]:.5f}'
        fixed="n/a" if row["fixed_train_loss"] is None else f'{row["fixed_train_loss"]:.5f}'
        lines.append(f'| {row["global_steps"]} | {row["psnr"]:.3f} | {row["ssim"]:.4f} | {row["lpips"]:.4f} | {row["coverage"]:.3%} | {row["focus_psnr"]:.3f} | {fixed} | {loss} | {row["extra_training_seconds"]:.1f} |')
    lines += ["", "| Steps | Total global training seconds | Iterations/s | CUDA allocated GB | CUDA reserved GB | Whole GPU sampled peak MiB | Gaussian count |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f'| {row["global_steps"]} | {row["total_global_training_seconds"]:.1f} | {row["iterations_per_second"]:.2f} | {row["peak_allocated_gb"]} | {row["peak_reserved_gb"]} | {row["nvidia_smi_peak_mib"]} | {row["gaussians"]} |')
    lines += ["", "Training timing excludes checkpoint I/O, evaluation and process startup. Total adds the original 105.53 seconds of global optimization. The earlier source peak included SLAM and is not a mapping-only peak. Whole-GPU memory includes WSL/display overhead; CUDA values are allocator statistics."]
    lines += ["", "![PSNR and loss](psnr_loss.png)", "", "![Per-frame metrics](per_frame_metrics.png)", "", "## Camera and Geometry Findings", "",
              "The mapper stores the most recently inserted submap camera for a frame. AMB3R backend._frame_poses instead averages predictions over all overlapping submaps to export the trajectory. JointMappingSink.on_finish does not reconcile those two camera definitions. The source checkpoint has a maximum matrix-element discrepancy of 0.393814. This experiment preserves both definitions and reports mapping-camera renders separately as a diagnostic.",
              "", "Depth residual images use fixed AMB3R depth reprojected into the exported camera when a mapping keyframe exists; other frames use the cached MoGe metric depth prior. Neither is ground truth. Relative residual is clipped to 0.5 for display; raw arrays and validity are saved. Cross-submap render disagreement can include genuine occlusion or different surfaces, so it is evidence of conflict rather than a direct geometry accuracy metric."]
    for step,data in results.items():
        focus=[r for r in data["per_frame"] if 169<=r["frame_id"]<=188]
        kf=[r for r in focus if "mapping_pose_psnr" in r]
        missing=sum(r["missing_mse_contribution"] for r in focus)
        total=missing+sum(r["covered_mse_contribution"] for r in focus)
        overlap=[r for r in data["cross_submap_renders"] if r["overlap_pixels"]>=512]
        lines += ["",f"### {step} steps", "",
                  f'Focus mean coverage: {data["focus169_188"]["coverage"]:.2%}. Uncovered pixels account for {missing/max(total,1e-12):.2%} of focus squared RGB error.',
                  f'Mapping keyframes in focus: {len(kf)}. Mean PSNR at exported poses: {np.mean([r["psnr"] for r in kf]):.3f}; at original mapping poses: {np.mean([r["mapping_pose_psnr"] for r in kf]):.3f} dB.',
                  f'Cross-submap pairs with >=512 overlapping pixels: {len(overlap)}. Median fraction of overlap with >10% depth disagreement: {np.median([r["disagreement_10pct"] for r in overlap]):.2%}.' if overlap else "Insufficient cross-submap overlap for a depth disagreement estimate."]
        worst=min(focus,key=lambda r:r["psnr"])
        lines.append(f'Worst focus frame: {worst["frame_id"]}, PSNR {worst["psnr"]:.3f}, coverage {worst["coverage"]:.2%}.')
    lines += ["", "## Artifacts and Reproduction", "",
              "Each step directory contains checkpoints/final.pt, unchanged trajectory.npz, refinement_report.json, and diagnostics/ with 219-frame metrics, render/comparison videos, diagnostic panels, raw depth arrays and ownership statistics.",
              "", "```bash", "cd /home/tuenguyen/Eureka",
              "thirdparty/neural-harmonic-textures/.venv-amb3r-gs-slam/bin/python -m amb3r_gs_slam.refine_long",
              "# Recovery after interruption:",
              "thirdparty/neural-harmonic-textures/.venv-amb3r-gs-slam/bin/python -m amb3r_gs_slam.refine_long --resume outputs/amb3r_gs_slam/refinement/checkpoints/latest.pt",
              "```", "", "## Limits", "",
              "No ground-truth trajectory or 3D scan is available. ATE/RPE and absolute geometry accuracy cannot be measured. Gaussian count is capped at one million; no new SLAM observations are inserted. The original fixed learning rates and ongoing MCMC noise are preserved, so more steps do not guarantee convergence."]
    if len(results)!=4:
        lines += ["", "This report is partial: not all requested milestones have been evaluated yet."]
    (root / "REFINEMENT_ANALYSIS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(table,indent=2))


if __name__ == "__main__":
    main()
