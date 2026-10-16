# Results

## Dataset and protocol

- 219 RGB frames at 512 x 384, `frame_stride=1`.
- Official AMB3R-SLAM with VGGT-Omega; 27 submaps, no verified loop closure.
- Evaluation uses the existing 18 frame IDs:
  `6,14,23,49,60,67,92,109,118,125,142,152,161,169,181,188,199,215`.
- This is transductive evaluation because AMB3R geometry saw all 219 frames.
  ATE/RPE and metric depth error cannot be reported for this private monocular
  video because it has no trajectory/depth ground truth.

## Full integrated run

Output: `outputs/amb3r_gs_slam/full_quality`

| Item | Result |
|---|---:|
| Frames / submaps / loops | 219 / 27 / 0 |
| Runtime | 305.05 s |
| SLAM + incremental mapping | 170.06 s, 1.288 input FPS |
| Global SH3 refinement | 1,000 steps, 105.53 s |
| Export and visualization | 28.21 s |
| Final Gaussian budget | 1,000,000; 970,333 exported above opacity threshold |
| Torch peak allocated VRAM | 7.89 GB |
| Observed total `nvidia-smi` memory | 9.83 GB during mapping |
| SAM human frames | 36 / 219 |
| Human pixels rejected at Gaussian insertion | 153,499 |
| Static-mask exclusions across queried stages | 2,601,236 pixel observations |
| Coverage-consistent candidate points skipped | 3,599,861 |
| GS pose proposals / accepted | 25 / 1 |
| Final PGO map synchronizations | 26 changed non-anchor submaps |

Official SAM 3 preprocessing took 65.36 s and peaked at 9.20 GB. Official
MoGe-3 preprocessing took 119.78 s, averaged 0.384 s/frame after loading, and
peaked at 3.26 GB.

MoGe locked one sequence scale of `3.3518557` from an eight-frame window with
log-MAD `0.0071216`. It did not fit a scale per camera. Later windows with high
dispersion or disagreement were rejected and logged.

The one accepted real GS-to-SLAM update was on submap 5:

- combined objective: `0.177594 -> 0.174741`;
- photometric residual: `0.158570 -> 0.155870`;
- geometric residual: `0.190215 -> 0.188716`;
- translation correction: `0.00409` scene units;
- rotation correction: `0.2042 deg`;
- independent geometry support: 97,717 pixels.

The update changed AMB3R `st.S`, was added as an external PGO factor, reanchored
the frontend, moved the owned Gaussian group, incremented state versions and was
used by later submaps. The other 24 candidates were rolled back by trust or
multi-view geometry gates.

## Image quality and ablations

Every incremental row uses the same 1,000 global steps, one-million-primitive
checkpoint budget, images, poses, seed and 18 evaluation views.

| Configuration | Runtime | Peak VRAM | PSNR | SSIM | LPIPS VGG | Coverage | Render FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior offline 3DGS + MCMC + SH, 30k | n/a | n/a | **19.268** | **0.6369** | **0.3093** | n/a | 217.0 |
| Incremental GS only | 242.12 s | 7.36 GB | 17.219 | **0.5204** | **0.5248** | **95.81%** | 83.85 |
| + SAM 3 static masking | 246.51 s | 7.36 GB | 17.171 | 0.5179 | 0.5256 | 95.55% | 83.42 |
| + MoGe metric scale | 274.01 s | 7.49 GB | 16.953 | 0.5132 | 0.5259 | 94.91% | 80.17 |
| + MoGe depth/normal, feedback off | 300.00 s | 7.83 GB | 16.898 | 0.5144 | 0.5279 | 95.62% | 83.03 |
| Full system, feedback on | 305.05 s | 7.89 GB | 16.909 | 0.5118 | 0.5281 | 95.07% | 79.41 |

The integrated model is not yet as converged as the offline 30k trainer. The
600k diagnostic run proved why a fixed low budget was invalid: frame 199 had
zero rendered coverage and frame 215 only 0.06%. Raising the budget and refining
for 1,000 steps changed those values to 99.91% and 89.35%, with PSNR 19.48 and
16.66 respectively.

Best full-system validation view is frame 92 at PSNR 25.23, SSIM 0.7978. Worst is frame 169
at PSNR 8.85 and 42.3% coverage. Remaining failures are blur/ghosting and poor
cross-submap geometry around frames 169-188, not an evaluator artifact.

SAM masking costs 0.048 dB overall PSNR but removes 153,499 human-derived
initialization points and 1.76 million queried pixel observations. That is an
intentional static-map tradeoff, and static-region metrics are recorded in each
`metrics.json`.

MoGe establishes a consistent metric gauge, but its scale and depth/normal
constraints reduce photometric quality on this sequence. Feedback improves
PSNR by only 0.0107 dB over feedback-off while SSIM and LPIPS become slightly
worse. The implementation therefore proves a real bidirectional state update,
but this run does not support a claim that GS feedback improved final quality.

## Artifacts

- Final checkpoint: `outputs/amb3r_gs_slam/full_quality/checkpoints/final.pt`
- Standard SH3 PLY: `outputs/amb3r_gs_slam/full_quality/gaussians/final_gaussians_splat.ply`
- Trajectory/K: `outputs/amb3r_gs_slam/full_quality/trajectory.npz`
- Event audit: `outputs/amb3r_gs_slam/full_quality/joint_events.jsonl`
- Metrics: `outputs/amb3r_gs_slam/full_quality/evaluation/metrics.json`
- Render videos: `outputs/amb3r_gs_slam/full_quality/render.mp4` and `comparison.mp4`
- SAM overlay: `outputs/amb3r_gs_slam/sam3/overlay.mp4`
- Machine-readable ablations: `outputs/amb3r_gs_slam/ABLATION_SUMMARY.json`

## Remaining work

The 30k offline baseline remains the quality reference and outperforms every
incremental model here. A longer mapping-only
refinement from the final checkpoint can reduce blur without rerunning SLAM,
but it does not provide additional GS-to-SLAM feedback because tracking has
already finished. Strict unseen evaluation requires a new split where the
AMB3R model, submap geometry and Gaussian insertion never receive test frames.
