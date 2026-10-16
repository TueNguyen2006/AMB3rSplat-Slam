# AMB3R-GS-SLAM

Incremental monocular reconstruction using the existing official AMB3R-SLAM +
VGGT-Omega pipeline, official SAM 3 human masks, official MoGe-3 metric priors,
and the NVIDIA NHT fork of gsplat with MCMC.

## What is coupled

The online state flow is:

```text
AMB3R submap -> static geometry -> coverage-aware Gaussian insertion
              -> local RGB/depth/normal mapping
              -> guarded SE(3) camera proposal
              -> AMB3R authoritative Sim(3) state + pose-graph factor
              -> Gaussian ownership transform -> next submap
```

`--feedback` and `--no-feedback` select the GS-to-SLAM branch without changing
AMB3R tracking, submap construction, loop retrieval, or pose-graph code.

## Existing installation

- Joint runner: `thirdparty/neural-harmonic-textures/.venv-amb3r-gs-slam`
- SAM 3: `thirdparty/sam3/.venv-sam3`
- MoGe-3: `thirdparty/MoGe/.venv-moge3`
- Omega checkpoint: `thirdparty/amb3r-slam/checkpoints/vggt_omega_1b_512.pt`
- SAM checkpoint: `thirdparty/sam3/checkpoints/sam3.pt`

The environments are separate from the earlier AMB3R, MASt3R and Fast-PGSR
environments.

## Reproduce

Precompute SAM 3 using the required prompt exactly:

```bash
cd /home/tuenguyen/Eureka
PYTHONPATH=$PWD/thirdparty/sam3:$PWD \
  thirdparty/sam3/.venv-sam3/bin/python -m amb3r_gs_slam.precompute_sam3 \
  --video outputs/frame_extractor_bandicam_dense/20261015_221912/keyframes.mp4 \
  --output outputs/amb3r_gs_slam/sam3 \
  --checkpoint thirdparty/sam3/checkpoints/sam3.pt \
  --prompt Human
```

The MoGe-3 cache already exists at `outputs/amb3r_gs_slam/moge3`. To rebuild it:

```bash
PYTHONPATH=$PWD/thirdparty/MoGe:$PWD \
  thirdparty/MoGe/.venv-moge3/bin/python -m amb3r_gs_slam.precompute_moge3 \
  --images outputs/amb3r_slam_omega_gsplat/images \
  --cameras outputs/amb3r_slam_omega_gsplat/cameras.npz \
  --output outputs/amb3r_gs_slam/moge3 --refine-steps 3
```

Run the integrated reconstruction:

```bash
./scripts/run_amb3r_gs_slam.sh \
  --output outputs/amb3r_gs_slam/full_quality \
  --max-gaussians 1000000 --global-steps 1000
```

Resume Gaussian-only refinement from the final or an interrupted-run checkpoint:

```bash
./scripts/run_amb3r_gs_slam.sh \
  --output outputs/amb3r_gs_slam/resumed \
  --resume outputs/amb3r_gs_slam/full_quality/checkpoints/final.pt \
  --mapping-only --global-steps 1000
```

`latest.pt` is written atomically after each submap and retained if a run is
interrupted. It is removed after `final.pt` has been written successfully.

Rebuild the machine-readable ablation table after evaluating all runs:

```bash
thirdparty/neural-harmonic-textures/.venv-amb3r-gs-slam/bin/python \
  -m amb3r_gs_slam.summarize_results \
  --root outputs/amb3r_gs_slam
```

Evaluate on the established 18-view split and open the native GPU viewer:

```bash
./scripts/evaluate_amb3r_gs_slam.sh outputs/amb3r_gs_slam/full_quality
./scripts/view_amb3r_gs_slam.sh outputs/amb3r_gs_slam/full_quality
```

Viewer controls are printed by the existing native D3D12/OpenGL viewer. Press
`S` to save a screenshot.

## Outputs

- `trajectory.npz`: final `c2w`, `w2c`, per-frame `K`, image size, frame IDs,
  keyframes and submap Sim(3).
- `checkpoints/latest.pt`: temporary atomic recovery checkpoint after every submap.
- `checkpoints/final.pt`: scene, MCMC state, optimizer state and poses.
- `gaussians/final_gaussians_splat.ply`: canonical SH3 3DGS PLY.
- `joint_events.jsonl`: every insertion, feedback decision and state commit.
- `render.mp4`, `comparison.mp4`, `renders/`: visual diagnostics.
- `evaluation/metrics.json`: PSNR, SSIM, normalized VGG LPIPS and rendering FPS.

## Important limits

SAM masks are applied to ALIKED keypoints, geometric overlap/alignment,
MoGe scale support, Gaussian insertion, and all mapping losses. VGGT-Omega still
receives the unmodified RGB image, so human pixels can indirectly affect its
learned feature inference. Loop candidate retrieval also sees RGB, but every
accepted geometric constraint is verified with the static mask.

The provided 18-view evaluation is transductive: AMB3R saw all 219 images while
building geometry. It is comparable to the prior offline result on the same
split, but it is not a strict unseen-view benchmark.
