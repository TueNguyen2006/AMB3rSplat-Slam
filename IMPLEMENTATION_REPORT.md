# Implementation Report

## Integration points

The official AMB3R entry point remains `amb3r_slam.pipeline.AMB3R_SLAM.run` in
`thirdparty/amb3r-slam/amb3r_slam/pipeline.py`. The original frontend tracking,
hierarchical backend, long-context matching, loop verification and pose graph
remain authoritative.

Small optional hooks were added to:

- `frontend.py`: static validity in overlap tests.
- `tools/align.py`: validity-aware confidence and overlap alignment.
- `tools/covis.py`: discard human ALIKED keypoints before descriptor matching.
- `backend.py`: geometry prior, submap event sink, guarded external Sim(3)
  factors, authoritative commits, and PGO-to-map notifications.

The integration itself is isolated in `amb3r_gs_slam/`:

- `state.py`: versioned RGB/K/SE(3)/Sim(3), priors and Gaussian ownership.
- `masking.py`: cached human masks and numerically safe effective weights.
- `moge_adapter.py`: robust sequence-level metric gauge and rejection tests.
- `gaussian_mapper.py`: keyframe insertion, alpha/depth coverage, local/global
  MCMC mapping, SH, guarded pose refinement and ownership transforms.
- `amb3r_adapter.py`: synchronous SLAM-to-GS and GS-to-SLAM control flow.
- `run.py`: launcher, recovery checkpoints, telemetry and exports.

## Algorithms reused

- WaterSplat-SLAM `mast3r_slam/wt_gs_backend.py`: keyframe-driven insertion
  (`add_next_kf_render`), window mapping (`map`) and global refinement.
- DROID-Splat `src/gaussian_mapping.py`: ownership/index mapping,
  `reanchor_gaussians`, `rescale_gaussians`, pose feedback and alternating
  `mapping_step`. Its Gaussian implementation also informed coverage growth.
- Splat-SLAM `src/mapper.py` and `thirdparty/glorie_slam/depth_video.py`:
  pose/depth optimization with map deformation and guarded shared geometry.
- GSO-SLAM `src/GS/gaussian_mapper.cpp`: RGB/SSIM, normal and sparse-depth
  constraints in the mapping objective.
- SAM 3 `sam3/model/sam3_video_inference.py`: text-prompted video propagation;
  text prompts permit new detections during propagation.
- MoGe-3 `moge/model/v3.py::infer`: metric depth, pointmap, normal, validity and
  intrinsics with forced projection.
- NVIDIA NHT gsplat `gsplat/strategy/mcmc.py::MCMCStrategy`: official MCMC
  relocation, sampling and noise injection.

No underwater medium MLP, attenuation or backscattering component was copied.

## Feedback cycle

`JointMappingSink._feedback` optimizes a local SE(3) candidate against GS RGB,
AMB3R depth and MoGe depth/normal while masking humans. It requires independent
multi-view improvement, bounded translation/rotation, finite values and enough
support. Accepted updates call
`HierarchicalBackend.apply_external_submap_pose`, which updates `st.S`, adds a
pose-graph prior, reanchors the active frontend, increments state versions and
notifies the mapper. `on_pose_commit` then transforms owned means, scale and
quaternions and updates camera caches. Rejected candidates leave SLAM unchanged.

SH remains degree zero while Sim(3) updates can occur. After final PGO, it is
promoted to degree three and globally refined, avoiding an invalid unrotated
higher-order SH basis during map synchronization.

## Tests

`pytest -c amb3r_gs_slam/pytest.ini amb3r_gs_slam/tests` passes 9 tests covering:

- c2w/w2c and projection round trips;
- empty, human and all-zero weight masks without NaN;
- Sim(3) Gaussian covariance/rotation/scale transforms;
- Adam momentum preservation across incremental insertion;
- final SH3 promotion;
- ALIKED pre-match human filtering;
- authoritative external pose commits and frontend reanchor;
- a controlled GS residual correction committed into real backend state and
  synchronized back into the owned Gaussian map.

## End-to-end evidence

The full 219-frame run produced 27 submaps and a one-million-Gaussian SH3
checkpoint without OOM. The joint runner proposed 25 GS pose corrections and
accepted one. That update changed the authoritative AMB3R `st.S` for submap 5,
added an external pose-graph factor, reanchored the frontend, moved 25,000 owned
Gaussians, incremented the pose/map versions, and was observed by the following
submap event. Its combined residual changed from `0.177591` to `0.174741` with
97,717 independently supported geometry pixels.

The feedback-off/full comparison is deliberately reported without claiming a
quality gain: feedback changed PSNR by +0.0107 dB, SSIM by -0.00257 and LPIPS by
+0.00020. It demonstrates executable state feedback and synchronization, not a
statistically meaningful reconstruction improvement on this sequence.

## Known limits

- VGGT-Omega sees original RGB, as described in the README.
- The ALIKED custom CUDA operation is not built in the Python 3.11 joint env;
  the official PyTorch fallback runs and loop logic remains enabled, but slower.
- This sequence produced no verified loop closure. Final adjacent-edge PGO still
  ran and synchronized all changed submaps.
- MoGe provides a metric prior, not ground truth. The first strongly supported
  global gauge is locked; later inconsistent windows are rejected rather than
  changing each camera independently.
