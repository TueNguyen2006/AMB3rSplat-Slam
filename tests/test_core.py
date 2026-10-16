from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from amb3r_gs_slam.geometry import invert_pose, project_world, unproject_depth, validate_pose
from amb3r_gs_slam.masking import StaticMaskProvider, safe_masked_mean
from amb3r_gs_slam.state import FrameState, SharedReconstructionState


def test_projection_roundtrip_and_inverse():
    height, width = 24, 32
    K = torch.tensor([[40.0, 0.0, 15.5], [0.0, 41.0, 11.5], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.4, -0.2, 1.3])
    depth = torch.linspace(1.0, 3.0, height * width).reshape(height, width)
    points = unproject_depth(depth, K, c2w)
    uv, z = project_world(points.reshape(-1, 3), c2w, K)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    expected = torch.stack((x, y), -1).reshape(-1, 2).float()
    assert torch.allclose(uv, expected, atol=2e-5)
    assert torch.allclose(z, depth.reshape(-1), atol=2e-6)
    assert torch.allclose(c2w @ invert_pose(c2w), torch.eye(4), atol=1e-6)
    validate_pose(c2w)


def test_human_mask_weighting_and_edge_cases(tmp_path: Path):
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[5:9, 7:12] = 255
    Image.fromarray(mask).save(tmp_path / "000000.png")
    provider = StaticMaskProvider(tmp_path, 1, dilation=0)
    static = provider([0], (16, 20))[0]
    assert int((~static).sum()) == 20
    confidence = torch.rand(16, 20)
    original = confidence.clone()
    effective = provider.effective_weight(confidence, 0)
    assert torch.equal(confidence, original)
    assert torch.count_nonzero(effective[~static]) == 0
    assert torch.isfinite(safe_masked_mean(confidence, torch.zeros_like(confidence)))
    empty = StaticMaskProvider(None, 1)([0], (16, 20))[0]
    assert bool(empty.all())


def test_shared_state_pose_validation(tmp_path: Path):
    shared = SharedReconstructionState(tmp_path)
    frame = FrameState(0, 0.0, "frame.png", torch.eye(3), torch.eye(4))
    shared.add_frame(frame)
    sim3 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    shared.add_submap(0, [0], sim3)
    out = shared.validate()
    assert out["frames"] == 1
    assert out["max_c2w_w2c_error"] < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_gaussian_sim3_ownership_transform(tmp_path: Path):
    from amb3r_gs_slam.gaussian_mapper import IncrementalGaussianMapper

    device = torch.device("cuda")
    height, width = 24, 32
    K = torch.tensor([[32.0, 0.0, 15.5], [0.0, 32.0, 11.5], [0.0, 0.0, 1.0]],
                     device=device)
    pose = torch.eye(4, device=device)
    depth = torch.full((height, width), 2.0, device=device)
    points = unproject_depth(depth, K, pose)
    rgb = torch.full((height, width, 3), 0.5, device=device)
    shared = SharedReconstructionState(tmp_path)
    mapper = IncrementalGaussianMapper(
        Path(__file__).parents[2] / "thirdparty/neural-harmonic-textures/gsplat",
        shared, max_gaussians=2_000,
    )
    mapper.insert(0, 0, points, rgb, torch.ones_like(depth),
                  torch.ones_like(depth, dtype=torch.bool), K, pose, depth, max_new=500)
    before_mean = mapper.params["means"].detach().clone()
    before_scale = mapper.params["scales"].detach().clone()
    half = np.pi / 4
    delta = torch.tensor([1.0, -2.0, 0.5, 0.0, 0.0, np.sin(half), np.cos(half), 1.25],
                         device=device)
    moved = mapper.transform_owned(0, delta)
    assert moved == 500
    from amb3r_slam.tools.geometry import quat_to_rmat
    rotation = quat_to_rmat(delta[3:7]).float()
    assert torch.allclose(mapper.params["means"],
                          1.25 * before_mean @ rotation.T + delta[:3].float(), atol=1e-5)
    assert torch.allclose(mapper.params["scales"], before_scale + np.log(1.25), atol=1e-6)
    expected_quat = torch.tensor([np.cos(half), 0.0, 0.0, np.sin(half)],
                                 device=device, dtype=torch.float32)
    assert torch.allclose(mapper.params["quats"], expected_quat.expand(500, 4), atol=1e-6)
    assert math.isfinite(mapper.mapping_step([0])["loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_optimizer_momentum_survives_incremental_insertion(tmp_path: Path):
    from amb3r_gs_slam.gaussian_mapper import IncrementalGaussianMapper

    device = torch.device("cuda")
    height, width = 16, 20
    K = torch.tensor([[24.0, 0.0, 9.5], [0.0, 24.0, 7.5], [0.0, 0.0, 1.0]],
                     device=device)
    pose = torch.eye(4, device=device)
    depth = torch.full((height, width), 2.0, device=device)
    points = unproject_depth(depth, K, pose)
    rgb = torch.rand(height, width, 3, device=device)
    mapper = IncrementalGaussianMapper(
        Path(__file__).parents[2] / "thirdparty/neural-harmonic-textures/gsplat",
        SharedReconstructionState(tmp_path), max_gaussians=1_000, coverage_alpha=1.1,
    )
    valid = torch.ones_like(depth, dtype=torch.bool)
    confidence = torch.ones_like(depth)
    mapper.insert(0, 0, points, rgb, confidence, valid, K, pose, depth, max_new=200)
    mapper.mapping_step([0])
    old_parameter = mapper.optimizers["means"].param_groups[0]["params"][0]
    old_momentum = mapper.optimizers["means"].state[old_parameter]["exp_avg"].clone()
    mapper.insert(0, 1, points + torch.tensor([0.0, 0.0, 0.1], device=device),
                  rgb, confidence, valid, K, pose, depth, max_new=100)
    new_parameter = mapper.optimizers["means"].param_groups[0]["params"][0]
    new_momentum = mapper.optimizers["means"].state[new_parameter]["exp_avg"]
    assert len(new_momentum) == 300
    assert torch.equal(new_momentum[:200], old_momentum)
    assert torch.count_nonzero(new_momentum[200:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_final_sh_promotion_preserves_dc_and_adds_degree_three(tmp_path: Path):
    from amb3r_gs_slam.gaussian_mapper import IncrementalGaussianMapper

    mapper = IncrementalGaussianMapper(
        Path(__file__).parents[2] / "thirdparty/neural-harmonic-textures/gsplat",
        SharedReconstructionState(tmp_path), max_gaussians=128,
    )
    device = torch.device("cuda")
    h, w = 8, 8
    K = torch.tensor([[12.0, 0.0, 3.5], [0.0, 12.0, 3.5], [0.0, 0.0, 1.0]], device=device)
    pose = torch.eye(4, device=device)
    depth = torch.ones(h, w, device=device)
    points = unproject_depth(depth, K, pose)
    rgb = torch.rand(h, w, 3, device=device)
    mapper.insert(0, 0, points, rgb, torch.ones_like(depth),
                  torch.ones_like(depth, dtype=torch.bool), K, pose, depth, max_new=64)
    dc = mapper.params["sh0"].detach().clone()
    assert mapper.promote_sh_degree(3)
    assert mapper.params["shN"].shape == (64, 15, 3)
    assert torch.equal(mapper.params["sh0"], dc)

