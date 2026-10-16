from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from amb3r_gs_slam.amb3r_adapter import JointMappingSink
from amb3r_gs_slam.gaussian_mapper import IncrementalGaussianMapper
from amb3r_gs_slam.geometry import unproject_depth
from amb3r_gs_slam.masking import StaticMaskProvider
from amb3r_gs_slam.state import FrameState, SharedReconstructionState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_synthetic_gs_feedback_commits_to_slam_and_synchronizes_map(tmp_path: Path):
    from amb3r_slam.backend import HierarchicalBackend, Submap, sim3_identity

    device = torch.device("cuda")
    height, width = 64, 80
    K = torch.tensor([[75.0, 0.0, 39.5], [0.0, 75.0, 31.5], [0.0, 0.0, 1.0]],
                     device=device)
    true_pose = torch.eye(4, device=device)
    depth = torch.full((height, width), 2.0, device=device)
    points = unproject_depth(depth, K, true_pose)
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=device),
                          torch.linspace(0, 1, width, device=device), indexing="ij")
    color = torch.stack((x, y, 0.25 + 0.5 * x * y), dim=-1)

    shared = SharedReconstructionState(tmp_path)
    shared.add_frame(FrameState(0, 0.0, "", K.cpu(), true_pose.cpu()))
    shared.add_submap(0, [], sim3_identity("cpu"))
    shared.add_submap(1, [0], sim3_identity("cpu"))
    mapper = IncrementalGaussianMapper(
        Path(__file__).parents[2] / "thirdparty/neural-harmonic-textures/gsplat",
        shared, max_gaussians=6_000,
    )
    mapper.insert(1, 0, points, color, torch.ones_like(depth),
                  torch.ones_like(depth, dtype=torch.bool), K, true_pose, depth,
                  max_new=5_000)
    with torch.no_grad():
        observed_rgb, _, observed_depth, _ = mapper.render(true_pose, K, (height, width))
    perturbed = true_pose.clone()
    perturbed[2, 3] = 0.02
    mapper.keyframes[0].update(rgb=observed_rgb.detach(), depth=observed_depth.detach(),
                               c2w=perturbed)

    sink = JointMappingSink(shared, mapper, StaticMaskProvider(None, 1), K[None], 10.0,
                            feedback=True, feedback_iterations=80,
                            log_path=tmp_path / "events.jsonl")

    class FrontEnd:
        def reanchor(self, frames, poses):
            self.frames, self.poses = frames, poses

    local_poses = perturbed[None]
    sm0 = Submap(0, 0, torch.empty(0, 4, 4, device=device),
                 torch.empty(0, 1, 3, device=device), torch.empty(0, 1, device=device),
                 frames=[])
    sm1 = Submap(0, 1, local_poses, points.reshape(1, -1, 3),
                 torch.ones(1, height * width, device=device), frames=[0])
    backend = object.__new__(HierarchicalBackend)
    backend.device = device
    backend.stats = {}
    backend.event_sink = sink
    backend._st = SimpleNamespace(
        S=[sim3_identity(device), sim3_identity(device)], submaps=[sm0, sm1],
        fe=FrontEnd(), online=torch.eye(4)[None].repeat(1, 1, 1), external_priors={},
    )
    sink.backend = backend
    before_means = mapper.params["means"].detach().clone()
    sink._feedback(backend, 1, sm1, [0])

    assert len(backend.stats["external_pose_updates"]) == 1
    assert any(event["event"] == "feedback_accepted" for event in sink.events)
    assert any(event["event"] == "map_synchronized" for event in sink.events)
    assert shared.pose_version == 1
    assert shared.map_version > 1
    assert not torch.allclose(mapper.params["means"], before_means)
    assert mapper.score_pose(0, mapper.keyframes[0]["c2w"])["geometry"] < 0.02
    assert not torch.allclose(backend._st.S[1], sim3_identity(device))
    accepted = next(event for event in sink.events if event["event"] == "feedback_accepted")
    assert accepted["after"] < accepted["before"]
    assert accepted["geometry_after"] < accepted["geometry_before"]

