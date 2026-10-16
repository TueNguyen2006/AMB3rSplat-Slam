from __future__ import annotations

from types import SimpleNamespace

import torch

from amb3r_slam.backend import HierarchicalBackend, Submap, sim3_identity
from amb3r_slam.tools.covis import AlikedCovis


class Sink:
    def __init__(self):
        self.commits = []

    def on_pose_commit(self, submap_id, old, new, submap, source="unknown", info=None):
        self.commits.append((submap_id, old.clone(), new.clone(), source, info))


class FrontEnd:
    def __init__(self):
        self.reanchored = None

    def reanchor(self, frames, poses):
        self.reanchored = (list(frames), poses.clone())


def test_external_pose_is_authoritative_and_reanchors_target_submap():
    backend = object.__new__(HierarchicalBackend)
    backend.device = torch.device("cpu")
    backend.stats = {}
    backend.event_sink = Sink()
    frontend = FrontEnd()
    pose = torch.eye(4)[None].repeat(2, 1, 1)
    points = torch.zeros(2, 4, 3)
    confidence = torch.ones(2, 4)
    submap0 = Submap(0, 2, pose, points, confidence, frames=[0, 1])
    submap1 = Submap(2, 4, pose, points, confidence, frames=[2, 3])
    S0, S1 = sim3_identity("cpu"), sim3_identity("cpu")
    S1[0] = 4.0
    backend._st = SimpleNamespace(
        S=[S0, S1], submaps=[submap0, submap1], fe=frontend,
        online=torch.eye(4)[None].repeat(4, 1, 1), external_priors={},
    )
    proposed = S0.clone()
    proposed[1], proposed[7] = 2.0, 1.1
    backend.apply_external_submap_pose(0, proposed, info={"accepted": True})
    assert torch.equal(backend._st.S[0], proposed)
    assert frontend.reanchored[0] == [0, 1]
    assert torch.allclose(frontend.reanchored[1][:, 1, 3], torch.full((2,), 2.0))
    assert backend.event_sink.commits[-1][3] == "gs_feedback"


def test_aliked_features_drop_human_keypoints_before_matching():
    class Detector:
        def __call__(self, _image):
            return {
                "keypoints": [torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])],
                "descriptors": [torch.eye(3)],
            }

    class Validity:
        def __call__(self, _frames, hw, device=None):
            valid = torch.ones((1, *hw), dtype=torch.bool, device=device)
            valid[0, hw[0] // 2, hw[1] // 2] = False
            return valid

    covis = object.__new__(AlikedCovis)
    covis.device = "cpu"
    covis.det = Detector()
    covis.validity_provider = Validity()
    covis._feat = {}
    image = torch.zeros(1, 3, 5, 5)
    keypoints, descriptors = covis.features(image, 0)
    assert len(keypoints) == 2
    assert descriptors.shape == (2, 3)

