import torch

from amb3r_gs_slam.refine_long import fixed_digest
from amb3r_gs_slam.refinement_diagnostics import warp_depth


def test_depth_reference_reprojection_and_empty_mask():
    K = torch.tensor([[4.,0.,2.],[0.,4.,2.],[0.,0.,1.]])
    item = {"depth":torch.full((5,5),2.), "K":K, "c2w":torch.eye(4),
            "static":torch.ones(5,5,dtype=torch.bool)}
    depth, valid = warp_depth(item, torch.eye(4), K)
    assert valid.all() and torch.equal(depth,item["depth"])
    shifted = torch.eye(4)
    shifted[2,3] = .5
    depth, valid = warp_depth(item,shifted,K)
    assert torch.allclose(depth[valid],torch.full_like(depth[valid],1.5))
    item["static"].zero_()
    depth, valid = warp_depth(item, shifted, K)
    assert not valid.any() and torch.isfinite(depth).all() and not depth.any()


def test_fixed_digest_detects_pose_and_prior_changes():
    state={7:{"c2w":torch.eye(4),"depth":torch.ones(2,2),"static":torch.ones(2,2,dtype=torch.bool)}}
    original=fixed_digest(state)
    state[7]["c2w"][0,3]=.1
    assert original!=fixed_digest(state)
    state[7]["c2w"][0,3]=0
    assert original==fixed_digest(state)
    state[7]["depth"][0,0]=2
    assert original!=fixed_digest(state)
