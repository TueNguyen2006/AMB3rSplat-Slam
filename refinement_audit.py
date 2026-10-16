"""CPU audit of immutable inputs and evaluation exposure in saved maps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .refine_long import ROOT, fixed_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m = p["mapper"]
    split = json.loads((ROOT / "outputs/amb3r_slam_nht/prepared/split.json").read_text())
    kfs = m["keyframes"]
    cameras = np.load(ROOT / "outputs/amb3r_gs_slam/full_quality/trajectory.npz")
    pose_errors = {int(i): float(np.abs(v["c2w"].numpy()-cameras["c2w"][int(i)]).max())
                   for i,v in kfs.items()}
    with args.checkpoint.open("rb") as f:
        checkpoint_hash = hashlib.file_digest(f, "sha256").hexdigest()
    result = dict(checkpoint=str(args.checkpoint), sha256=checkpoint_hash,
                  fixed_digest=fixed_digest(kfs), global_steps=m["stats"]["global_steps"],
                  mcmc_steps=m["step"], sh_degree=m["sh_degree"], keyframes=len(kfs),
                  validation18_mapping_intersection=sorted(set(kfs)&set(split["val_frame_ids"])),
                  guard_mapping_intersection=sorted(set(kfs)&set(split["excluded_frame_ids"])),
                  keyframe_pose_vs_trajectory_max=max(pose_errors.values()),
                  poses_finite=bool(np.isfinite(cameras["c2w"]).all()),
                  inverse_error=float(np.max(np.abs(cameras["c2w"]@cameras["w2c"]-np.eye(4)))),
                  gaussian_count=len(m["scene"]["splats"]["means"]),
                  learning_rates={key:state["param_groups"][0]["lr"] for key,state in m["optimizers"].items()})
    if args.compare:
        other = torch.load(args.compare, map_location="cpu", weights_only=False)
        result["fixed_inputs_identical"] = fixed_digest(other["mapper"]["keyframes"]) == result["fixed_digest"]
        result["poses_c2w_identical"] = torch.equal(p["poses_c2w"], other["poses_c2w"])
        assert result["fixed_inputs_identical"] and result["poses_c2w_identical"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
