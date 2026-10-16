"""Measure the unchanged training objective over every fixed mapping keyframe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .gaussian_mapper import IncrementalGaussianMapper, _depth_normal
from .masking import safe_masked_mean
from .refine_long import ROOT
from .state import SharedReconstructionState


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    payload=torch.load(args.run / "checkpoints/final.pt",map_location="cpu",weights_only=False)
    payload["mapper"]["optimizers"]={}
    mapper=IncrementalGaussianMapper(ROOT / "thirdparty/neural-harmonic-textures/gsplat",
                                     SharedReconstructionState(args.output.parent))
    mapper.load_state_dict(payload["mapper"])
    del payload
    rows=[]
    for fid,item in sorted(mapper.keyframes.items()):
        rgb,alpha,depth,_=mapper.render(item["c2w"],item["K"],item["rgb"].shape[:2])
        static=item["static"].bool()
        photo=safe_masked_mean((rgb-item["rgb"]).abs().mean(-1),static.float())
        valid=static&(item["depth"]>1e-6)&(alpha[...,0]>.05)
        geo=safe_masked_mean((depth.clamp_min(1e-6).log()-item["depth"].clamp_min(1e-6).log()).abs(),valid.float())
        md=normal=torch.tensor(0.,device="cuda")
        if "moge_depth" in item:
            valid=static&item["moge_valid"].bool()&(item["moge_depth"]>1e-6)&(alpha[...,0]>.05)
            md=safe_masked_mean((depth.clamp_min(1e-6).log()-item["moge_depth"].clamp_min(1e-6).log()).abs(),valid.float())
            nr=1.-(_depth_normal(depth,item["K"])*item["moge_normal"]).sum(-1).abs().clamp(0,1)
            normal=safe_masked_mean(nr,valid.float())
        rows.append(dict(frame_id=fid,photo=float(photo),slam_depth=float(geo),
                         moge_depth=float(md),moge_normal=float(normal),
                         total=float(photo+.05*geo+.04*md+.01*normal)))
    result=dict(global_step=mapper.stats["global_steps"],keyframes=len(rows),
                average={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!="frame_id"},
                per_frame=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result["average"]))


if __name__=="__main__":
    main()
