"""CPU-only Gaussian ownership, scale and cross-owner overlap measurements."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    p=torch.load(args.run / "checkpoints/final.pt",map_location="cpu",weights_only=False)["mapper"]
    scene=p["scene"]; splats=scene["splats"]
    xyz=splats["means"].numpy()
    scales=np.exp(splats["scales"].numpy())
    opacity=torch.sigmoid(splats["opacities"]).numpy()
    component=scene["component_index"].numpy()
    submap=np.empty_like(component)
    for i,name in enumerate(scene["component_names"]):
        match=re.search(r"submap_(\d+)",str(name))
        submap[component==i]=int(match.group(1)) if match else i
    active=opacity>.01
    def quantiles(x):
        return dict(zip(("p10","median","p90","p99"),map(float,np.quantile(x,[.1,.5,.9,.99])))) if len(x) else {}
    groups=[]
    for sid in np.unique(submap):
        mask=(submap==sid)&active
        groups.append(dict(submap=int(sid),count=int(mask.sum()),max_axis_scale=quantiles(scales[mask].max(-1)),
                           median_axis_scale=quantiles(np.median(scales[mask],axis=-1))))
    sigma=np.median(scales,axis=-1)
    rng=np.random.default_rng(42)
    overlap=[]
    for a in range(19,25):
        first=np.flatnonzero((submap==a)&active)
        if len(first)>20000: first=rng.choice(first,20000,replace=False)
        for b in range(a+1,min(a+3,27)):
            second=np.flatnonzero((submap==b)&active)
            if not len(first) or not len(second): continue
            distance,index=cKDTree(xyz[second]).query(xyz[first],workers=2)
            ratio=distance/np.maximum(np.minimum(sigma[first],sigma[second[index]]),1e-8)
            overlap.append(dict(submaps=[a,b],sample_count=len(first),
                                distance=quantiles(distance),normalized_distance=quantiles(ratio),
                                fraction_within_min_median_sigma=float((ratio<1).mean())))
    result=dict(global_steps=p["stats"]["global_steps"],active_gaussians=int(active.sum()),
                total_gaussians=len(xyz),scale_max_axis=quantiles(scales[active].max(-1)),
                groups=groups,cross_owner_overlap=overlap,
                interpretation="Nearest-center overlap normalized by median axis scale is a proxy, not proof of duplicated surfaces.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("groups","cross_owner_overlap")}))


if __name__=="__main__":
    main()
