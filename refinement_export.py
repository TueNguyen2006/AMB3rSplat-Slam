"""Export an existing refined checkpoint without changing its parameters."""
import argparse
from pathlib import Path

import torch

from .gaussian_mapper import IncrementalGaussianMapper
from .refine_long import ROOT
from .run import write_standard_3dgs_ply
from .state import SharedReconstructionState


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    output=args.run / "gaussians/final_gaussians_splat.ply"
    if output.exists():
        print(f"Already exported: {output}")
        return
    payload=torch.load(args.run / "checkpoints/final.pt",map_location="cpu",weights_only=False)
    payload["mapper"]["optimizers"]={}
    mapper=IncrementalGaussianMapper(ROOT / "thirdparty/neural-harmonic-textures/gsplat",
                                     SharedReconstructionState(args.run))
    mapper.load_state_dict(payload["mapper"])
    with torch.no_grad():
        count=write_standard_3dgs_ply(output,mapper)
    print(f"Exported {count} Gaussians: {output}")


if __name__=="__main__":
    main()
