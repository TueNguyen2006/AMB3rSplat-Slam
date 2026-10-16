"""CPU-only source image and camera-ownership diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .refine_long import ROOT


def main():
    out=ROOT / "outputs/amb3r_gs_slam/refinement"
    data=ROOT / "outputs/amb3r_slam_omega_gsplat"
    trajectory=np.load(ROOT / "outputs/amb3r_gs_slam/full_quality/trajectory.npz")
    payload=torch.load(ROOT / "outputs/amb3r_gs_slam/full_quality/checkpoints/final.pt",map_location="cpu",weights_only=False)
    kfs=payload["mapper"]["keyframes"]
    rows=[]
    for fid in range(219):
        rgb=np.asarray(Image.open(data / "images" / f"{fid:06d}.png").convert("RGB"),dtype=np.float32)/255
        row=dict(frame_id=fid,mean_brightness=float(rgb.mean()),
                 clipped_highlight_fraction=float((rgb.min(-1)>.98).mean()))
        if fid in kfs:
            a=kfs[fid]["c2w"].numpy(); b=trajectory["c2w"][fid]
            angle=np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))
            row.update(mapping_submap=int(kfs[fid]["submap"]),
                       translation_difference=float(np.linalg.norm(a[:3,3]-b[:3,3])),
                       rotation_difference_deg=float(np.rad2deg(angle)))
        rows.append(row)
    (out / "source_image_pose_diagnostics.json").write_text(json.dumps(rows,indent=2))
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    axes[0].plot([r["frame_id"] for r in rows],[r["clipped_highlight_fraction"] for r in rows])
    axes[0].set_ylabel("Clipped RGB fraction")
    k=[r for r in rows if "translation_difference" in r]
    axes[1].plot([r["frame_id"] for r in k],[r["translation_difference"] for r in k],".-")
    axes[1].set_ylabel("Translation difference")
    axes[2].plot([r["frame_id"] for r in k],[r["rotation_difference_deg"] for r in k],".-")
    axes[2].set_ylabel("Rotation difference (deg)"); axes[2].set_xlabel("Frame")
    for ax in axes: ax.axvspan(169,188,color="gray",alpha=.15)
    fig.suptitle("Fixed source: mapping camera versus exported averaged camera")
    fig.tight_layout(); fig.savefig(out / "source_camera_differences.png",dpi=180)
    print(json.dumps([r for r in rows if r["frame_id"] in (169,175,188)],indent=2))


if __name__=="__main__":
    main()
