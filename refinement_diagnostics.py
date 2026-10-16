"""Fixed-camera image/coverage and cross-submap diagnostics; no inference."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .evaluate import masked_psnr
from .gaussian_mapper import IncrementalGaussianMapper
from .masking import StaticMaskProvider
from .state import SharedReconstructionState

ROOT = Path(__file__).resolve().parents[1]


def colorize(values, valid, maximum, cmap="magma"):
    values = np.nan_to_num(values, nan=0, posinf=maximum, neginf=0)
    rgb = (plt.get_cmap(cmap)(np.clip(values / maximum, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def panel(array, title):
    image = Image.new("RGB", (array.shape[1], array.shape[0] + 26), "white")
    image.paste(Image.fromarray(array), (0, 26))
    ImageDraw.Draw(image).text((8, 6), title, fill="black")
    return np.asarray(image)


def warp_depth(item, c2w, K):
    """Z-buffer fixed source depth into the export camera for comparable residuals."""
    ref = item["depth"]
    h, w = ref.shape
    y, x = torch.meshgrid(torch.arange(h, device=ref.device), torch.arange(w, device=ref.device), indexing="ij")
    uv = torch.stack((x.float(), y.float(), torch.ones_like(ref)), -1)
    xyz = (uv @ torch.linalg.inv(item["K"]).T) * ref[..., None]
    transform = torch.linalg.inv(c2w) @ item["c2w"]
    xyz = xyz @ transform[:3,:3].T + transform[:3,3]
    uvz = xyz @ K.T
    u = (uvz[...,0] / uvz[...,2].clamp_min(1e-6)).round().long()
    v = (uvz[...,1] / uvz[...,2].clamp_min(1e-6)).round().long()
    valid = (item["static"].bool() & (ref>0) & (xyz[...,2]>0) &
             torch.isfinite(xyz).all(-1) & (u>=0)&(u<w)&(v>=0)&(v<h))
    target = torch.full((h*w,), float("inf"), device=ref.device)
    target.scatter_reduce_(0, (v*w+u)[valid], xyz[...,2][valid], reduce="amin")
    target = target.reshape(h,w)
    finite = torch.isfinite(target)
    return torch.where(finite, target, 0.), finite


@torch.no_grad()
def cross_geometry(mapper):
    """Project fixed keyframe priors across adjacent submaps (not ground truth)."""
    records = []
    frames = sorted(mapper.keyframes)
    for a, b in zip(frames, frames[1:]):
        source, target = mapper.keyframes[a], mapper.keyframes[b]
        if int(source["submap"]) == int(target["submap"]):
            continue
        depth = source["depth"]
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(0, h, 4, device="cuda"),
                              torch.arange(0, w, 4, device="cuda"), indexing="ij")
        z = depth[y, x]
        uv = torch.stack((x.float(), y.float(), torch.ones_like(z)), -1)
        xyz = (uv @ torch.linalg.inv(source["K"]).T) * z[..., None]
        relative = torch.linalg.inv(target["c2w"]) @ source["c2w"]
        xyz2 = xyz @ relative[:3, :3].T + relative[:3, 3]
        projected = xyz2 @ target["K"].T
        u = projected[..., 0] / projected[..., 2].clamp_min(1e-6)
        v = projected[..., 1] / projected[..., 2].clamp_min(1e-6)
        grid = torch.stack((2*u/(w-1)-1, 2*v/(h-1)-1), -1)[None]
        ref = F.grid_sample(target["depth"][None, None], grid, align_corners=True)[0, 0]
        static2 = F.grid_sample(target["static"].float()[None, None], grid,
                                mode="nearest", align_corners=True)[0, 0] > .5
        valid = ((z > 0) & source["static"][y, x].bool() & static2 & (ref > 0)
                 & (xyz2[..., 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        err = (xyz2[..., 2] - ref).abs() / ref.clamp_min(1e-6)
        # Exclude points behind the target surface as likely occluded.
        front = valid & (xyz2[..., 2] <= ref * 1.10)
        records.append(dict(source=a, target=b, source_submap=int(source["submap"]),
                            target_submap=int(target["submap"]), overlap=float(valid.float().mean()),
                            support=int(front.sum()),
                            median_relative_depth=float(err[front].median()) if front.any() else None,
                            inconsistent_10pct=float((err[front] > .10).float().mean()) if front.any() else None))
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    data = ROOT / "outputs/amb3r_slam_omega_gsplat"
    trajectory = np.load(args.run / "trajectory.npz")
    split = json.loads((ROOT / "outputs/amb3r_slam_nht/prepared/split.json").read_text())
    payload = torch.load(args.run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    payload["mapper"]["optimizers"] = {}
    mapper = IncrementalGaussianMapper(ROOT / "thirdparty/neural-harmonic-textures/gsplat",
                                       SharedReconstructionState(out), max_gaussians=1000000)
    mapper.load_state_dict(payload["mapper"])
    step = mapper.stats["global_steps"]
    del payload
    masks = StaticMaskProvider(ROOT / "outputs/amb3r_gs_slam/sam3/masks", 219, dilation=3)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    owners = mapper.scene.component_index
    submap_owner = torch.empty_like(owners)
    ownership = []
    for component, name in enumerate(mapper.scene.component_names):
        match = re.search(r"submap_(\d+)", str(name))
        sid = int(match.group(1)) if match else component
        submap_owner[owners == component] = sid
        ownership.append(dict(component=component, name=str(name), submap=sid,
                              count=int((owners == component).sum())))
    base_opacity = torch.sigmoid(mapper.params["opacities"])
    rows, overlaps = [], []
    video = imageio.get_writer(out / "render.mp4", fps=5, codec="libx264", quality=8)
    comparisons = imageio.get_writer(out / "comparison.mp4", fps=5, codec="libx264", quality=8)
    pose_differences = []
    for fid in range(len(trajectory["frame_ids"])):
        target_np = np.asarray(Image.open(data / "images" / f"{fid:06d}.png").convert("RGB")).copy()
        target = torch.from_numpy(target_np).cuda().float() / 255.
        h, w = target.shape[:2]
        c2w = torch.from_numpy(trajectory["c2w"][fid]).cuda().float()
        K = torch.from_numpy(trajectory["K"][fid]).cuda().float()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred, alpha, depth, info = mapper.render(c2w, K, (h, w))
        torch.cuda.synchronize()
        render_s = time.perf_counter() - start
        pred = pred.clamp(0, 1)
        static = masks([fid], (h, w), device="cuda")[0]
        visible = alpha[..., 0] > .02
        squared = (pred-target).square().mean(-1)
        pred_n, gt_n = pred.permute(2,0,1)[None], target.permute(2,0,1)[None]
        ssim.reset(); lpips.reset()
        row = dict(frame_id=fid, global_step=step,
                   psnr=masked_psnr(pred, target, torch.ones_like(static)),
                   static_psnr=masked_psnr(pred, target, static),
                   ssim=float(ssim(pred_n, gt_n)), lpips=float(lpips(pred_n, gt_n)),
                   coverage=float(visible.float().mean()), alpha_055=float((alpha[...,0]>.55).float().mean()),
                   mean_alpha=float(alpha.mean()), render_seconds=render_s,
                   missing_mse_contribution=float((squared * ~visible).mean()),
                   covered_mse_contribution=float((squared * visible).mean()),
                   mapping_keyframe=fid in mapper.keyframes)
        if fid in mapper.keyframes:
            item = mapper.keyframes[fid]
            ref, ref_valid = warp_depth(item, c2w, K)
            reference = "fixed AMB3R depth reprojected to export camera"
            pose_differences.append(float((c2w-item["c2w"]).abs().max()))
            train_pred, train_alpha, _, _ = mapper.render(item["c2w"], item["K"], (h,w))
            train_pred = train_pred.clamp(0,1)
            ssim.reset(); lpips.reset()
            row.update(mapping_pose_psnr=masked_psnr(train_pred, target, torch.ones_like(static)),
                       mapping_pose_ssim=float(ssim(train_pred.permute(2,0,1)[None], gt_n)),
                       mapping_pose_lpips=float(lpips(train_pred.permute(2,0,1)[None], gt_n)),
                       mapping_pose_coverage=float((train_alpha[...,0]>.02).float().mean()),
                       pose_translation_difference=float((c2w[:3,3]-item["c2w"][:3,3]).norm()))
            if 169<=fid<=188:
                imageio.imwrite(out / "frames" / f"{fid:06d}_pose_comparison.png",
                                np.concatenate([target_np, (pred.cpu().numpy()*255).astype(np.uint8),
                                                (train_pred.cpu().numpy()*255).astype(np.uint8)],1))
        else:
            with np.load(ROOT / "outputs/amb3r_gs_slam/moge3" / f"{fid:06d}.npz") as prior:
                ref = torch.from_numpy(prior["depth"]).cuda().float()
                ref_valid = torch.from_numpy(prior["valid"]).cuda().bool() & static & (ref>0)
            reference = "cached MoGe metric prior (not GT)"
        valid_depth = ref_valid & visible & torch.isfinite(ref) & torch.isfinite(depth)
        residual = (depth-ref).abs() / ref.clamp_min(1e-6)
        row.update(depth_reference=reference, depth_support=int(valid_depth.sum()),
                   depth_relative_median=float(residual[valid_depth].median()) if valid_depth.any() else None)
        radii = info["radii"]
        visible_g = (radii[0] > 0).all(-1) if radii.ndim == 3 else radii[0] > 0
        row["visible_gaussians"] = int(visible_g.sum())
        counts = [(int(s), int(((submap_owner == s) & visible_g & (base_opacity>.01)).sum()))
                  for s in torch.unique(submap_owner)]
        counts.sort(key=lambda x: x[1], reverse=True)
        row["visible_owners"] = counts
        pred_np = (pred.cpu().numpy()*255).astype(np.uint8)
        error_np = squared.sqrt().cpu().numpy()
        alpha_np = alpha[...,0].cpu().numpy()
        depth_np = depth.cpu().numpy()
        valid_np = valid_depth.cpu().numpy()
        row["missing_pixel_rmse"] = float(squared[~visible].mean().sqrt()) if (~visible).any() else 0.
        row["covered_pixel_rmse"] = float(squared[visible].mean().sqrt()) if visible.any() else None
        video.append_data(pred_np)
        comparisons.append_data(np.concatenate([target_np, pred_np], 1))
        if fid in range(169, 189) or fid in split["val_frame_ids"]:
            alpha_rgb = np.repeat((alpha_np*255).astype(np.uint8)[...,None],3,-1)
            depth_rgb = colorize(depth_np, visible.cpu().numpy(), 10., "viridis")
            residual_rgb = colorize(residual.cpu().numpy(), valid_np, .5)
            error_rgb = colorize(error_np, np.ones_like(valid_np), .5)
            grid = np.concatenate([
                np.concatenate([panel(target_np, f"GT {fid}"), panel(pred_np, f"Render {step} steps"), panel(alpha_rgb, "Alpha 0..1")],1),
                np.concatenate([panel(depth_rgb, "Depth 0..10 scene units"), panel(residual_rgb, "Relative depth residual 0..0.5"), panel(error_rgb, "RGB RMSE 0..0.5")],1)],0)
            imageio.imwrite(out / "frames" / f"{fid:06d}_diagnostic.png", grid)
            np.savez_compressed(out / "frames" / f"{fid:06d}_depth.npz",
                                depth=depth_np, reference=ref.cpu().numpy(),
                                reference_label=reference, valid=valid_np, alpha=alpha_np)
        if 169 <= fid <= 188:
            rendered_groups = []
            for sid, count in counts[:3]:
                if count == 0:
                    continue
                group = submap_owner == sid
                p = mapper.params
                r, a, _ = mapper.rasterization(
                    means=p["means"], quats=F.normalize(p["quats"], dim=-1),
                    scales=p["scales"].exp(), opacities=base_opacity*group,
                    colors=torch.cat([p["sh0"],p["shN"]],1), viewmats=torch.linalg.inv(c2w)[None],
                    Ks=K[None], width=w, height=h, packed=False, render_mode="RGB+ED", sh_degree=3)
                rendered_groups.append((sid, r[0,...,3], a[0,...,0]))
            for i, (sid, d1, a1) in enumerate(rendered_groups):
                for sid2, d2, a2 in rendered_groups[i+1:]:
                    overlap = (a1>.1)&(a2>.1)&static
                    relative = (d1-d2).abs()/torch.minimum(d1,d2).clamp_min(1e-6)
                    overlaps.append(dict(frame_id=fid, submaps=[sid,sid2], overlap_pixels=int(overlap.sum()),
                                         overlap_fraction=float(overlap.float().mean()),
                                         median_relative_depth=float(relative[overlap].median()) if overlap.any() else None,
                                         disagreement_10pct=float((relative[overlap]>.1).float().mean()) if overlap.any() else None))
        rows.append(row)
        if fid % 25 == 0:
            print(f"diagnostic {step}: {fid}/219", flush=True)
    video.close(); comparisons.close()
    val = [r for r in rows if r["frame_id"] in split["val_frame_ids"]]
    def average(rs):
        return {key:float(np.mean([r[key] for r in rs])) for key in
                ("psnr","static_psnr","ssim","lpips","coverage","mean_alpha","visible_gaussians")}
    result = dict(global_step=step, all_frames=average(rows), validation18=average(val),
                  focus169_188=average([r for r in rows if 169<=r["frame_id"]<=188]),
                  gaussians=len(owners), ownership=ownership, per_frame=rows, cross_submap_renders=overlaps,
                  fixed_geometry_cross_projection=cross_geometry(mapper),
                  max_keyframe_vs_export_pose_difference=max(pose_differences, default=0),
                  validation_frames_used_in_mapping=sorted(set(mapper.keyframes)&set(split["val_frame_ids"])),
                  protocol="Same 18 transductive views; checkpoint mapping keyframes may include validation RGB")
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["validation18"], indent=2))


if __name__ == "__main__":
    main()
