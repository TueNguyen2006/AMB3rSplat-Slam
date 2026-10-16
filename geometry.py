"""Camera and Lie-group helpers using OpenCV c2w conventions."""

from __future__ import annotations

import torch
from torch import Tensor


def skew(v: Tensor) -> Tensor:
    out = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def so3_exp(w: Tensor) -> Tensor:
    theta = w.norm(dim=-1, keepdim=True)
    K = skew(w / theta.clamp_min(1e-12))
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(*w.shape[:-1], 3, 3)
    a = torch.sin(theta)[..., None]
    b = (1.0 - torch.cos(theta))[..., None]
    small = theta[..., None] < 1e-6
    R = eye + a * K + b * (K @ K)
    return torch.where(small, eye + skew(w), R)


def se3_exp(xi: Tensor) -> Tensor:
    """Exponential map for [translation, axis-angle] local camera updates."""
    t, w = xi[..., :3], xi[..., 3:]
    T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = so3_exp(w)
    T[..., :3, 3] = t
    return T


def invert_pose(T: Tensor) -> Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = torch.eye(4, dtype=T.dtype, device=T.device).expand_as(T).clone()
    out[..., :3, :3] = R.transpose(-1, -2)
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", R.transpose(-1, -2), t)
    return out


def project_world(points: Tensor, c2w: Tensor, K: Tensor) -> tuple[Tensor, Tensor]:
    w2c = invert_pose(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    pix = cam @ K.T
    uv = pix[:, :2] / pix[:, 2:].clamp_min(1e-8)
    return uv, z


def unproject_depth(depth: Tensor, K: Tensor, c2w: Tensor) -> Tensor:
    H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    rays = torch.stack(
        ((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], torch.ones_like(x)),
        dim=-1,
    )
    cam = rays * depth[..., None]
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def rotation_angle(R: Tensor) -> Tensor:
    return torch.acos(((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1))


def validate_pose(T: Tensor, atol: float = 2e-4) -> None:
    if T.shape[-2:] != (4, 4) or not torch.isfinite(T).all():
        raise ValueError("pose must be finite (..., 4, 4)")
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    err = (T[..., :3, :3].transpose(-1, -2) @ T[..., :3, :3] - eye).abs().max()
    det = torch.linalg.det(T[..., :3, :3])
    if float(err) > atol or float((det - 1).abs().max()) > atol:
        raise ValueError(f"invalid rotation: orthogonality={float(err):.3g}")
