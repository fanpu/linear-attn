"""Sampling a regular grid. data[k, j, i] sits at world (lo + (i, j, k) * h); lo/hi are corner-voxel centres."""
import torch
import torch.nn.functional as F


def sample_grid(data, lo, hi, pts, mode="linear", padding="zeros"):
    squeeze = data.dim() == 3
    x = data[None] if squeeze else data
    if x.dtype != pts.dtype:
        x = x.to(pts.dtype)
    lo_ = torch.as_tensor(lo, dtype=pts.dtype, device=pts.device)
    hi_ = torch.as_tensor(hi, dtype=pts.dtype, device=pts.device)
    g = (pts - lo_) / (hi_ - lo_) * 2 - 1
    shp = pts.shape[:-1]
    out = F.grid_sample(x[None], g.reshape(1, -1, 1, 1, 3), mode="bilinear" if mode == "linear" else "nearest",
                        padding_mode=padding, align_corners=True)
    out = out.reshape(x.shape[0], -1).T.reshape(*shp, x.shape[0])
    return out[..., 0] if squeeze else out


def clip_keep(pts, clip):
    keep = torch.ones(pts.shape[:-1], dtype=torch.bool, device=pts.device)
    for point, normal in clip:
        p = torch.as_tensor(point, dtype=pts.dtype, device=pts.device)
        n = torch.as_tensor(normal, dtype=pts.dtype, device=pts.device)
        keep &= ((pts - p) * n).sum(-1) <= 0
    return keep
