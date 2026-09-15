import math
import torch
from r3d.camera import Camera, ray_box
from r3d.grid import sample_grid
from r3d.voxels import march_voxels, render_voxels, voxel_occluder


def test_single_voxel_exact_footprint_depth_and_normal():
    solid = torch.zeros(3, 3, 3, dtype=torch.bool); solid[1, 1, 1] = True
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=64, ortho_height=4.0)
    hit = render_voxels(solid.long(), (-1, -1, -1), (1, 1, 1), cam)
    assert hit["mask"].sum().item() == 256
    assert torch.allclose(hit["depth"][hit["mask"]].float(), torch.tensor(4.5))
    assert torch.allclose(hit["normal"][hit["mask"]].float(), torch.tensor([1.0, 0, 0]))
    assert (hit["label"][hit["mask"]] == 1).all() and (hit["label"][~hit["mask"]] == -1).all()


def test_matches_brute_force_on_random_rays():
    g = torch.Generator().manual_seed(0)
    solid = torch.rand(16, 16, 16, generator=g) < 0.03
    lo, hi = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    h = 2 / 15
    o = torch.randn(200, 3, generator=g, dtype=torch.float64) * 0.3 + torch.tensor([4.0, 3.0, 2.5], dtype=torch.float64)
    tgt = (torch.rand(200, 3, generator=g, dtype=torch.float64) * 2 - 1)
    d = tgt - o; d = d / d.norm(dim=-1, keepdim=True)
    t, n, cell = march_voxels(solid, lo, hi, o, d)
    blo, bhi = [-1 - h / 2] * 3, [1 + h / 2] * 3
    tn, tf = ray_box(o, d, blo, bhi)
    step = 2e-3
    for r in range(200):
        if tf[r] <= tn[r]:
            assert math.isinf(t[r]); continue
        ts = torch.arange(tn[r].item(), tf[r].item(), step, dtype=torch.float64)
        v = sample_grid(solid.double(), lo, hi, o[r] + ts[:, None] * d[r], mode="nearest") > 0.5
        if not v.any():
            assert math.isinf(t[r]); continue
        tb = ts[v.nonzero()[0, 0]].item()
        assert t[r].item() <= tb + 1e-9 and tb - t[r].item() < step + 1e-9
        p = o[r] + t[r] * d[r]
        ijk = torch.floor((p + 1e-7 * d[r] - torch.tensor(blo, dtype=torch.float64)) / h).long()
        assert cell[r].tolist() == ijk.tolist()
        assert abs(n[r].norm().item() - 1) < 1e-12 and (n[r] * d[r]).sum() < 0


def test_clip_drops_whole_cells_by_centre():
    solid = torch.ones(3, 3, 3, dtype=torch.bool)
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    t, n, _ = march_voxels(solid, (-1, -1, -1), (1, 1, 1), o, d, clip=[((0.5, 0, 0), (1, 0, 0))])
    assert abs(t.item() - 4.5) < 1e-9 and n[0].tolist() == [1.0, 0.0, 0.0]
    t, _, _ = march_voxels(solid, (-1, -1, -1), (1, 1, 1), o, d, clip=[((-0.1, 0, 0), (1, 0, 0))])
    assert abs(t.item() - 5.5) < 1e-9


def test_voxel_occluder():
    solid = torch.zeros(3, 3, 3, dtype=torch.bool); solid[1, 1, 1] = True
    occ = voxel_occluder(solid, (-1, -1, -1), (1, 1, 1))
    o = torch.tensor([[3.0, 0, 0]] * 2, dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64)
    assert occ(o, d, torch.tensor([10.0, 10.0], dtype=torch.float64)).tolist() == [True, False]
