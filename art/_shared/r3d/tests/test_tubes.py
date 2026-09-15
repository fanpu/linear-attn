import math
import xml.etree.ElementTree as ET
import torch
from r3d.camera import Camera
from r3d.tubes import sample_polyline, splat_spheres, splat_additive, visible_runs, write_svg

CAM = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=64, ortho_height=4.0)   # 16 px per unit


def test_sample_polyline_spacing_and_endpoints():
    P = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=torch.float64)
    pts, s = sample_polyline(P, 0.3)
    assert torch.allclose(pts[0], P[0]) and torch.allclose(pts[-1], P[-1])
    assert (pts[1:] - pts[:-1]).norm(dim=1).max() <= 0.3 + 1e-12
    assert (s[1:] >= s[:-1]).all() and s[-1].item() == 2.0


def test_single_sphere_area_depth_normal():
    hit = splat_spheres(torch.zeros(1, 3, dtype=torch.float64), 1.0, CAM)
    assert abs(hit["mask"].sum().item() - math.pi * 16 ** 2) / (math.pi * 16 ** 2) < 0.03
    assert abs(hit["depth"][32, 32].item() - 4.0) < 0.01
    assert torch.allclose(hit["normal"][32, 32], torch.tensor([1.0, 0, 0], dtype=torch.float64), atol=0.05)
    top = hit["normal"][17, 32]                                  # near the top edge the normal tilts to +z
    assert top[2] > 0.7


def test_nearer_sphere_wins():
    c = torch.tensor([[0.0, 0, 0], [1.0, 0.2, 0]], dtype=torch.float64)   # second is nearer the camera at +x
    hit = splat_spheres(c, 0.8, CAM, attrs=torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    assert hit["attr"][32, 32, 0].item() == 2.0
    assert hit["attr"][hit["mask"]].unique().tolist() == [1.0, 2.0]


def test_additive_conserves_mass_and_respects_depth():
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(100, 3, generator=g, dtype=torch.float64) - 0.5)
    acc = splat_additive(pts, CAM, sigma_px=1.0)
    assert abs(acc.sum().item() - 100) < 2
    wall = torch.full((64, 64), 4.0, dtype=torch.float64)       # opaque wall at x = 1, in front of every point
    assert splat_additive(pts, CAM, sigma_px=1.0, depth=wall).sum().item() == 0.0
    col = splat_additive(pts, CAM, color=torch.tensor([[1.0, 0, 0]] * 100, dtype=torch.float64))
    assert col.shape == (64, 64, 3) and col[..., 1].sum() == 0


def test_splat_spheres_is_chunk_invariant():
    g = torch.Generator().manual_seed(1)
    M = 300
    x = torch.rand(M, generator=g, dtype=torch.float64) * 3.0 - 1.0    # distinct depths = 5 - x
    y = (torch.rand(M, generator=g, dtype=torch.float64) - 0.5) * 3.0
    z = (torch.rand(M, generator=g, dtype=torch.float64) - 0.5) * 3.0
    centers = torch.stack([x, y, z], dim=1)
    radius = 0.1 + torch.rand(M, generator=g, dtype=torch.float64) * 0.2   # 0.1 - 0.3, overlapping on screen
    attrs = torch.arange(M, dtype=torch.float64)[:, None]
    big = splat_spheres(centers, radius, CAM, attrs=attrs, chunk=2 ** 22)
    small = splat_spheres(centers, radius, CAM, attrs=attrs, chunk=200)    # forces per=1 sphere per chunk
    assert torch.equal(big["mask"], small["mask"])
    assert torch.allclose(big["depth"], small["depth"], atol=1e-12)
    assert torch.allclose(big["normal"], small["normal"], atol=1e-12)
    assert torch.allclose(big["attr"], small["attr"], atol=1e-12)


def test_splat_additive_is_chunk_invariant():
    g = torch.Generator().manual_seed(2)
    M = 500
    pts = (torch.rand(M, 3, generator=g, dtype=torch.float64) - 0.5) * 2.0   # within CAM view
    color = torch.rand(M, 3, generator=g, dtype=torch.float64)
    depth_buf = torch.full((64, 64), 5.0, dtype=torch.float64)
    depth_buf[:32, :] = 3.0                                                 # occludes points on that half

    acc_big = splat_additive(pts, CAM, sigma_px=1.0, depth=depth_buf, chunk=2 ** 22)
    acc_small = splat_additive(pts, CAM, sigma_px=1.0, depth=depth_buf, chunk=50)
    assert torch.allclose(acc_big, acc_small, atol=1e-10)

    col_big = splat_additive(pts, CAM, sigma_px=1.0, color=color, chunk=2 ** 22)
    col_small = splat_additive(pts, CAM, sigma_px=1.0, color=color, chunk=50)
    assert torch.allclose(col_big, col_small, atol=1e-10)


def test_visible_runs_and_svg(tmp_path):
    hit = splat_spheres(torch.zeros(1, 3, dtype=torch.float64), 0.5, CAM)
    line, _ = sample_polyline(torch.tensor([[-1.0, -1.5, 0], [-1.0, 1.5, 0]], dtype=torch.float64), 0.01)
    runs = visible_runs(line, CAM, hit["depth"], eps=1e-6)       # line at x = -1 passes behind the sphere
    assert len(runs) == 2
    front, _ = sample_polyline(torch.tensor([[1.0, -1.5, 0], [1.0, 1.5, 0]], dtype=torch.float64), 0.01)
    assert len(visible_runs(front, CAM, hit["depth"], eps=1e-6)) == 1
    p = tmp_path / "x.svg"
    write_svg(p, runs, 64, 64, background="#ffffff")
    root = ET.parse(p).getroot()
    assert len([e for e in root.iter() if e.tag.endswith("polyline")]) == 2
