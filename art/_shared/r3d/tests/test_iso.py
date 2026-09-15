import math
import torch
from r3d.camera import Camera
from r3d.iso import march_iso, iso_normals, render_iso, iso_occluder

LO, HI = (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)


def _sphere(n=65):
    a = torch.linspace(-1.5, 1.5, n, dtype=torch.float64)
    Z, Y, X = torch.meshgrid(a, a, a, indexing="ij")
    return 1 - torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)            # solid (>= 0) is the unit ball


def test_centre_ray_hits_unit_sphere():
    f = _sphere()
    o = torch.tensor([[5.0, 0, 0], [5.0, 0.6, 0.0], [5.0, 1.2, 0]], dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0]] * 3, dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.0, step=0.02)
    assert abs(t[0].item() - 4.0) < 5e-3
    assert abs(t[1].item() - (5 - math.sqrt(1 - 0.36))) < 5e-3
    assert math.isinf(t[2].item())


def test_normals_point_outward():
    f = _sphere()
    p = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0.6, 0.8, 0]], dtype=torch.float64)
    n = iso_normals(f, LO, HI, p, 0.0)
    assert torch.allclose(n, p, atol=0.03)


def test_clip_plane_makes_a_flat_cap_with_plane_normal():
    f = _sphere()
    clip = [((0.5, 0, 0), (1, 0, 0))]                          # remove x > 0.5
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.0, step=0.02, clip=clip)
    assert abs(t.item() - 4.5) < 1e-3
    n = iso_normals(f, LO, HI, o + t[:, None] * d, 0.0, clip=clip)
    assert torch.allclose(n, torch.tensor([[1.0, 0, 0]], dtype=torch.float64), atol=1e-6)


def test_solid_touching_the_box_gets_box_face_normal():
    f = torch.ones(9, 9, 9, dtype=torch.float64)             # everything solid
    o = torch.tensor([[0.0, 0, 5]], dtype=torch.float64); d = torch.tensor([[0.0, 0, -1.0]], dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.5, step=0.05)
    assert abs(t.item() - 3.5) < 1e-9
    n = iso_normals(f, LO, HI, o + t[:, None] * d, 0.5)
    assert torch.allclose(n, torch.tensor([[0.0, 0, 1.0]], dtype=torch.float64))


def test_render_iso_silhouette_area_and_depth():
    f = _sphere().float()
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=96, height=96, ortho_height=3.0)
    hit = render_iso(f, LO, HI, cam, 0.0)
    px_area = (3.0 / 96) ** 2
    assert abs(hit["mask"].sum().item() * px_area - math.pi) / math.pi < 0.03
    assert abs(hit["depth"][48, 48].item() - 4.0) < 0.02
    assert torch.isinf(hit["depth"][0, 0])


def test_occluder():
    occ = iso_occluder(_sphere(), LO, HI, 0.0, 0.02)
    o = torch.tensor([[3.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    assert occ(o, d, torch.tensor([10.0, 10.0], dtype=torch.float64)).tolist() == [True, False]
    assert occ(o, d, torch.tensor([1.0, 10.0], dtype=torch.float64)).tolist() == [False, False]
