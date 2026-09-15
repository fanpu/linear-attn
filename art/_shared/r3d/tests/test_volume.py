import math
import torch
from r3d.camera import Camera
from r3d.grid import sample_grid, clip_keep
from r3d.volume import TransferFunction, lut_tf, march_volume, render_volume, over

LO, HI = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)


def _linear_grid(D=7, H=6, W=5):
    z = torch.linspace(-1, 1, D); y = torch.linspace(-1, 1, H); x = torch.linspace(-1, 1, W)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    return (2 * X + 3 * Y - Z).double()


def test_sample_grid_is_exact_for_linear_fields():
    g = _linear_grid()
    p = torch.rand(100, 3, dtype=torch.float64) * 2 - 1
    v = sample_grid(g, LO, HI, p)
    assert torch.allclose(v, 2 * p[:, 0] + 3 * p[:, 1] - p[:, 2], atol=1e-10)
    vc = sample_grid(torch.stack([g, -g]), LO, HI, p)
    assert vc.shape == (100, 2) and torch.allclose(vc[:, 1], -v)


def test_sample_grid_nearest_returns_voxel_values():
    g = _linear_grid()
    p = torch.tensor([[-1.0, -1.0, -1.0], [0.49, 0.39, 0.34]], dtype=torch.float64)  # x step .5, y .4, z 1/3
    v = sample_grid(g, LO, HI, p, mode="nearest")
    assert torch.allclose(v, torch.tensor([2 * -1 + 3 * -1 + 1, 2 * 0.5 + 3 * 0.2 - 1 / 3], dtype=torch.float64), atol=1e-9)


def test_clip_keep():
    p = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]])
    assert clip_keep(p, [((0, 0, 0), (1, 0, 0))]).tolist() == [False, True]


def _const_tf(s):
    return lambda v: (torch.tensor([1.0, 0.5, 0.25], dtype=v.dtype).expand(*v.shape, 3), torch.full_like(v, s))


def test_constant_cube_matches_beer_lambert():
    data = torch.ones(8, 8, 8, dtype=torch.float64)
    o = torch.tensor([[5.0, 0.1, -0.2]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    rgb, a = march_volume(data, LO, HI, o, d, _const_tf(0.7), step=0.013)
    assert math.isclose(a.item(), 1 - math.exp(-1.4), rel_tol=1e-9)
    assert torch.allclose(rgb[0], torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64) * a[0])


def test_clip_plane_and_depth_limit_shorten_the_path():
    data = torch.ones(8, 8, 8, dtype=torch.float64)
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    _, a = march_volume(data, LO, HI, o, d, _const_tf(1.0), step=0.01, clip=[((0, 0, 0), (1, 0, 0))])
    assert math.isclose(a.item(), 1 - math.exp(-1.0), rel_tol=1e-3)
    _, a = march_volume(data, LO, HI, o, d, _const_tf(1.0), step=0.01, tmax=torch.tensor([4.5], dtype=torch.float64))
    assert math.isclose(a.item(), 1 - math.exp(-0.5), rel_tol=1e-3)


def test_lut_tf_labels_and_transfer_function():
    tf = lut_tf(torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]), torch.tensor([0.0, 5.0, 9.0]))
    rgb, s = tf(torch.tensor([2.0, 1.0, 0.0]))
    assert s.tolist() == [9.0, 5.0, 0.0] and rgb[0].tolist() == [0.0, 1.0, 0.0]
    t2 = TransferFunction("magma", 0.0, 2.0, opacity=lambda x: x, density=3.0)
    rgb, s = t2(torch.tensor([-1.0, 1.0, 5.0]))
    assert torch.allclose(s, torch.tensor([0.0, 1.5, 3.0])) and rgb.shape == (3, 3)


def test_render_volume_image_and_over():
    data = torch.ones(8, 8, 8)
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=32, height=32, ortho_height=4.0)
    rgb, a = render_volume(data, LO, HI, cam, _const_tf(50.0), step=0.02)
    assert rgb.shape == (32, 32, 3) and a.shape == (32, 32)
    assert a[16, 16] > 0.999 and a[0, 0] == 0          # centre hits the cube, corner misses it
    img = over(rgb, a, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(img[0, 0], torch.tensor([0.0, 0.0, 1.0]))
