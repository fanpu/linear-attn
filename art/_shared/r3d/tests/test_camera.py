import math
import numpy as np
import torch
from r3d.camera import Camera, orbit, turntable, stereo_pair, ray_box


def test_ortho_centre_ray_and_projection_roundtrip():
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=48, ortho_height=4.0)
    o, d = cam.rays()
    assert o.shape == (48, 64, 3)
    assert torch.allclose(d[0, 0], torch.tensor([-1.0, 0, 0]))
    pts = o.reshape(-1, 3).double() + 2.0 * d.reshape(-1, 3).double()
    pix, depth = cam.project(pts)
    rows, cols = torch.meshgrid(torch.arange(48) + 0.5, torch.arange(64) + 0.5, indexing="ij")
    assert torch.allclose(pix[:, 0], cols.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(pix[:, 1], rows.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(depth, torch.full_like(depth, 2.0), atol=1e-6)


def test_image_orientation_up_is_row_zero_right_is_last_col():
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=32, height=32, ortho_height=2.0)
    pix, _ = cam.project(torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.9, 0.0]]))
    # looking along -x with z up: +z is the top of the image, right = forward x up = +y
    assert pix[0, 1] < 4 and abs(pix[0, 0] - 16) < 1e-6
    assert pix[1, 0] > 28


def test_perspective_roundtrip_and_depth_conversion():
    cam = Camera(eye=(0, -6, 2), target=(0, 0, 0), width=40, height=30, fov_deg=35.0)
    o, d = cam.rays(dtype=torch.float64)
    t = torch.rand(30, 40, dtype=torch.float64) * 5 + 1
    pts = (o + t[..., None] * d).reshape(-1, 3)
    pix, depth = cam.project(pts)
    rows, cols = torch.meshgrid(torch.arange(30) + 0.5, torch.arange(40) + 0.5, indexing="ij")
    assert torch.allclose(pix[:, 0], cols.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(pix[:, 1], rows.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(cam.t_to_depth(t, d).reshape(-1), depth, atol=1e-9)
    assert torch.allclose(cam.depth_to_t(depth.reshape(30, 40), d), t, atol=1e-9)


def test_orbit_turntable_stereo():
    cam = orbit((1, 2, 3), 10.0, az_deg=90, el_deg=0, width=8, height=8)
    assert np.allclose(cam.eye, (1, 12, 3))
    frames = turntable(4, (0, 0, 0), 5.0, el_deg=30, width=8, height=8)
    assert len(frames) == 4 and np.allclose(frames[2].eye[:2], -np.asarray(frames[0].eye[:2]))
    left, right = stereo_pair(frames[0], 0.5)
    assert math.isclose(np.linalg.norm(np.subtract(right.eye, left.eye)), 0.5)


def test_ray_box_hits_and_misses():
    o = torch.tensor([[-5.0, 0, 0], [-5.0, 3, 0], [0.0, 0, 0]])
    d = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1]])
    tn, tf = ray_box(o, d, (-1, -1, -1), (1, 1, 1))
    assert torch.allclose(tn[0], torch.tensor(4.0)) and torch.allclose(tf[0], torch.tensor(6.0))
    assert tf[1] <= tn[1]
    assert tn[2] == 0 and torch.allclose(tf[2], torch.tensor(1.0))


def test_pixel_scale_perspective_needs_depth():
    cam = Camera(eye=(0, -6, 2), target=(0, 0, 0), width=40, height=30, fov_deg=35.0)
    try:
        cam.pixel_scale()
    except ValueError as e:
        assert "depth" in str(e)
    else:
        raise AssertionError("perspective pixel_scale() without depth must raise ValueError")
    assert math.isclose(cam.pixel_scale(2.0), 2 * 2.0 * math.tan(math.radians(17.5)) / 30)
    assert Camera(eye=(5, 0, 0), target=(0, 0, 0), height=50, ortho_height=4.0).pixel_scale() == 4.0 / 50


def test_project_builds_its_tensors_on_the_points_device():
    for cam in (Camera(eye=(5, 0, 0), target=(0, 0, 0)), Camera(eye=(5, 0, 0), target=(0, 0, 0), fov_deg=40.0)):
        pix, z = cam.project(torch.zeros(4, 3, device="meta"))          # meta: any CPU-built tensor would raise
        assert pix.device.type == "meta" and z.device.type == "meta" and pix.shape == (4, 2)
        pix, z = cam.project(torch.zeros(4, 3), device="meta")
        assert pix.device.type == "meta" and z.device.type == "meta"
        pix, z = cam.project([[0.0, 0.0, 0.0]])
        assert pix.device.type == "cpu" and pix.dtype == torch.float64
