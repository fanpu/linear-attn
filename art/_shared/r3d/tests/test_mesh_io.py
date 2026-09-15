import math, subprocess
import numpy as np
import torch
from PIL import Image
from r3d.mesh import marching_cubes, mesh_volume, is_watertight, write_stl, read_stl, tube_mesh, scale_to_mm
from r3d.io import save_png, glow_tonemap, write_film


def _ball(n=64):
    a = np.linspace(-1.5, 1.5, n)
    Z, Y, X = np.meshgrid(a, a, a, indexing="ij")
    return 1 - np.sqrt(X ** 2 + Y ** 2 + Z ** 2)


def test_marching_cubes_ball_volume_orientation_watertight():
    v, f = marching_cubes(_ball(), 0.0, (-1.5,) * 3, (1.5,) * 3)
    assert is_watertight(f)
    assert abs(mesh_volume(v, f) - 4 / 3 * math.pi) / (4 / 3 * math.pi) < 0.02


def test_marching_cubes_axis_order_and_box_capping():
    Z, Y, X = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 1, 11), np.linspace(0, 4, 41), indexing="ij")
    field = np.ones_like(X)                                         # solid fills a 4 x 1 x 2 box
    v, f = marching_cubes(field, 0.5, (0, 0, 0), (4, 1, 2))
    assert is_watertight(f)
    ext = v.max(0) - v.min(0)
    assert np.allclose(ext, [4 + 0.1, 1 + 0.1, 2 + 0.1], atol=1e-6)  # capped half a voxel outside
    assert mesh_volume(v, f) > 0


def test_stl_roundtrip_and_scale(tmp_path):
    v, f = marching_cubes(_ball(24), 0.0, (-1.5,) * 3, (1.5,) * 3)
    v = scale_to_mm(v, 50.0)
    assert np.isclose((v.max(0) - v.min(0)).max(), 50.0) and np.allclose(v.min(0), 0)
    p = tmp_path / "b.stl"
    write_stl(p, v, f)
    tri = read_stl(p)
    assert tri.shape == (len(f), 3, 3)
    assert np.allclose(tri, v[f].astype(np.float32))


def test_tube_mesh_volume():
    P = np.array([[0.0, 0, 0], [0, 0, 1], [0, 0, 2]])
    v, f = tube_mesh(P, 0.1, n_sides=64)
    assert is_watertight(f)
    poly = 0.5 * 64 * math.sin(2 * math.pi / 64) * 0.01
    assert abs(mesh_volume(v, f) - poly * 2) / (poly * 2) < 1e-6
    bent = np.array([[math.cos(t), math.sin(t), 0.3 * t] for t in np.linspace(0, 6, 200)])
    vb, fb = tube_mesh(bent, 0.05, n_sides=12)
    assert is_watertight(fb) and mesh_volume(vb, fb) > 0


def test_png_tonemap_and_film(tmp_path):
    img = torch.zeros(10, 12, 3); img[..., 0] = 2.0
    save_png(tmp_path / "a.png", img)
    arr = np.asarray(Image.open(tmp_path / "a.png"))
    assert arr.shape == (10, 12, 3) and arr[0, 0, 0] == 255 and arr[0, 0, 1] == 0
    assert np.isclose(float(glow_tonemap(torch.tensor(1.0), 2.0)), 1 - math.exp(-2))
    for i in range(3):
        save_png(tmp_path / f"f_{i:05d}.png", np.full((32, 32), i / 2))
    mp4, gif = tmp_path / "x.mp4", tmp_path / "x.gif"
    write_film(str(tmp_path / "f_%05d.png"), mp4, fps=10, gif=gif, gif_width=32)
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip()
    assert n == "3" and gif.exists()
