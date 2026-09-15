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


def test_tube_functions_project_on_the_points_device(monkeypatch):
    seen = []
    orig = Camera.project

    def spy(self, pts, *args, **kw):
        seen.append(kw.get("device", args[0] if args else None))
        return orig(self, pts, *args, **kw)

    monkeypatch.setattr(Camera, "project", spy)
    pts = torch.zeros(3, 3, dtype=torch.float64)
    splat_spheres(pts, 0.2, CAM)
    splat_additive(pts, CAM)
    visible_runs(pts, CAM, torch.full((64, 64), math.inf, dtype=torch.float64), eps=0.0)
    assert len(seen) == 3 and all(dv == pts.device for dv in seen)


def _splat_spheres_single_kernel(centers, radius, cam, attrs):
    """The pre-bucketing implementation (one global kernel), kept as the reference output."""
    from r3d.tubes import _offsets
    dt, dev = centers.dtype, centers.device
    H, W = cam.height, cam.width
    M = centers.shape[0]
    R = torch.as_tensor(radius, dtype=dt, device=dev).expand(M)
    pix, z = orig_project(cam, centers)
    pix, z = pix.to(dev, dt), z.to(dev, dt)
    ps = torch.full_like(z, cam.pixel_scale()) if cam.fov_deg is None else cam.pixel_scale(z)
    Rpx = R / ps
    f, r, u = (torch.tensor(v, dtype=dt, device=dev) for v in cam.basis())
    K = int(math.ceil(Rpx.max().item()))
    ox, oy = _offsets(K, dev)
    zbuf = torch.full((H * W,), math.inf, dtype=dt, device=dev)
    nbuf = torch.zeros(H * W, 3, dtype=dt, device=dev)
    abuf = torch.zeros(H * W, attrs.shape[1], dtype=dt, device=dev)
    col, row = pix[:, 0:1], pix[:, 1:2]
    pc, pr = torch.floor(col) + ox + 0.5, torch.floor(row) + oy + 0.5
    rp = Rpx[:, None]
    dx, dy = (pc - col) / rp, (pr - row) / rp
    rho2 = dx * dx + dy * dy
    inside = (rho2 <= 1) & (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H) & (z[:, None] > 0)
    nz = torch.sqrt((1 - rho2).clamp_min(0))
    depth = (z[:, None] - R[:, None] * nz)[inside]
    flat = (pr.long() * W + pc.long())[inside]
    nrm = (dx[..., None] * r - dy[..., None] * u - nz[..., None] * f)[inside]
    cmin = torch.full((H * W,), math.inf, dtype=dt, device=dev).scatter_reduce(0, flat, depth, "amin")
    win = depth == cmin[flat]
    fw = flat[win]
    zbuf[fw], nbuf[fw] = depth[win], nrm[win]
    sid = torch.arange(M, device=dev)[:, None].expand_as(inside)[inside]
    abuf[fw] = attrs[sid[win]].to(dt)
    return dict(depth=zbuf.reshape(H, W), normal=nbuf.reshape(H, W, 3), attr=abuf.reshape(H, W, -1),
                mask=torch.isfinite(zbuf).reshape(H, W))


orig_project = Camera.project


def _mixed_radius_scene(seed):
    g = torch.Generator().manual_seed(seed)
    M = 200
    x = torch.rand(M, generator=g, dtype=torch.float64) * 3.0 - 1.5      # distinct depths: no z-buffer ties
    yz = (torch.rand(M, 2, generator=g, dtype=torch.float64) - 0.5) * 3.2
    centers = torch.cat([x[:, None], yz], 1)
    radius = 0.01 * 150.0 ** torch.rand(M, generator=g, dtype=torch.float64)   # 0.01 - 1.5: sub-pixel to ~24 px
    return centers, radius, torch.arange(M, dtype=torch.float64)[:, None]


def test_bucketed_kernels_match_single_kernel_output(monkeypatch):
    import r3d.tubes as tubes
    Ks = []
    orig_offsets = tubes._offsets
    monkeypatch.setattr(tubes, "_offsets", lambda K, device: (Ks.append(K), orig_offsets(K, device))[1])
    persp = Camera(eye=(6, 0.3, 0.2), target=(0, 0, 0), width=64, height=64, fov_deg=40.0)
    for cam, seed in ((CAM, 3), (persp, 4)):
        centers, radius, attrs = _mixed_radius_scene(seed)
        ref = _splat_spheres_single_kernel(centers, radius, cam, attrs)
        Ks.clear()
        for chunk in (2 ** 22, 3000):
            got = splat_spheres(centers, radius, cam, attrs=attrs, chunk=chunk)
            assert torch.equal(got["mask"], ref["mask"])
            assert torch.equal(got["depth"], ref["depth"])
            assert torch.equal(got["normal"], ref["normal"])
            assert torch.equal(got["attr"], ref["attr"])
        assert len(set(Ks)) >= 3 and min(Ks) <= 2                  # several kernel sizes, small ones for small splats
        assert all(K & (K - 1) == 0 for K in Ks)                    # powers of two


def test_huge_splat_radius_raises():
    persp = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=64, fov_deg=40.0)
    centers = torch.tensor([[0.0, 0, 0], [0.5, 0.1, 0], [4.999, 0, 0]], dtype=torch.float64)  # last: 1 mm from the eye
    try:
        splat_spheres(centers, 0.05, persp)
    except ValueError as e:
        msg = str(e)
        assert "1 sample" in msg and "max_radius_px=256" in msg and "camera" in msg and "radius" in msg
    else:
        raise AssertionError("a near perspective point must raise instead of allocating a huge kernel")
    behind = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [7.0, 0, 0]], dtype=torch.float64)   # at and behind the eye
    assert splat_spheres(behind, 0.05, persp)["mask"].any()                                # culled, no error
    try:
        splat_spheres(centers[:2], 0.05, persp, max_radius_px=0.5)                          # both are ~0.9 px
    except ValueError as e:
        assert "2 sample" in str(e)
    else:
        raise AssertionError("max_radius_px must be honoured")
