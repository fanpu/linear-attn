import torch
from r3d.shade import lambert, hemisphere_dirs, ambient_occlusion, hard_shadow

UP = torch.tensor([[0.0, 0, 1]], dtype=torch.float64)


def never(o, d, tmax):
    return torch.zeros(o.shape[0], dtype=torch.bool)


def wall_at_x0(o, d, tmax):                     # solid half-space x < 0
    t = -o[:, 0] / torch.where(d[:, 0] == 0, torch.full_like(d[:, 0], 1e-30), d[:, 0])
    return (d[:, 0] < 0) & (t >= 0) & (t <= tmax)


def test_lambert():
    n = torch.tensor([[0.0, 0, 1], [1.0, 0, 0], [0, 0, -1.0]], dtype=torch.float64)
    v = lambert(n, (0, 0, 2), ambient=0.25)
    assert torch.allclose(v, torch.tensor([1.0, 0.25, 0.25], dtype=torch.float64))


def test_hemisphere_dirs_are_cosine_weighted():
    d = hemisphere_dirs(4096, seed=3)
    assert torch.allclose(d.norm(dim=1), torch.ones(4096, dtype=torch.float64))
    assert (d[:, 2] > 0).all()
    assert abs(d[:, 2].mean().item() - 2 / 3) < 0.01
    assert abs(d[:, 0].mean().item()) < 0.01


def test_ao_open_corner_and_radius():
    p = torch.tensor([[0.0, 0, 0]], dtype=torch.float64)
    assert ambient_occlusion(p, UP, never, n_rays=64).item() == 1.0
    ao = ambient_occlusion(p, UP, wall_at_x0, n_rays=256, radius=10.0)
    assert abs(ao.item() - 0.5) < 0.05
    far = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)
    assert ambient_occlusion(far, UP, wall_at_x0, n_rays=256, radius=0.5).item() == 1.0


def test_ao_handles_arbitrary_normals():
    p = torch.tensor([[0.5, 0, 0]], dtype=torch.float64)
    n = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)       # facing away from the wall
    assert ambient_occlusion(p, n, wall_at_x0, n_rays=64, radius=10.0).item() == 1.0


def test_hard_shadow():
    p = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    n = torch.tensor([[0.0, 0, 1], [0, 0, -1.0]], dtype=torch.float64)
    assert hard_shadow(p, n, (-1, 0, 1), wall_at_x0).tolist() == [0.0, 0.0]   # blocked, and facing away
    assert hard_shadow(p, n, (1, 0, 1), wall_at_x0).tolist() == [1.0, 0.0]
