"""Quick analytic checks of the chart and the estimators (CPU, seconds)."""
import numpy as np
from common import *

# chart: exp map lands on S^3, is isometric at the centre, and matches the analytic singular values
t = np.random.default_rng(0).uniform(-HALF, HALF, (1000, 3))
x = expmap(t)
assert np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-14)
eps = 1e-6
J = np.stack([(expmap(t + eps * e) - expmap(t - eps * e)) / (2 * eps) for e in np.eye(3)], -1)
sv = np.linalg.svd(J, compute_uv=False)
_, s = stretch_singular_values(t)
assert np.allclose(sv.max(1), 1, atol=1e-8) and np.allclose(sv.min(1), s, atol=1e-8)
assert np.isclose(np.degrees(np.arccos(expmap(np.array([2 * HALF / 256, 0, 0])) @ BASE)), np.degrees(2 * HALF / 256))

# box counting: a plane has D = 2 exactly; boundary of white noise fills space (D = 3)
R = 128
z = np.arange(R)
f = np.broadcast_to(z[None, None, :] - 60.3 + 0 * z[:, None, None], (R, R, R)) + 0.2 * z[:, None, None] * 0
d, c = dim3(np.array(f, float), level=0.0)
assert abs(d - 2) < 1e-9, (d, c)
g = np.random.default_rng(1).normal(size=(R, R, R))
d, c = dim3(g)
assert abs(d - 3) < 0.01, (d, c)
# 2D: a line has 1, noise has 2
d, _ = dim2(np.array(np.broadcast_to(z[None, :] - 40.5, (R, R)), float), level=0.0)
assert abs(d - 1) < 1e-9
d, _ = dim2(np.random.default_rng(2).normal(size=(R, R)))
assert abs(d - 2) < 0.01
# oblique planes stay inside the cube; trilinear sampling reproduces a linear field
for pl in slice_planes():
    P = slice_points(pl, SLICE_NPX[1], 2 / 256)
    assert np.abs(P).max() < 1
    V = grid_coords(256) / HALF
    lin = V[:, None, None] * 1.0 + 2 * V[None, :, None] - V[None, None, :]
    got = sample_trilinear(lin, P)
    assert np.allclose(got, P[..., 0] + 2 * P[..., 1] - P[..., 2], atol=1e-9)
    got2 = sample_trilinear(lin[::2, ::2, ::2], P, stride=2)
    assert np.allclose(got2, P[..., 0] + 2 * P[..., 1] - P[..., 2], atol=1e-2)
print("ok")
