"""The chart for Invariant Tori: energy level set H = h of two-player RPS replicator learning -> R^3.

State.  The C integrator (replicator_c.py) uses logits L = (u1, u2, v1, v2) with
u_k = log(x_k / x_0), v_k = log(y_k / y_0), k = 1, 2 (0 = rock, 1 = paper, 2 = scissors).

Step 1, zero-mean coordinates.  Per player the zero-mean logit z = log x - mean(log x) lives in the
plane {z : sum z = 0}.  With the orthonormal basis of that plane
    e_a = (-1, 1, 0) / sqrt 2,   e_b = (-1, -1, 2) / sqrt 6
we set w = (e_a . z, e_b . z), and u = (w_x, w_y) in R^4.  Since e_a, e_b are orthogonal to (1,1,1),
w = E (0, u1, u2) without removing the mean first.

Step 2, energy.  H = -1/3 sum log x_i - 1/3 sum log y_i = lse(E^T w_x) + lse(E^T w_y), because
-1/3 sum log softmax(z)_i = lse(z) - mean(z) and mean(z) = 0.  H is strictly convex on R^4 with
minimum 2 log 3 at u = 0, so t -> H(t q) is strictly increasing for t > 0 on every ray.

Step 3, radial map.  q = u / |u| in S^3; on a level set H = h > 2 log 3 this is a homeomorphism.

Step 4, stereographic projection from a unit pole p:  X = B^T q / (1 - p.q), with B (4x3) an
orthonormal basis of p-perp.  Inverse: q = (2 B X + (|X|^2 - 1) p) / (|X|^2 + 1).

Inverse chart: X -> q -> solve H(t q) = h for t (bisection, then Newton polish) -> u = t q
-> zero-mean logits -> strategies (softmax).
"""
import numpy as np

LOG3 = np.log(3.0)
E = np.array([[-1.0, 1.0, 0.0], [-1.0, -1.0, 2.0]]) / np.array([[np.sqrt(2.0)], [np.sqrt(6.0)]])


# ----------------------------------------------------------------------------- coordinates
def logits_to_u(L):
    """C-integrator logits (..., 4) -> zero-mean orthonormal coordinates u (..., 4)."""
    L = np.asarray(L, np.float64)
    zx = np.concatenate([np.zeros(L.shape[:-1] + (1,)), L[..., 0:2]], -1)
    zy = np.concatenate([np.zeros(L.shape[:-1] + (1,)), L[..., 2:4]], -1)
    return np.concatenate([zx @ E.T, zy @ E.T], -1)


def u_to_logits(u):
    zx = u[..., 0:2] @ E; zy = u[..., 2:4] @ E
    return np.stack([zx[..., 1] - zx[..., 0], zx[..., 2] - zx[..., 0],
                     zy[..., 1] - zy[..., 0], zy[..., 2] - zy[..., 0]], -1)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)


def _lse(z):
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))[..., 0]


def u_to_strategies(u):
    """u (..., 4) -> x (..., 3), y (..., 3)."""
    return _softmax(u[..., 0:2] @ E), _softmax(u[..., 2:4] @ E)


def strategies_to_u(x, y):
    lx = np.log(x); ly = np.log(y)
    return np.concatenate([lx @ E.T, ly @ E.T], -1)


def logits_to_strategies(L):
    return u_to_strategies(logits_to_u(L))


def energy_u(u):
    return _lse(u[..., 0:2] @ E) + _lse(u[..., 2:4] @ E)


def energy_logits(L):
    return energy_u(logits_to_u(L))


def denergy_dt(q, t):
    """d/dt H(t q) = q . grad H(t q)."""
    x, y = u_to_strategies(t[..., None] * q)
    return (q[..., 0:2] * (x @ E.T)).sum(-1) + (q[..., 2:4] * (y @ E.T)).sum(-1)


# ----------------------------------------------------------------------------- radial map
def to_sphere(u):
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def radius_for_level(q, h, n_bisect=64, n_newton=3):
    """t > 0 with H(t q) = h for unit q (..., 4).  H(tq) >= t/sqrt(6) (zero-mean z of norm r has
    max_i z_i >= r/sqrt 6, lse >= max, and |w_x| + |w_y| >= 1), so [0, sqrt(6) h] brackets the root."""
    q = np.asarray(q, np.float64)
    lo = np.zeros(q.shape[:-1]); hi = np.full(q.shape[:-1], np.sqrt(6.0) * h)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        up = energy_u(mid[..., None] * q) > h
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    t = 0.5 * (lo + hi)
    for _ in range(n_newton):  # polish; derivative > 0 on t > 0 (strict convexity)
        f = energy_u(t[..., None] * q) - h
        d = denergy_dt(q, t)
        tn = t - f / d
        t = np.where(np.isfinite(tn) & (np.abs(tn - t) < 1e-6 * np.maximum(1, t)), tn, t)
    return t


# ----------------------------------------------------------------------------- stereographic
def pole_basis(p):
    """Orthonormal basis (4, 3) of p-perp, deterministic (Householder-free QR of [p | I])."""
    p = np.asarray(p, np.float64); p = p / np.linalg.norm(p)
    Q, _ = np.linalg.qr(np.column_stack([p, np.eye(4)]))
    return Q[:, 1:4]  # Q[:, 0] = +-p, so the remaining columns span p-perp


def stereo(q, p, B):
    return (q @ B) / (1.0 - q @ p)[..., None]


def inv_stereo(X, p, B):
    r2 = (X * X).sum(-1, keepdims=True)
    return (2.0 * (X @ B.T) + (r2 - 1.0) * p) / (r2 + 1.0)


def angle_to_pole(q, p):
    return np.arccos(np.clip(q @ p, -1.0, 1.0))


def stereo_scale(X):
    """Local length magnification of the chart, |dX| / |dq| = (1 + |X|^2) / 2."""
    return 0.5 * (1.0 + (X * X).sum(-1))


class Chart:
    """Forward: logits or strategies -> X in R^3.  Inverse: X -> strategies on H = h."""

    def __init__(self, pole, h=2.8):
        self.p = np.asarray(pole, np.float64) / np.linalg.norm(pole)
        self.B = pole_basis(self.p)
        self.h = float(h)

    def forward_u(self, u):
        return stereo(to_sphere(u), self.p, self.B)

    def forward_logits(self, L):
        return self.forward_u(logits_to_u(L))

    def forward_strategies(self, x, y):
        return self.forward_u(strategies_to_u(x, y))

    def inverse_u(self, X, h=None):
        h = self.h if h is None else h
        q = inv_stereo(np.asarray(X, np.float64), self.p, self.B)
        q = to_sphere(q)
        t = radius_for_level(q, h)
        return t[..., None] * q

    def inverse(self, X, h=None):
        return u_to_strategies(self.inverse_u(X, h))


# ----------------------------------------------------------------------------- section function
def section_g(x, y):
    """SAF Fig. 1 section function, 0-indexed: g = x_P - x_R + y_P - y_R."""
    return x[..., 1] - x[..., 0] + y[..., 1] - y[..., 0]


def rps_matrix(e):
    return np.array([[e, -1, 1], [1, e, -1], [-1, 1, e]], float)


def section_gdot(x, y, eps):
    """dg/dt under the replicator flow with A = A(eps), B = A(-eps)."""
    Ay = y @ rps_matrix(eps).T; Bx = x @ rps_matrix(-eps).T
    xd = x * (Ay - (x * Ay).sum(-1, keepdims=True))
    yd = y * (Bx - (y * Bx).sum(-1, keepdims=True))
    return xd[..., 1] - xd[..., 0] + yd[..., 1] - yd[..., 0]
