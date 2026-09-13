"""NumPy float64 versions of the maps (fast on CPU for small state dims).

prod-k : GD on f(x) = 1/2 (x_1...x_k - 1)^2
bal-k  : the balanced invariant line x_i = u of the same GD map
logistic : x -> r x (1-x)
All functions are vectorised over a leading batch axis and take eta with shape (N,).
"""
import numpy as np

BIG = 1e6


def others(x):
    """prod_{j!=i} x_j, x: (N,k)."""
    N, k = x.shape
    left = np.ones_like(x)
    right = np.ones_like(x)
    for i in range(1, k):
        left[:, i] = left[:, i - 1] * x[:, i - 1]
    for i in range(k - 2, -1, -1):
        right[:, i] = right[:, i + 1] * x[:, i + 1]
    return left * right


def prod_step(x, eta):
    k = x.shape[1]
    if k == 2:
        a, b = x[:, 0], x[:, 1]
        r = (a * b - 1.0) * eta
        return np.stack([a - r * b, b - r * a], 1)
    if k == 4:
        a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        ab = a * b
        cd = c * d
        r = (ab * cd - 1.0) * eta
        return np.stack([a - r * b * cd, b - r * a * cd, c - r * ab * d, d - r * ab * c], 1)
    g = others(x)
    P = x[:, 0] * g[:, 0]
    return x - (eta * (P - 1.0))[:, None] * g


def prod_tangent(x, v, eta):
    """(I - eta H(x)) v for f = 1/2(P-1)^2."""
    N, k = x.shape
    g = others(x)
    P = x[:, 0] * g[:, 0]
    gv = (g * v).sum(1)
    d2v = np.zeros_like(x)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            t = v[:, j].copy()
            for l in range(k):
                if l != i and l != j:
                    t *= x[:, l]
            d2v[:, i] += t
    return v - eta[:, None] * (g * gv[:, None] + (P - 1.0)[:, None] * d2v)


def prod_jac(x, eta):
    """Jacobian I - eta H, (N,k,k)."""
    N, k = x.shape
    g = others(x)
    P = x[:, 0] * g[:, 0]
    H = g[:, :, None] * g[:, None, :]
    for i in range(k):
        for j in range(k):
            if i != j:
                t = np.ones(N)
                for l in range(k):
                    if l != i and l != j:
                        t = t * x[:, l]
                H[:, i, j] += (P - 1.0) * t
    return np.eye(k)[None] - eta[:, None, None] * H


def bal_step(u, eta, k):
    return u - eta * (u ** k - 1.0) * u ** (k - 1)


def bal_deriv(u, eta, k):
    return 1.0 - eta * ((2 * k - 1) * u ** (2 * k - 2) - (k - 1) * u ** (k - 2))


def bal_transverse(u, eta, k):
    return 1.0 - eta ** 2 * (u ** k - 1.0) ** 2 * u ** (2 * k - 4)


def logistic_step(x, r):
    return r * x * (1.0 - x)


def logistic_deriv(x, r):
    return r * (1.0 - 2.0 * x)
