"""Core math for the Saxe-dynamics project: closed forms, datasets, and float64 gradient-descent simulators.

Conventions (used everywhere in this directory)
------------------------------------------------
Loss      L(W) = 1/2 E ||y - W_L ... W_1 x||^2  = 1/2 tr(Sxx W^T W) - tr(Syx^T W) + const,   W = W_L...W_1
Flow      dW_l/dt = -dL/dW_l ;  time t = (learning rate) x (number of full-batch GD steps), so tau = 1.
Moments   Sxx = E[x x^T],  Syx = E[y x^T] = U S V^T  (Saxe et al. 2014 eq. 3, with expectations instead of sums).
Mode      u_a(t) = u_a^T W(t) v_a   (strength of the network's map along target mode a).

Closed forms (Saxe, McClelland & Ganguli 2014, whitened inputs Sxx = I, balanced decoupled init):
    2 layers:  du/dt = 2 u (s - u)                      ->  u(t) = s e^{2st} / (e^{2st} - 1 + s/u0)     (eq. 12)
    L layers:  du/dt = L u^{2-2/L} (s - u)              (eq. 15, with N_l - 1 = L weight matrices)
"""
from __future__ import annotations

import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------------------------

def sigmoid_mode(t, s, u0):
    """Exact 2-layer mode strength (Saxe 2014 eq. 12). Vectorised over t (and s/u0 by broadcasting)."""
    t = np.asarray(t, dtype=np.float64)
    e = np.exp(np.clip(2 * s * t, -700, 700))
    return s * e / (e - 1.0 + s / u0)


def t_cross(s, u0, frac):
    """Time for the 2-layer balanced solution to go from u0 to frac*s (inverse of eq. 12, i.e. eq. 11)."""
    uf = frac * s
    return 1.0 / (2 * s) * np.log(uf * (s - u0) / (u0 * (s - uf)))


def t_half(s, u0):
    return t_cross(s, u0, 0.5)


def width_10_90(s):
    """Transition width t(90%) - t(10%) of the 2-layer sigmoid; independent of u0 (exactly ln(81)/(2s))."""
    return math.log(81.0) / (2 * s)


def deep_mode_rhs(u, s, L):
    return L * np.power(np.maximum(u, 0.0), 2.0 - 2.0 / L) * (s - u)


def deep_mode(t, s, u0, L, n_sub=40):
    """Balanced depth-L mode strength (Saxe 2014 eq. 15), integrated in log-u with RK4 on a fine grid.

    Returns u evaluated at the (sorted, starting at 0) times t. For L=2 this reproduces sigmoid_mode.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    z = math.log(u0)  # z = log u,  dz/dt = L u^{1-2/L} (s - u)

    def f(z):
        u = math.exp(z)
        return L * u ** (1.0 - 2.0 / L) * (s - u)

    tc = 0.0
    for i, ti in enumerate(t):
        dt_tot = ti - tc
        if dt_tot > 0:
            n = max(1, int(math.ceil(dt_tot * n_sub * max(s, 1.0) * L)))
            h = dt_tot / n
            for _ in range(n):
                k1 = f(z); k2 = f(z + h / 2 * k1); k3 = f(z + h / 2 * k2); k4 = f(z + h * k3)
                z += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tc = ti
        out[i] = math.exp(z)
    return out


def first_crossing(t, u, level):
    """Linearly interpolated first time u(t) >= level (nan if never)."""
    u = np.asarray(u)
    idx = np.argmax(u >= level)
    if u[idx] < level:
        return np.nan
    if idx == 0:
        return t[0]
    t0, t1, u0_, u1 = t[idx - 1], t[idx], u[idx - 1], u[idx]
    return t0 + (level - u0_) * (t1 - t0) / (u1 - u0_)


# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------

def target_from_singular_values(s, n_out, n_in, seed=0):
    """Random orthogonal U, V and Syx = U diag(s) V^T (n_out x n_in)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n_out, n_out)))
    V, _ = np.linalg.qr(rng.standard_normal((n_in, n_in)))
    S = np.zeros((n_out, n_in))
    S[: len(s), : len(s)] = np.diag(s)
    return U @ S @ V.T, U, np.asarray(s, dtype=np.float64), V


ITEMS = ["canary", "robin", "salmon", "sunfish", "oak", "pine", "rose", "daisy"]
CATEGORY = ["bird", "bird", "fish", "fish", "tree", "tree", "flower", "flower"]
# (feature name, items that have it). A hand-built hierarchy in the spirit of Saxe et al. 2019 (PNAS), Fig. 4.
FEATURES = [
    ("grows", ITEMS), ("is alive", ITEMS),
    ("can move", ITEMS[:4]), ("has skin", ITEMS[:4]),
    ("has roots", ITEMS[4:]), ("has leaves", ITEMS[4:]),
    ("can fly", ITEMS[:2]), ("has wings", ITEMS[:2]), ("has feathers", ITEMS[:2]), ("has a beak", ITEMS[:2]),
    ("can swim", ITEMS[2:4]), ("has gills", ITEMS[2:4]), ("has scales", ITEMS[2:4]), ("has fins", ITEMS[2:4]),
    ("has bark", ITEMS[4:6]), ("is tall", ITEMS[4:6]),
    ("has petals", ITEMS[6:]), ("smells sweet", ITEMS[6:]),
    ("is yellow", ["canary"]), ("sings", ["canary"]), ("has a red breast", ["robin"]), ("eats worms", ["robin"]),
    ("is pink inside", ["salmon"]), ("swims upstream", ["salmon"]), ("is round", ["sunfish"]), ("is bright", ["sunfish"]),
    ("has acorns", ["oak"]), ("lives for centuries", ["oak"]), ("has needles", ["pine"]), ("is evergreen", ["pine"]),
    ("has thorns", ["rose"]), ("is red", ["rose"]), ("is white", ["daisy"]), ("grows in lawns", ["daisy"]),
]


def semantic_dataset():
    """Items (one-hot, scaled so E[x x^T] = I) -> binary feature vectors.

    Returns dict with X (P x N1), Y (P x N3), Syx, Sxx, and the SVD (U, s, V) of Syx.
    """
    P = len(ITEMS)
    Y = np.zeros((P, len(FEATURES)))
    for j, (_, have) in enumerate(FEATURES):
        for it in have:
            Y[ITEMS.index(it), j] = 1.0
    X = math.sqrt(P) * np.eye(P)  # whitened: E[x x^T] = (1/P) sum_i P e_i e_i^T = I
    Sxx = X.T @ X / P
    Syx = Y.T @ X / P
    U, s, Vt = np.linalg.svd(Syx, full_matrices=False)
    # sign convention: make the largest-magnitude entry of each v positive... prefer "animals positive"
    for a in range(len(s)):
        k = np.argmax(np.abs(Vt[a]))
        if Vt[a, 0] < -1e-9 or (abs(Vt[a, 0]) < 1e-9 and Vt[a, k] < 0):
            Vt[a] *= -1; U[:, a] *= -1
    return dict(X=X, Y=Y, Sxx=Sxx, Syx=Syx, U=U, s=s, V=Vt.T, items=ITEMS, features=[f for f, _ in FEATURES])


# ----------------------------------------------------------------------------------------------
# Simulators (full-batch GD on population moments, float64, batched over a leading config axis)
# ----------------------------------------------------------------------------------------------

def gd_deep_linear(Ws, Sxx, Syx, lr, n_steps, record_every, U=None, V=None, record_W=False, record_layers=()):
    """Full-batch gradient descent on L(W) = 1/2 tr(Sxx W^T W) - tr(Syx^T W),  W = W_L ... W_1.

    Ws: list of tensors [W_1, ..., W_L], each (B, n_{l+1}, n_l)  (B = batch of independent configs)
    Sxx: (B, n0, n0) or (n0, n0); Syx: (B, nL, n0) or (nL, n0)
    Returns dict with t (R,), loss (R, B), modes (R, B, r) if U,V given, optionally W (R, B, nL, n0) and
    recorded layer matrices.
    """
    Ws = [w.clone() for w in Ws]
    L = len(Ws)
    rec_t, rec_loss, rec_modes, rec_W = [], [], [], []
    rec_layers = {l: [] for l in record_layers}
    if U is not None:
        U = torch.as_tensor(U); V = torch.as_tensor(V)

    def product(ws):
        P = ws[0]
        for w in ws[1:]:
            P = w @ P
        return P

    for step in range(n_steps + 1):
        W = product(Ws)
        if step % record_every == 0:
            loss = 0.5 * torch.einsum("bij,bjk,bik->b", W, Sxx.expand(W.shape[0], -1, -1), W) \
                - torch.einsum("bij,bij->b", Syx.expand_as(W), W)
            rec_t.append(step * lr)
            rec_loss.append(loss.clone())
            if U is not None:
                rec_modes.append(torch.einsum("...ia,bij,...ja->ba", U, W, V))
            if record_W:
                rec_W.append(W.clone())
            for l in record_layers:
                rec_layers[l].append(Ws[l].clone())
        if step == n_steps:
            break
        G = W @ Sxx - Syx  # dL/dW  (B, nL, n0)
        # dL/dW_l = (W_L..W_{l+1})^T G (W_{l-1}..W_1)^T
        left = [None] * L   # left[l]  = W_L ... W_{l+1}
        right = [None] * L  # right[l] = W_{l-1} ... W_1
        acc = None
        for l in range(L):
            right[l] = acc
            acc = Ws[l] if acc is None else Ws[l] @ acc
        acc = None
        for l in reversed(range(L)):
            left[l] = acc
            acc = Ws[l] if acc is None else acc @ Ws[l]
        grads = []
        for l in range(L):
            g = G
            if left[l] is not None:
                g = left[l].transpose(-1, -2) @ g
            if right[l] is not None:
                g = g @ right[l].transpose(-1, -2)
            grads.append(g)
        for l in range(L):
            Ws[l] -= lr * grads[l]
    out = dict(t=np.array(rec_t), loss=torch.stack(rec_loss).numpy())
    if U is not None:
        out["modes"] = torch.stack(rec_modes).numpy()
    if record_W:
        out["W"] = torch.stack(rec_W).numpy()
    for l in record_layers:
        out[f"W{l+1}"] = torch.stack(rec_layers[l]).numpy()
    return out


def decoupled_init(U, V, widths, u0, L, rng=None, imbalance=1.0):
    """Saxe's decoupled, balanced init: W_1 = R_1 D V^T, W_l = R_l D R_{l-1}^T, W_L = U D R_{L-1}^T.

    Every active mode starts with per-layer strength a0 = u0^{1/L} (times imbalance on W_1, divided on W_L
    for L = 2), so the end-to-end strength is u0. widths = [n0, n1, ..., nL].
    """
    rng = rng or np.random.default_rng(0)
    r = min(min(widths), U.shape[1], V.shape[1])
    Rs = [V] + [np.linalg.qr(rng.standard_normal((n, n)))[0] for n in widths[1:-1]] + [U]
    Ws = []
    a0 = u0 ** (1.0 / L)
    for l in range(L):
        scale = a0
        if L == 2:
            scale = a0 * (imbalance if l == 0 else 1.0 / imbalance)
        Ws.append(Rs[l + 1][:, :r] @ (np.eye(r) * scale) @ Rs[l][:, :r].T)
    return Ws


def gaussian_init(widths, sigma, rng):
    return [rng.standard_normal((widths[l + 1], widths[l])) * sigma for l in range(len(widths) - 1)]


def effective_u0_two_layer(W1, W2, U, V, r):
    """Balanced-equivalent initial strength for random 2-layer init: |(a+b)/2|^2 per mode.

    a_alpha = column alpha of W1 V (weights from input mode alpha), b_alpha = row alpha of U^T W2.
    Linearising the flow at the origin, a+b grows as e^{st} and a-b decays, so the trajectory becomes
    balanced with strength |(a0+b0)/2|^2 e^{2st} (see post, 'where it breaks').
    """
    A = W1 @ V            # (n1, n0): columns a_alpha
    B = (U.T @ W2).T      # (n1, n2): columns b_alpha
    return np.array([np.sum(((A[:, a] + B[:, a]) / 2) ** 2) for a in range(r)])
