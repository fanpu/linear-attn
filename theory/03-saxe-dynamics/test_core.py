"""Small tests for the core math.  Run:  OMP_NUM_THREADS=4 .venv/bin/python 03-saxe-dynamics/test_core.py"""
import numpy as np
import torch

import saxe_core as sc


def test_sigmoid_solves_ode():
    s, u0 = 2.0, 1e-4
    t = np.linspace(0, 6, 60001)
    u = sc.sigmoid_mode(t, s, u0)
    du = np.gradient(u, t)
    assert np.max(np.abs(du - 2 * u * (s - u))[5:-5]) < 1e-5
    assert abs(sc.first_crossing(t, u, s / 2) - sc.t_half(s, u0)) < 1e-4
    assert abs((sc.first_crossing(t, u, 0.9 * s) - sc.first_crossing(t, u, 0.1 * s)) - sc.width_10_90(s)) < 2e-3


def test_deep_mode_matches_closed_form_for_L2():
    t = np.linspace(0, 5, 200)
    a = sc.deep_mode(t, 1.5, 1e-3, 2)
    b = sc.sigmoid_mode(t, 1.5, 1e-3)
    assert np.max(np.abs(a - b)) < 1e-6


def test_semantic_dataset_whitened():
    d = sc.semantic_dataset()
    assert np.allclose(d["Sxx"], np.eye(8))
    assert np.allclose(d["U"] @ np.diag(d["s"]) @ d["V"].T, d["Syx"])


def test_gd_decoupled_matches_analytic():
    s = [5.0, 2.0, 1.0, 0.5]
    Syx, U, s_, V = sc.target_from_singular_values(s, 6, 6, seed=1)
    u0, lr = 1e-4, 2e-4
    Ws = sc.decoupled_init(U, V, [6, 6, 6], u0, 2)
    Ws = [torch.tensor(w)[None] for w in Ws]
    out = sc.gd_deep_linear(Ws, torch.eye(6), torch.tensor(Syx), lr, 60000, 500, U=U[:, :4], V=V[:, :4])
    for a, sa in enumerate(s):
        pred = sc.sigmoid_mode(out["t"], sa, u0)
        err = np.max(np.abs(out["modes"][:, 0, a] - pred)) / sa
        assert err < 5e-3, (sa, err)


def test_gd_depth3_matches_ode():
    s = [3.0, 1.0]
    Syx, U, s_, V = sc.target_from_singular_values(s, 4, 4, seed=2)
    u0, lr = 1e-3, 2e-4
    Ws = [torch.tensor(w)[None] for w in sc.decoupled_init(U, V, [4, 4, 4, 4], u0, 3)]
    out = sc.gd_deep_linear(Ws, torch.eye(4), torch.tensor(Syx), lr, 40000, 1000, U=U[:, :2], V=V[:, :2])
    for a, sa in enumerate(s):
        pred = sc.deep_mode(out["t"], sa, u0, 3)
        err = np.max(np.abs(out["modes"][:, 0, a] - pred)) / sa
        assert err < 5e-3, (sa, err)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
