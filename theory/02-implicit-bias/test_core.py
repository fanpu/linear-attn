"""Small checks of the core math.  Run: ../.venv/bin/python -m pytest -q test_core.py  (or python test_core.py)."""
import numpy as np
from core import (hero_2d, geometry_2d, svm, soudry_wtilde, flow_logtime, q_fun, q_alpha_min, basis_pursuit,
                  min_l2_interp, margin, angle)


def test_q_derivative_is_asinh():
    z = np.linspace(-30, 30, 2001)
    dq = np.gradient(q_fun(z), z)
    assert np.abs(dq[5:-5] - np.arcsinh(z[5:-5] / 2)).max() < 1e-3


def test_q_alpha_limits():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 30)); w = np.zeros(30); w[:3] = [1.5, -2, 1]; y = X @ w
    assert np.linalg.norm(q_alpha_min(X, y, 100.0)[0] - min_l2_interp(X, y)) < 1e-6
    assert np.linalg.norm(q_alpha_min(X, y, 1e-5)[0] - basis_pursuit(X, y)) < 1e-2


def test_soudry_wtilde_matches_logtime_flow():
    X, y = hero_2d()
    Z = y[:, None] * X
    wh, al = svm(X, y)
    wt, S = soudry_wtilde(X, y, wh, al)
    assert len(S) == 2
    s, W = flow_logtime(Z, np.log(1e40), s_eval=np.array([np.log(1e40)]))
    assert np.linalg.norm(W[-1] - wh * s[-1] - wt) < 1e-5


def test_geometry_dataset_directions():
    X, y = geometry_2d()
    deg = {k: np.degrees(np.arctan2(*svm(X, y, k)[0][::-1])) for k in ("l2", "linf", "l1")}
    assert abs(deg["l2"] - 63.435) < 0.01 and abs(deg["linf"] - 45) < 0.01 and abs(deg["l1"] - 90) < 0.01
    Z = y[:, None] * X
    w2 = svm(X, y, "l2")[0]
    assert abs(margin(Z, w2, "l2") - 1 / np.linalg.norm(w2)) < 1e-8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
