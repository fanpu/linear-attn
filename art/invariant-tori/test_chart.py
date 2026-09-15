"""Tests for the Invariant Tori chart.  Orbit tests read the cached orbits, so run
`compute_orbits.py all` and `compute_chart.py pole` first.

    OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_chart.py
"""
import glob
import json
import os

import numpy as np
import pytest

import chart as C

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
H0 = 2.8


def stored_orbits():
    need = [os.path.join(CACHE, 'orbits_eps0.npz'), os.path.join(CACHE, 'orbits_eps05.npz')]
    missing = [f for f in need if not os.path.exists(f)]
    assert not missing, f'run compute_orbits.py all first; missing {missing}'
    out = {}
    d = np.load(need[0]); out['eps0'] = d['L']
    d = np.load(need[1]); out['eps05_chaotic'] = d['chaotic_L']; out['eps05_regular'] = d['regular_L']
    files = sorted(glob.glob(os.path.join(CACHE, 'sweep', 'eps_*.npz')))
    assert len(files) == 26, 'run compute_orbits.py sweep first'
    for f in files:
        out['sweep_' + os.path.basename(f)[4:6]] = np.load(f)['L']
    return out


def test_energy_formula_matches_definition():
    rng = np.random.default_rng(1)
    L = 3 * rng.normal(size=(1000, 4))
    x, y = C.logits_to_strategies(L)
    H = -(np.log(x).sum(-1) + np.log(y).sum(-1)) / 3
    assert np.abs(C.energy_logits(L) - H).max() < 1e-12
    assert np.abs(C.u_to_logits(C.logits_to_u(L)) - L).max() < 1e-12
    assert abs(C.energy_u(np.zeros(4)) - 2 * np.log(3)) < 1e-15


def test_basis_and_stereo_are_exact():
    rng = np.random.default_rng(2)
    assert np.allclose(C.E @ C.E.T, np.eye(2), atol=1e-15) and np.allclose(C.E.sum(1), 0, atol=1e-15)
    p = C.to_sphere(rng.normal(size=4)); B = C.pole_basis(p)
    assert np.allclose(B.T @ B, np.eye(3), atol=1e-14) and np.abs(B.T @ p).max() < 1e-14
    q = C.to_sphere(rng.normal(size=(10000, 4)))
    q = q[q @ p < 0.99]
    assert np.abs(C.inv_stereo(C.stereo(q, p, B), p, B) - q).max() < 1e-12


def test_energy_strictly_increasing_along_rays():
    """H(t q) strictly increasing in t > 0 along 1e4 random rays, out to 1.5x the h = 2.8 level."""
    rng = np.random.default_rng(3)
    q = C.to_sphere(rng.normal(size=(10000, 4)))
    tstar = C.radius_for_level(q, H0)
    assert np.abs(C.energy_u(tstar[:, None] * q) - H0).max() < 1e-12
    s = np.linspace(0.0, 1.5, 601)[:, None] * tstar[None, :]          # (601, 1e4)
    Hs = C.energy_u(s[..., None] * q[None])
    dH = np.diff(Hs, axis=0)
    assert (dH > 0).all(), f'min increment {dH.min():.3e}'
    assert (C.denergy_dt(q[None], s[1:]) > 0).all()                   # analytic derivative, t > 0


def test_inverse_chart_round_trip_on_orbit_points():
    """logits -> X -> strategies (bisection on H(t q) = h) agrees to < 1e-9 on 1e4 orbit points.
    h is each point's own energy, so this isolates the chart from integrator drift; the h = 2.8
    round trip (which includes drift) is also bounded."""
    orbits = stored_orbits()
    p = np.array(json.load(open(os.path.join(CACHE, 'pole.json')))['pole'])
    ch = C.Chart(p, H0)
    rng = np.random.default_rng(4)
    pts = []
    for L in orbits.values():
        flat = L.reshape(-1, 4)
        pts.append(flat[rng.choice(len(flat), 10000 // len(orbits) + 1, replace=False)])
    P = np.concatenate(pts)[:10000]
    assert len(P) == 10000
    x, y = C.logits_to_strategies(P)
    X = ch.forward_logits(P)
    xr, yr = ch.inverse(X, h=C.energy_logits(P))
    err = max(np.abs(xr - x).max(), np.abs(yr - y).max())
    assert err < 1e-9, err
    xr2, yr2 = ch.inverse(X)
    assert max(np.abs(xr2 - x).max(), np.abs(yr2 - y).max()) < 1e-7


def test_energy_drift_along_stored_orbits():
    orbits = stored_orbits()
    worst = {}
    for name, L in orbits.items():
        H = C.energy_logits(L)
        assert np.abs(H[:, 0] - H0).max() < 1e-12, name
        worst[name] = np.abs(H - H[:, :1]).max()
    assert max(worst.values()) < 1e-8, worst
