"""Bifurcation diagram of the symmetric-start congestion MWU map along lines of constant y*,
plus the Lyapunov exponent along the same line. python compute_bifurcation.py"""
import numpy as np
import torch
from gamelib import DT, gpu_setup, orbit_cong1d, lyap_cong1d

if __name__ == '__main__':
    gpu_setup()
    out = {}
    for ys, s0, s1 in ((0.4168, 5.0, 40.0), (0.7, 5.0, 80.0)):
        s = torch.linspace(s0, s1, 3000, dtype=DT)
        y = torch.full_like(s, ys)
        X = orbit_cong1d(s, y, T0=3000, T1=600)
        L = lyap_cong1d(s[None], y[None], T0=3000, T1=5000)[0]
        out[f'x_{ys}'] = X.astype(np.float32); out[f's_{ys}'] = s.numpy(); out[f'L_{ys}'] = L
        print(ys, 'frac chaotic', np.mean(L > 0))
    np.savez_compressed('cache/bifurcation.npz', **out)
