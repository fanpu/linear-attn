"""Kac-Rice check (paper Thm 3.8): E[length of T_L^{-1}(0)] = 2 pi kappa'(1)^{L/2} for regular
activations. Monte Carlo over independent GP draws (lmax 128, 400 draws, independent seeds per row), nodal length by marching squares on a
Gauss-Legendre grid, summed as great-circle segment lengths. Output cache/kacrice.json"""
import json, numpy as np, ducc0, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

LMAX, NT, NP, NDRAW = 128, 384, 768, 400
sp = np.load("cache/spectra.npz"); names = list(sp["names"])
th = ducc0.misc.GL_thetas(NT)
ph = np.arange(NP + 1) / NP * 2 * np.pi
out = {}
def nodal_length(m):
    mm = np.concatenate([m, m[:, :1]], 1)
    cs = plt.contour(ph, th, mm, levels=[0.0])
    tot = 0.0
    for seg in cs.allsegs[0]:
        p, t = seg[:, 0], seg[:, 1]
        v = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], 1)
        dots = np.clip((v[1:] * v[:-1]).sum(1), -1, 1)
        tot += np.arccos(dots).sum()
    plt.close("all")
    return tot
for a in ["relu", "gelu", "tanh", "sin"]:
    for L in [1, 4, 8]:
        C = sp["C"][names.index(f"{a}_L{L}")][:LMAX + 1]
        lens = []
        for s in range(NDRAW):
            z = white_alm(LMAX, 100000 * (1 + ["relu", "gelu", "tanh", "sin"].index(a)) + 1000 * L + s)
            alm = alm_from_white(z, C, LMAX)
            m = ducc0.sht.synthesis_2d(alm=alm[None], spin=0, lmax=LMAX, geometry="GL", ntheta=NT, nphi=NP, nthreads=4)[0]
            lens.append(nodal_length(m))
        lens = np.array(lens)
        pred = 2 * np.pi * kappa_prime_1(a) ** (L / 2)
        out[f"{a}_L{L}"] = dict(mean=float(lens.mean()), sem=float(lens.std() / np.sqrt(NDRAW)), predicted=pred)
        print(a, L, f"{lens.mean():.3f} +- {lens.std()/np.sqrt(NDRAW):.3f}  predicted {pred:.3f}", flush=True)
json.dump(out, open("cache/kacrice.json", "w"), indent=1)
