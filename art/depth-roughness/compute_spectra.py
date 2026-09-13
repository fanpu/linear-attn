"""Angular power spectra C_l^{(L)} of the infinite-width kernels kappa_L on S^2 (lmax 8192),
plus flat-sky spectral densities S(k) for the deep-zoom bands. Output: cache/spectra.npz"""
import numpy as np, time, json
from common import *

LMAX = 8192
LS = list(range(0, 13))
names, funcs = [], []
for a in ACTS:
    for L in LS:
        names.append(f"{a}_L{L}"); funcs.append(kernel_d(a, L))
names.append("rbf"); funcs.append(kernel_d("rbf", 0))
t0 = time.time()
C = spectra(funcs, LMAX, device="cuda")
C = np.clip(C, 0, None)
print("spectra", C.shape, time.time() - t0)
l = np.arange(LMAX + 1)
var = ((2 * l + 1) * C).sum(1) / (4 * np.pi)
# flat-sky S(k) on a log grid for the zoom bands (Heaviside / ReLU / rbf)
ks = np.logspace(np.log10(4096), 12, 400)
flat = {}
for a in ["heaviside", "relu"]:
    for L in range(1, 7):
        flat[f"{a}_L{L}"] = flat_spectrum(kernel_d(a, L), ks)
flat["rbf"] = np.zeros_like(ks)
np.savez_compressed("cache/spectra.npz", names=np.array(names), C=C, var_captured=var, ks=ks,
                    flat_names=np.array(list(flat)), flat=np.stack(list(flat.values())))
info = {n: dict(var_captured_lmax8192=float(v), kappa_prime_1=kappa_prime_1(n.split("_L")[0]) if n != "rbf" else None)
        for n, v in zip(names, var)}
json.dump(info, open("cache/spectra_info.json", "w"), indent=1)
print({n: round(float(v), 4) for n, v in zip(names, var) if n.startswith("heaviside")})
