"""Multi-scale sampler for the depth-L limiting GP around a point of S^2, for deep zooms.

field(x) = T_SHT(x)                      exact spherical-harmonic sample, l <= LS = 8192
         + sum_k band_k(X, Y)            flat-sky Gaussian bands, l in (LS 4^k, LS 4^(k+1)]

All pieces share one seed; the SHT part is the same draw as a full-sphere render with lmax 8192.
Band k lives on a periodic box of side B_k = 128 lambda_k (lambda_k = 2 pi / (LS 4^k) its longest
wavelength), N = 8192 samples (16 samples per shortest wavelength), centred on the zoom centre.
X, Y are gnomonic tangent-plane coordinates (radians at the centre). Band k is only used when the
frame side F <= B_k (no periodic wrap inside a frame). Approximation: flat-sky spectrum S(k) ~ C_l
for l >= 8192 (checked to <1.5% against the Legendre spectrum at l <= 8192), and gnomonic metric
distortion <= 1% inside B_0 = 0.1 rad.
"""
import numpy as np
import torch
import torch.nn.functional as Fnn
from common import *

LS = 8192
NB = 8192


def _interp_loglog(ks, S, k):
    lk = np.log(ks)
    lS = np.log(np.clip(S, 1e-300, None))
    return np.exp(np.interp(np.log(k), lk, lS))


class MultiScaleField:
    def __init__(self, kernel, seed, center, nbands=10, device="cuda", sht_grid_half=0.2,
                 sht_n=8192, verbose=True):
        self.kernel, self.seed = kernel, seed
        self.e1, self.e2, self.c = rot_frame(center)
        self.dev = torch.device(device)
        sp = np.load("cache/spectra.npz")
        names = list(sp["names"])
        self.C = sp["C"][names.index(kernel)][: LS + 1]
        fnames = list(sp["flat_names"])
        self.ks, self.S = sp["ks"], sp["flat"][fnames.index(kernel)]
        z = white_alm(LS, seed)
        self.alm = alm_from_white(z, self.C, LS)
        del z
        self.verbose = verbose
        # local precomputed SHT grids (bicubic-sampled for small frames)
        self.grids = []
        for half, n in [(sht_grid_half, sht_n), (sht_grid_half / 100, 1024)]:
            s = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2) * half
            X, Y = np.meshgrid(s, -s)
            v = self._dirs(X, Y)
            th, ph = vec_to_thetaphi(v)
            g = synth(self.alm, LS, th, ph, nthreads=16)
            self.grids.append((half, torch.tensor(g, dtype=torch.float64)))
            if verbose:
                print(f"  SHT grid half={half:g} n={n} done", flush=True)
        self.bands = []
        for k in range(nbands):
            self.bands.append(self._make_band(k))
            if verbose:
                print(f"  band {k}: B={self.bands[-1][0]:.3e}", flush=True)

    def _dirs(self, X, Y):
        v = self.c + X[..., None] * self.e1 + Y[..., None] * self.e2
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def _make_band(self, k):
        lo, hi = LS * 4.0**k, LS * 4.0 ** (k + 1)
        lam = 2 * np.pi / lo
        h = lam / 64
        B = NB * h
        g = torch.Generator(device=self.dev).manual_seed(int(self.seed) * 1000 + k)
        w = torch.randn(NB, NB, generator=g, device=self.dev, dtype=torch.float64)
        F = torch.fft.fft2(w)
        del w
        f1 = torch.fft.fftfreq(NB, d=h, device=self.dev, dtype=torch.float64) * 2 * np.pi
        kk = torch.sqrt(f1[:, None] ** 2 + f1[None, :] ** 2)
        mask = (kk > lo) & (kk <= hi)
        kv = kk[mask].cpu().numpy()
        amp = torch.zeros_like(kk)
        amp[mask] = torch.tensor(np.sqrt(_interp_loglog(self.ks, self.S, kv)), device=self.dev)
        del kk, mask
        F *= amp
        del amp
        band = (torch.fft.ifft2(F).real * (NB / B)).float().cpu()
        del F
        torch.cuda.empty_cache()
        return (B, h, band)

    # -------------------------------------------------------------------------------------------
    def _sample_grid(self, arr, x0, dx, X, Y, wrap):
        """Bicubic sample a regular grid (origin index 0 at coordinate x0 for columns, rows going
        down from +x0) at plane coords X, Y (torch, on device)."""
        n = arr.shape[0]
        ci = (X - x0) / dx          # column index (float)
        ri = (-Y - x0) / dx         # row index
        cmin, cmax = int(np.floor(ci.min().item())) - 3, int(np.ceil(ci.max().item())) + 4
        rmin, rmax = int(np.floor(ri.min().item())) - 3, int(np.ceil(ri.max().item())) + 4
        if wrap:
            rr = torch.arange(rmin, rmax + 1) % n
            cc = torch.arange(cmin, cmax + 1) % n
        else:
            rr = torch.arange(rmin, rmax + 1).clamp(0, n - 1)
            cc = torch.arange(cmin, cmax + 1).clamp(0, n - 1)
        crop = arr[rr][:, cc].to(self.dev, torch.float64)
        H, W = crop.shape
        gx = (ci - cmin) / (W - 1) * 2 - 1
        gy = (ri - rmin) / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        out = Fnn.grid_sample(crop[None, None], grid, mode="bicubic", align_corners=True)
        return out[0, 0]

    def eval(self, cx, cy, F, n, fade=True, max_band=None):
        """Frame of side F (radians, tangent plane) centred at (cx, cy), n x n pixel centres.
        Returns float64 numpy [n, n]."""
        s = (torch.arange(n, dtype=torch.float64, device=self.dev) + 0.5) / n * F - F / 2
        X = (cx + s)[None, :].expand(n, n)
        Y = (cy - s)[:, None].expand(n, n)
        if F > 2 * self.grids[0][0] * 0.95 or abs(cx) + F / 2 > self.grids[0][0] or abs(cy) + F / 2 > self.grids[0][0]:
            v = self._dirs(X.cpu().numpy(), Y.cpu().numpy())
            th, ph = vec_to_thetaphi(v)
            out = torch.tensor(synth(self.alm, LS, th, ph, nthreads=16), device=self.dev)
        else:
            half, g = self.grids[0]
            if F < 2 * self.grids[1][0] * 0.9 and abs(cx) + F / 2 < self.grids[1][0] and abs(cy) + F / 2 < self.grids[1][0]:
                half, g = self.grids[1]
            nn_ = g.shape[0]
            out = self._sample_grid(g, -half, 2 * half / (nn_ - 1), X, Y, wrap=False)
        for k, (B, h, band) in enumerate(self.bands):
            if max_band is not None and k > max_band:
                break
            if F > B:
                break
            wgt = 1.0
            if F > B / 2:
                if not fade:
                    break
                t = np.log2(F / (B / 2))           # 0..1
                wgt = np.cos(np.pi / 2 * t) ** 2
            out = out + wgt * self._sample_grid(band, -NB / 2 * h, h, X, Y, wrap=True)
        return out.cpu().numpy()
