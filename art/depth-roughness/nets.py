"""Finite-width random Heaviside networks on S^2 (paper's calibration, Gamma_b = 0), evaluated exactly
at arbitrary directions, cheaply at deep zoom.

T(x) = sqrt(2/n) v . H(h_L(x)),  h_1 = W0 x,  h_{l+1} = sqrt(2/n) W_l H(h_l),  all entries N(0,1).

Tricks (exact, not approximations):
  * layer-1 codes change only across the n great circles {W0_j . x = 0}; inside a window only the
    columns J whose circle crosses the window vary, so h_2 = W1 base + W1[:, J] (a_J - base_J);
  * W1 columns are generated on the fly from counter-based seeds, so n = 65536 fits in memory;
  * layers >= 3 use full matmuls (use small n there).
"""
import numpy as np
import torch

BLOCK = 1024


class HeavisideNet:
    def __init__(self, n, L, seed, device="cuda"):
        self.n, self.L, self.seed = n, L, seed
        self.dev = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W0 = torch.randn(n, 3, generator=g, dtype=torch.float64).to(self.dev)
        self.v = torch.randn(n, generator=g, dtype=torch.float64).to(self.dev)
        self.scale = np.sqrt(2.0 / n)
        self._deep = {}

    # W_l column blocks, float32, deterministic per (seed, layer, block)
    def _block(self, layer, b):
        g = torch.Generator(device=self.dev).manual_seed(self.seed * 7919 + layer * 104729 + b)
        m = min(BLOCK, self.n - b * BLOCK)
        return torch.randn(self.n, m, generator=g, device=self.dev, dtype=torch.float32)

    def _cols(self, layer, idx):
        """W_layer[:, idx] as float32 [n, len(idx)]."""
        idx = torch.as_tensor(idx, device=self.dev)
        out = torch.empty(self.n, len(idx), device=self.dev, dtype=torch.float32)
        blocks = torch.unique(idx // BLOCK)
        for b in blocks.tolist():
            sel = (idx // BLOCK) == b
            out[:, sel] = self._block(layer, b)[:, idx[sel] - b * BLOCK]
        return out

    def _matvec_all(self, layer, a):
        """W_layer @ a for a single float vector a [n]."""
        out = torch.zeros(self.n, device=self.dev, dtype=torch.float64)
        for b in range((self.n + BLOCK - 1) // BLOCK):
            blk = self._block(layer, b)
            out += (blk.double() @ a[b * BLOCK:b * BLOCK + blk.shape[1]].double())
        return out

    def _full(self, layer):
        if layer not in self._deep:
            self._deep[layer] = torch.cat([self._block(layer, b) for b in range((self.n + BLOCK - 1) // BLOCK)], 1)
        return self._deep[layer]

    @torch.no_grad()
    def __call__(self, V, chunk_elems=int(1.5e8)):
        """V: numpy [..., 3] float64 unit vectors. Returns numpy float64 [...]."""
        shp = V.shape[:-1]
        X = torch.as_tensor(V.reshape(-1, 3), dtype=torch.float64, device=self.dev)
        M, n = X.shape[0], self.n
        cs = max(1, chunk_elems // n)
        T = torch.empty(M, dtype=torch.float64, device=self.dev)
        if self.L == 1:
            for i in range(0, M, cs):
                T[i:i + cs] = ((X[i:i + cs] @ self.W0.T) >= 0).double() @ self.v
            return (T * self.scale).cpu().numpy().reshape(shp)
        base = (X[0] @ self.W0.T) >= 0
        vary = torch.zeros(n, dtype=torch.bool, device=self.dev)
        for i in range(0, M, cs):
            vary |= (((X[i:i + cs] @ self.W0.T) >= 0) != base).any(0)
        J = torch.nonzero(vary).flatten()
        h2base = (self._matvec_all(1, base.float()) * self.scale).float()
        WJ = self._cols(1, J) if len(J) else None
        W0J = self.W0[J]
        cs2 = max(1, chunk_elems // (n + max(len(J), 1)))
        deep = [self._full(l) for l in range(2, self.L)]
        for i in range(0, M, cs2):
            xb = X[i:i + cs2]
            if WJ is not None:
                d = ((xb @ W0J.T) >= 0).float() - base[J].float()
                h = h2base[None] + self.scale * (d @ WJ.T)
            else:
                h = h2base[None].expand(xb.shape[0], n)
            for W in deep:
                h = self.scale * (((h >= 0).float()) @ W.T)
            T[i:i + cs2] = (h >= 0).double() @ self.v
        return (T * self.scale).cpu().numpy().reshape(shp)
