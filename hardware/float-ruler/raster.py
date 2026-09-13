"""Tiny tick rasteriser with physically-motivated ink accumulation.

Each tick deposits `w` pixels of ink width over its height; coverage adds up and is mapped
to opacity by 1-exp(-gain*A).  This keeps dense formats (fp16's 1024 ticks/octave) honest:
they become a tone whose silhouette is the tick-height hierarchy, instead of a black bar.
"""
import numpy as np
import matplotlib.colors as mcolors

SS = 4  # horizontal supersampling


class InkLayer:
    def __init__(self, H, W, rgb=False):
        self.H, self.W = H, W
        self.A = np.zeros((H, W), np.float32)
        self.C = np.zeros((H, W, 3), np.float32) if rgb else None

    def ticks(self, x_px, y_base, h_px, w_px=1.0, colors=None, up=True):
        """x_px: float pixel x; y_base: baseline pixel row; h_px: tick heights (px)."""
        x_px = np.asarray(x_px, np.float64)
        h_px = np.broadcast_to(np.asarray(h_px, np.float64), x_px.shape)
        hmax = int(np.ceil(h_px.max())) + 1 if len(h_px) else 1
        Wss = self.W * SS
        R = np.zeros((hmax + 1, Wss), np.float32)
        k = max(1, int(round(w_px * SS)))
        offs = np.arange(k) - (k - 1) / 2.0
        cols = np.rint(x_px[:, None] * SS + offs[None, :]).astype(np.int64)
        tops = np.clip(hmax - np.rint(h_px).astype(np.int64), 0, hmax)
        tops = np.broadcast_to(tops[:, None], cols.shape)
        ok = (cols >= 0) & (cols < Wss)
        wgt = w_px * SS / k
        np.add.at(R, (tops[ok], cols[ok]), wgt)
        R = np.cumsum(R, axis=0)[:hmax]  # rows 0..hmax-1, bottom row = baseline-1
        R = R.reshape(hmax, self.W, SS).sum(axis=2) / SS
        if up:
            r0 = y_base - hmax
            src = R
        else:
            r0 = y_base
            src = R[::-1]
        a0, a1 = max(r0, 0), min(r0 + hmax, self.H)
        self.A[a0:a1] += src[a0 - r0:a1 - r0]
        if colors is not None and self.C is not None:
            colors = np.asarray(colors)[:, :3]
            for c in range(3):
                Rc = np.zeros((hmax + 1, Wss), np.float32)
                np.add.at(Rc, (tops[ok], cols[ok]), (wgt * np.broadcast_to(colors[:, c:c + 1], cols.shape))[ok])
                Rc = np.cumsum(Rc, axis=0)[:hmax].reshape(hmax, self.W, SS).sum(axis=2) / SS
                if not up:
                    Rc = Rc[::-1]
                self.C[a0:a1, :, c] += Rc[a0 - r0:a1 - r0]

    def hline(self, y, x0, x1, w_px=1.0, strength=1.0):
        y0 = int(round(y - w_px / 2))
        y1 = max(y0 + 1, int(round(y + w_px / 2)))
        self.A[max(y0, 0):y1, int(x0):int(x1)] += strength

    def alpha(self, gain=1.0):
        return 1.0 - np.exp(-gain * self.A)

    def color(self, fallback):
        if self.C is None:
            return None
        rgb = np.array(mcolors.to_rgb(fallback), np.float32)
        out = np.where(self.A[..., None] > 1e-6, self.C / np.maximum(self.A[..., None], 1e-6), rgb)
        return np.clip(out, 0, 1)


def composite(bg, layers):
    """layers: list of (alpha HxW, colour hex or HxWx3). 'over' compositing."""
    H, W = layers[0][0].shape
    out = np.ones((H, W, 3), np.float32) * np.array(mcolors.to_rgb(bg), np.float32)
    for a, col in layers:
        c = np.array(mcolors.to_rgb(col), np.float32) if isinstance(col, str) else col
        out = out * (1 - a[..., None]) + a[..., None] * c
    return np.clip(out, 0, 1)


def multiply(bg, layers):
    """Spot-ink multiply compositing (riso)."""
    H, W = layers[0][0].shape
    out = np.ones((H, W, 3), np.float32) * np.array(mcolors.to_rgb(bg), np.float32)
    for a, col in layers:
        c = np.array(mcolors.to_rgb(col), np.float32)
        out *= 1 - a[..., None] * (1 - c)
    return np.clip(out, 0, 1)
