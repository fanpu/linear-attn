"""Per-band ghost measurement.

For an output f and the over-text B, the residual is r = f - B. For a template page T
(mean removed) and a radial spatial-frequency band k (octaves, cycles per image),

    ghost_k(T) = <P_k r, P_k T> / <P_k T, P_k T>

is the least-squares amplitude of T's band-k content inside the residual: 1 means the band is
all still there, 0 means none of it. With T = A it is the palimpsest; with T = a decoy page that
shares A's hand and ruling but not its words it is the chance level; with T = B it is minus the
fraction of B's band not yet written.
"""
import numpy as np
import torch

# band edges in cycles per image; the last band runs to the corner of the spectrum
def band_edges(n):
    e = []
    c = 1
    while c < n / 2:
        e.append(c)
        c *= 2
    e.append(n / 2 * 1.5)  # last band runs from n/4 to the corner of the spectrum
    return e  # e.g. n=256: 1,2,4,...,64,192 -> bands [1,2),...,[32,64),[64,192]


def band_masks(n, device="cpu"):
    f = torch.fft.fftfreq(n, d=1.0 / n, device=device)  # cycles per image
    R = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    e = band_edges(n)
    masks = [((R >= lo) & (R < hi)).float() for lo, hi in zip(e[:-1], e[1:])]
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(e[:-1], e[1:])]
    return torch.stack(masks), labels  # (K, n, n)


class Ghost:
    """Precomputes template spectra so each snapshot costs one FFT."""

    def __init__(self, templates: dict, n, device="cpu"):
        self.names = list(templates)
        self.M, self.labels = band_masks(n, device)
        T = torch.stack([torch.as_tensor(templates[k], device=device, dtype=torch.float32) for k in self.names])
        T = T - T.mean(dim=(1, 2), keepdim=True)
        self.FT = torch.fft.fft2(T.double())  # (P, n, n)
        # per-band energies of each template (P, K); whole-image energy too
        self.E = torch.einsum("pij,kij->pk", self.FT.abs() ** 2, self.M.double())
        self.Eall = (self.FT.abs() ** 2).sum(dim=(1, 2))

    def __call__(self, r):
        """r: (n, n) residual. Returns (P, K) band coefficients and (P,) whole-image ones."""
        Fr = torch.fft.fft2(torch.as_tensor(r, device=self.FT.device).double())
        cross = (self.FT.conj() * Fr[None]).real  # (P, n, n)
        g = torch.einsum("pij,kij->pk", cross, self.M.double()) / self.E
        gall = cross.sum(dim=(1, 2)) / self.Eall
        return g.float().cpu().numpy(), gall.float().cpu().numpy()

    def band_energy(self, x):
        """Fraction of each band's energy (for describing where a page lives)."""
        F = torch.fft.fft2(torch.as_tensor(x, device=self.FT.device).double() - float(np.mean(x)))
        e = torch.einsum("ij,kij->k", F.abs() ** 2, self.M.double())
        return (e / e.sum()).cpu().numpy()


def bandpass(x, n, lo, hi):
    """Hard radial band-pass of a numpy image, [lo, hi) cycles per image."""
    f = np.fft.fftfreq(n, d=1.0 / n)
    R = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    m = (R >= lo) & (R < hi)
    return np.real(np.fft.ifft2(np.fft.fft2(x) * m))


def psnr(f, t):
    mse = float(np.mean((np.asarray(f, np.float64) - t) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))
