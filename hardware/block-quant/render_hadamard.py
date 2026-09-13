"""Hadamard before/after plates from cache/hadamard_<tag>.npz.
   python render_hadamard.py [tags]
Row of panels: as shipped | RMSNorm gain fused in (what QuaRot rotates) | after the Hadamard rotation.
Per panel: NVFP4 per-block scale mosaic (true geometry: one block = 16 px wide x 1 px tall), the exp-b3
bitplane of the bf16 code of that matrix, and the per-column RMS profile.
"""
import sys

import numpy as np

import styles as S
from compute_bitplanes import plane_stats

OCT = (-2.5, 3.5)   # declared: scale maps shown from median-2.5 to median+3.5 octaves, per panel


def col_profile(W, h, style, fg, acc, ref=None):
    """1 px per input dim; bar height = log10(col RMS / median col RMS), shared axis -0.5..+1.5 decades."""
    rms = np.sqrt((W.astype(np.float64) ** 2).mean(0))
    v = np.log10(rms / np.median(rms))
    img = np.zeros((h, len(v), 3)) + (S.NIGHT if style == "night" else S.PAPER)
    zero = int(h * 0.75)
    for j, x in enumerate(v):
        t = int(np.clip(x / 1.5, -1 / 3, 1) * zero)
        if t >= 0:
            img[zero - t:zero, j] = acc
        else:
            img[zero:zero - t, j] = fg
    img[zero, :] = fg
    return img


def plate(tag, style="night", nrows=1024):
    d = np.load(f"{S.CACHE}/hadamard_{tag}.npz")
    variants = [v for v in ("raw", "fused", "rotated") if f"{v}_W" in d]
    titles = {"raw": "as shipped", "fused": "RMSNorm gain fused in", "rotated": "after Hadamard rotation"}
    W0 = d["raw_W"]
    H = min(nrows, W0.shape[0])
    Wd = min(1024, W0.shape[1])
    side, top, gap = 90, 250, 70
    prof_h = 260
    panel_h = H + 60 + H + 60 + prof_h + 230
    img_w = side * 2 + len(variants) * Wd + (len(variants) - 1) * gap
    if style == "night":
        bg, fg, mu, acc = S.NIGHT, np.array([0.93, 0.91, 0.86]), np.array([0.55, 0.55, 0.6]), np.array([1.0, 0.55, 0.25])
        cm = S.cmap("magma")
        bit = lambda b: np.where(b[..., None] == 1, np.array([0.96, 0.93, 0.84]), S.NIGHT)
    else:
        bg, fg, mu, acc = S.PAPER, S.RISO_BLACK, np.array([0.4, 0.38, 0.35]), S.RISO_PINK
        cm = None
        bit = lambda b: S.ink(b.astype(float), S.RISO_BLUE)
    img = S.canvas(top + panel_h + 120, img_w, bg)
    name = str(d["name"])
    rows = d["rows"]
    items = [(side, 50, "Rotation" if style == "night" else "ROTATION", 72, "Bold", fg if style == "night" else S.RISO_PINK),
             (side, 150, f"{name}  rows {rows[0]}:{rows[0] + H}   Q = H_1024 diag(random signs)/32;  "
                         f"{'W -> W diag(g) Q' if d['side'] == 'in' else 'W -> Q^T W'}", 28, "Sans", mu)]
    for j, v in enumerate(variants):
        x = side + j * (Wd + gap)
        y = top
        items.append((x, y - 50, titles[v], 36, "Bold", fg))
        ls = np.log2(d[f"{v}_NVFP4_scale"][:H, : Wd // 16])
        med = np.median(ls)
        n = np.clip((np.repeat(ls, 16, axis=1) - med - OCT[0]) / (OCT[1] - OCT[0]), 0, 1)
        S.paste(img, S.to_rgb(n, cm) if cm else S.ink(n ** 1.7, S.RISO_PINK), y, x)
        items.append((x, y + H + 12, "NVFP4 block scale, log2 (1 block = 16x1 px)", 22, "Mono", mu))
        y += H + 60
        # declared choice: show the exponent plane (b4..b0, bits 11..7) with the strongest stripe structure,
        # because fusing the gain shifts every exponent and so moves the informative plane
        best = None
        for eb in range(4, -1, -1):
            p = ((d[f"{v}_raw"][:H, :Wd] >> (7 + eb)) & 1).astype(np.uint8)
            stp = plane_stats(p)
            if stp["H"] > 0.05 and (best is None or stp["od_row"] + stp["od_col"] > best[2]["od_row"] + best[2]["od_col"]):
                best = (eb, p, stp)
        eb, b3, st = best
        S.paste(img, bit(b3), y, x)
        items.append((x, y + H + 12, f"bf16 exp b{eb} plane (most striped)  row x{st['od_row']:.0f}  col x{st['od_col']:.0f}", 22, "Mono", mu))
        y += H + 60
        S.paste(img, col_profile(d[f"{v}_W"][:, :Wd], prof_h, style, fg, acc), y, x)
        items.append((x, y + prof_h + 10, "column RMS / median (log, -0.5..+1.5 decades), 1 px per input dim", 22, "Mono", mu))
        items += [
            (x, y + prof_h + 60, f"kurtosis {d[f'{v}_kurtosis']:.1f}   col max/median {d[f'{v}_colmax_med']:.1f}   row max/median {d[f'{v}_rowmax_med']:.1f}", 26, "Mono", fg),
            (x, y + prof_h + 100, f"rel.MSE  NVFP4 {d[f'{v}_NVFP4_relmse']:.4f}   MXFP4 {d[f'{v}_MXFP4_relmse']:.4f}", 26, "Mono", fg),
            (x, y + prof_h + 140, f"col-RMS spread (CV) {d[f'{v}_colrms_cv']:.3f}   row-RMS CV {d[f'{v}_rowrms_cv']:.3f}", 26, "Mono", fg),
        ]
    foot = top + panel_h + 20
    items += [(side, foot, "A rotation on the input side mixes columns, so column (input-dim) outliers are spread out; row norms are untouched, so row stripes survive. "
                           "Scale maps: each panel centred on its own median (declared).", 24, "Sans", mu),
              (side, foot + 40, "Simplification: only the residual-stream rotation is applied (no online Hadamard inside attention/MLP, no activation quantization).  " + S.STACK, 22, "Sans", mu)]
    img = S.text(img, items)
    return S.save(img, f"hadamard_{style}_{tag}.png", palette=None if style == "night" else None)


if __name__ == "__main__":
    tags = sys.argv[1:] or ["L27_gate_proj", "L16_k_proj", "L26_q_proj", "L06_down_proj", "embed_rare"]
    for t in tags:
        for s in ("night", "riso"):
            plate(t, s)
