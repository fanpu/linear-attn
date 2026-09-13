"""Sohl-Dickstein 'Spectral' split-at-zero renders of the SIGNED quantities (declared style):
signed quantization error Q(W)-W (NVFP4 | MXFP4), the NVFP4-vs-MXFP4 block error log-ratio, and the
Hadamard before/after of the (fused) weights and their signed NVFP4 error.
   python render_spectral_style.py
"""
import numpy as np

import styles as S
from render_blockerr import pick_crop

INK = np.array([0.20, 0.18, 0.22])
BG = np.array([0.985, 0.975, 0.955])
MU = np.array([0.42, 0.40, 0.40])


def legend(img, y, x, w, fn, label_l, label_r, items):
    xs = np.linspace(-1, 1, w)
    ref = np.linspace(-1, 1, 20001)
    bar = fn(xs, ref)
    img[y:y + 22, x:x + w] = bar[None]
    items += [(x, y + 30, label_l, 18, "Mono", MU), (x + w, y + 30, label_r, 18, "Mono", MU, "ra"),
              (x + w // 2, y + 30, "0", 18, "Mono", MU, "ma")]


def signed_error(tag, maxrows=1024, zoom=(64, 128, 8), fn=S.spectral_seam, suffix="seam"):
    d = S.load(tag)
    H = min(maxrows, d["W"].shape[0])
    Wd = d["W"].shape[1]
    en, em = d["NVFP4_nea_err"], d["MXFP4_nea_err"]
    zh, zw, k = zoom
    y0, x0 = pick_crop(en, zh, zw)
    side, top, gap = 80, 210, 60
    ZW = zw * k
    total_w = max(2 * Wd + gap, 2 * ZW + gap) + 2 * side
    img = S.canvas(top + H + 140 + zh * k + 260, total_w, BG)
    S.paste(img, fn(en[:H], en), top, side)
    S.paste(img, fn(em[:H], em), top, side + Wd + gap)
    yz = top + H + 140
    S.paste(img, S.up(fn(en[y0:y0 + zh, x0:x0 + zw], en), k), yz, side)
    S.paste(img, S.up(fn(em[y0:y0 + zh, x0:x0 + zw], em), k), yz, side + ZW + gap)
    items = [(side, 36, "Signed error, Spectral", 58, "Bold", INK),
             (side, 112, f"{d['name']}  Q(W) - W per weight, rows 0:{H}.  left NVFP4, right MXFP4.  Below: rows {y0}:{y0 + zh}, cols {x0}:{x0 + zw} at x{k}", 24, "Sans", MU),
             (side, top - 40, "NVFP4", 28, "Mono", INK), (side + Wd + gap, top - 40, "MXFP4", 28, "Mono", INK),
             (side, yz - 44, "zoom (same colouring)", 24, "Mono", INK)]
    ly = yz + zh * k + 40
    legend(img, ly, side, 700, fn, "most negative", "most positive", items)
    how = ("each sign rank-normalized separately, dark ends meet at 0 (purple = just above, deep red = just below; pale = largest |error|)"
           if suffix == "seam" else "Sohl-Dickstein colab cdf_img (buffer 0.25): dark = largest |error|, pastel seam at 0")
    items += [(side, ly + 80, f"Declared 'Spectral' mapping: {how}. Each panel normalized to its own error distribution.", 22, "Sans", MU),
              (side, ly + 120, S.STACK, 20, "Sans", MU)]
    img = S.text(img, items)
    return S.save(img, f"spectral_signed_error_{suffix}_{tag}.png")


def nvmx_spectral(tag, maxrows=1024):
    d = S.load(tag)
    H = min(maxrows, d["W"].shape[0])
    Wd = d["W"].shape[1] // 32 * 32
    en = d["NVFP4_nea_err"][:H, :Wd].astype(np.float64).reshape(H, -1, 32)
    em = d["MXFP4_nea_err"][:H, :Wd].astype(np.float64).reshape(H, -1, 32)
    r = 0.5 * np.log2((em ** 2).mean(-1) / (en ** 2).mean(-1))
    side, top = 80, 210
    img = S.canvas(top + H + 230, 2 * side + Wd, BG)
    S.paste(img, np.repeat(S.spectral_seam(r), 32, axis=1), top, side)
    items = [(side, 36, "NVFP4 vs MXFP4, Spectral", 58, "Bold", INK),
             (side, 112, f"{d['name']}  rows 0:{H}; one value per 32-weight block: log2(RMSE MXFP4 / RMSE NVFP4)", 24, "Sans", MU),
             (side, 150, f"blue-green side: MXFP4 worse ({100 * (r > 0).mean():.0f}% of blocks); red-orange side: NVFP4 worse", 24, "Sans", MU)]
    legend(img, top + H + 30, side, 700, S.spectral_seam, "NVFP4 worse", "MXFP4 worse", items)
    items += [(side, top + H + 110, "Declared 'Spectral' split at 0, each side rank-normalized; block = 32 px wide x 1 px tall.", 22, "Sans", MU),
              (side, top + H + 150, S.STACK, 20, "Sans", MU)]
    img = S.text(img, items)
    return S.save(img, f"spectral_nvmx_{tag}.png")


def hadamard_spectral(tag, H=1024):
    d = np.load(f"{S.CACHE}/hadamard_{tag}.npz")
    base = "fused" if "fused_W" in d else "raw"
    Wd = min(1024, d["raw_W"].shape[1])
    side, top, gap = 80, 230, 60
    img = S.canvas(top + 2 * H + 90 + 260, 2 * side + 2 * Wd + gap, BG)
    panels = [(f"{base}_W", "weights W" + (" diag(g)" if base == "fused" else ""), 0, 0),
              ("rotated_W", "rotated weights", 0, 1),
              (f"{base}_NVFP4_err", "NVFP4 error, before", 1, 0),
              ("rotated_NVFP4_err", "NVFP4 error, after rotation", 1, 1)]
    items = [(side, 36, "Rotation, Spectral", 58, "Bold", INK),
             (side, 112, f"{d['name']}  rows {d['rows'][0]}:{d['rows'][0] + H}; top: signed weights, bottom: signed Q(W)-W (NVFP4); left before, right after Hadamard", 24, "Sans", MU)]
    for key, lab, r, c in panels:
        y = top + r * (H + 90)
        x = side + c * (Wd + gap)
        A = d[key][:H, :Wd]
        S.paste(img, S.spectral_seam(A, d[key]), y, x)
        items.append((x, y - 40, lab, 28, "Mono", INK))
    items += [(side, top + 2 * H + 110, "Declared 'Spectral' split at 0, each sign rank-normalized per panel (dark = near zero).  Column sign bias (vertical bands) is spread by the rotation; row structure survives.", 22, "Sans", MU),
              (side, top + 2 * H + 150, S.STACK, 20, "Sans", MU)]
    img = S.text(img, items)
    return S.save(img, f"spectral_hadamard_{tag}.png")


if __name__ == "__main__":
    for t in ["L26_q_proj", "embed_rare", "L16_k_proj"]:
        signed_error(t)
    signed_error("L26_q_proj", fn=S.spectral_colab, suffix="colab")
    for t in ["L26_q_proj", "L27_gate_proj"]:
        nvmx_spectral(t)
    for t in ["L27_gate_proj", "embed_rare", "L16_k_proj", "L26_q_proj"]:
        hadamard_spectral(t)
