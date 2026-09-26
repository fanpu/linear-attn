"""Final gallery plates. Shared frame from plates.frame(): vellum ground, one margin, one title and
caption block, one typeface. Every panel is a network's measured output on its own sample grid,
enlarged nearest-neighbour (one sample = k x k print pixels, no invented detail).

    python gallery_plates.py
"""
import json
import os

import numpy as np

import look
from look import normal_bl, two_ink, ghost_ink, up, to_img
from metrics import bandpass, band_edges
from pages import load
from plates import frame, run, pick

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "cache", "runs")
G = os.path.join(HERE, "gallery")
os.makedirs(G, exist_ok=True)


def R_(name):
    return run(os.path.join(R, name + ".npz"))


def typology(name, targets, title, caption, out, k=2, cols=None, crop=None):
    d = R_(name)
    st = d["snap_steps"]
    idx = pick(st, targets)
    pan = []
    for i in idx:
        f = d["snap_f"][i].astype(np.float32)
        if crop:
            y0, x0, h, w = crop
            f = f[y0:y0 + h, x0:x0 + w]
        pan.append(up(normal_bl(f), k))
    img = frame(pan, cols or len(pan), title, caption, labels=[f"step {int(st[i]):,}" for i in idx])
    img.save(out)
    print(out, img.size)
    return [int(st[i]) for i in idx]


def moment(name, step, title, caption, out, k=2, crop=None):
    d = R_(name)
    n = d["args"]["res"]
    B = load(n)["B"]
    i = pick(d["snap_steps"], [step])[0]
    f = d["snap_f"][i].astype(np.float32)
    if crop:
        y0, x0, h, w = crop
        f, B = f[y0:y0 + h, x0:x0 + w], B[y0:y0 + h, x0:x0 + w]
    img = frame([up(normal_bl(f), k), up(two_ink(f, B), k)], 2, title, caption.format(step=int(d["snap_steps"][i])))
    img.save(out)
    print(out, img.size)


def relearn_plate(group, suffix, i_step, title, caption, out, k=2):
    pan, lab = [], []
    for c, nm in (("AB", "held A, then B"), ("CB", "held C, then B"), ("noneB", "only ever B")):
        d = R_(f"{group}_{c}{suffix}")
        f = d["relearn_f"][i_step].astype(np.float32)
        l3 = d["loss3"]
        s = int(d["relearn_steps"][i_step])
        pan.append(up(normal_bl(f), k))
        lab.append(f"{nm}: {-10 * np.log10(l3[s - 1]):.1f} dB on A")
    img = frame(pan, 3, title, caption.format(step=s), labels=lab)
    img.save(out)
    print(out, img.size)


def kick_plate(name, targets, title, caption, out, k=2, crop=(0, 0, 128, 128)):
    d = R_(name)
    kk = np.load(os.path.join(R, name + "_kick.npz"))
    st = d["snap_steps"]
    idx = pick(st, targets)
    y0, x0, h, w = crop
    top = [up(normal_bl(d["snap_f"][i].astype(np.float32)[y0:y0 + h, x0:x0 + w]), k) for i in idx]
    bot = [up(normal_bl(kk["kick_f"][i].astype(np.float32)[y0:y0 + h, x0:x0 + w]), k) for i in idx]
    dw = d["snap_dw"].sum(1)
    lab = [f"step {int(st[i]):,}" for i in idx] + [f"Σ‖ΔW‖ = {dw[i]:.2f}" for i in idx]
    img = frame(top + bot, len(idx), title, caption, labels=lab, label_size=15)
    img.save(out)
    print(out, img.size)


def exposure_row(names, targets, lo, hi, title, caption, out, crop, k=3):
    """Declared exposure: ink density linearly mapped from [lo, hi] to [0, 1], the same for every
    panel (like raking light on a real palimpsest). Nothing else changes. One row per run."""
    pan, lab = [], []
    for name, tag in names:
        d = R_(name)
        st = d["snap_steps"]
        idx = pick(st, targets)
        y0, x0, h, w = crop
        pan += [up(normal_bl(np.clip((d["snap_f"][i].astype(np.float32)[y0:y0 + h, x0:x0 + w] - lo) / (hi - lo), 0, 1)), k) for i in idx]
        lab += [f"{tag} · step {int(st[i]):,}" for i in idx]
    img = frame(pan, len(targets), title, caption, labels=lab, label_size=15)
    img.save(out)
    print(out, img.size)


def slit_two_ink(name, t_max, title, caption, out, k=2, nslices=None):
    """Time runs left to right across one page: column block j is the page at the j-th snapshot
    (log-spaced steps 0..t_max). B's lines are vertical, so every block holds whole columns of B,
    and A's horizontal lines run across the blocks and can be read as they fade."""
    d = R_(name)
    n = d["args"]["res"]
    B = load(n)["B"]
    st = d["snap_steps"]
    sel = [i for i, s in enumerate(st) if s <= t_max]
    nslices = nslices or len(sel)
    edges = np.linspace(0, n, nslices + 1).astype(int)
    page = np.zeros((n, n), np.float32)
    used = []
    for j in range(nslices):
        i = sel[int(round(j * (len(sel) - 1) / max(nslices - 1, 1)))]
        used.append(int(st[i]))
        page[:, edges[j]:edges[j + 1]] = d["snap_f"][i].astype(np.float32)[:, edges[j]:edges[j + 1]]
    img = frame([up(two_ink(page, B), k)], 1, title, caption.format(first=used[0], last=used[-1], n=nslices))
    img.save(out)
    print(out, img.size, used)


def band_specimen(name, targets, title, caption, out, crop, k=2, lo_min=8):
    d = R_(name)
    n = d["args"]["res"]
    e = band_edges(n)
    st = d["snap_steps"]
    idx = pick(st, targets)
    y0, x0, h, w = crop
    rows = [(lo, hi) for lo, hi in zip(e[:-1], e[1:]) if lo >= lo_min]
    pan, lab = [], []
    for lo, hi in rows:
        bps = [bandpass(d["snap_f"][i].astype(np.float32), n, lo, hi)[y0:y0 + h, x0:x0 + w] for i in idx]
        s = np.percentile(np.abs(np.stack(bps)), 99.5)
        for j, bp in enumerate(bps):
            pan.append(up(normal_bl(np.clip(bp / s, 0, 1)), k))
            lab.append(f"{lo:g}–{hi:g} c/img · step {int(st[idx[j]]):,}")
    img = frame(pan, len(idx), title, caption, labels=lab, label_size=13)
    img.save(out)
    print(out, img.size)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["all"]
    if "all" in which or "ff" in which:
        s = typology("ff32_w256_adam_AB_s0_n256", [0, 2, 10, 40, 63, 84, 101, 135, 2000],
                     "Scraped, and the under-text returns",
                     "Fourier-feature network (σ 32, width 256), Adam. It held page A, then trained on page B. Its own output, "
                     "at nine steps of the second training.\nThe page goes blank within three steps and stays blank; near step 60 "
                     "the Lucretius comes back, under the Genesis, before it goes for good.",
                     os.path.join(G, "scraped_ff32_w256.png"), k=2, cols=3)
        relearn_plate("ff32_w256_adam", "_s0_n256", 4, "Under ultraviolet: relearning",
                      "Three copies of the same network, each at the end of training on page B, retrained on page A for {step} steps. "
                      "Only the one that once held A\nhas it back; the other two are still blank vellum. (Fourier features σ 32, width 256, Adam.)",
                      os.path.join(G, "uv_relearn_ff32_w256.png"), k=2)
        kick_plate("ff32_w256_adam_AB_s0_n256", [0, 2, 5, 10, 40],
                   "Scraped, not shaken",
                   "Top: training on page B. Bottom: the phase-1 network moved the same distance per weight tensor in a random direction. "
                   "A random push of the same size\nleaves A untouched; only B's gradient scrapes it off. (Fourier features σ 32, width 256, top-left quarter.)",
                   os.path.join(G, "kick_ff32_w256.png"), k=2)
    if "all" in which or "ff" in which or "resurface" in which:
        exposure_row([("ff32_w256_adam_AB_s0_n256", "held A"), ("ff32_w256_adam_CB_s0_n256", "held C")], [20, 39, 57, 69, 84], 0.08, 0.45,
                     "The under-text comes back",
                     "Fourier features (σ 32, width 256), Adam, top-left quarter, steps 20–84 of training on page B. Top: the network "
                     "that held page A (Lucretius). Bottom, the control: the same network that held page C (Iliad, lines at −33°).\n"
                     "Declared exposure: ink density 0.08–0.45 stretched to the full range, identically for all ten panels. "
                     "Each network's own first page returns.\nMeasured amplitude of A in the top run, finest band: 0.00 at step 20, "
                     "0.04 at 57, 0.13 at 76, 0.19 at 101, 0.05 at 149.  Amplitude of C in the bottom run, 32–64 c/img: 0.00 at step 20, "
                     "0.17 at 69, 0.01 at 149.  Neither run shows the other's page (|amplitude| < 0.06 throughout).",
                     os.path.join(G, "resurface_ff32_w256.png"), crop=(0, 0, 128, 128), k=3)
    if "all" in which or "siren" in which:
        moment("siren30_w256_adam_AB_s0_n256", 2, "Rubric",
               "SIREN (width 256), Adam, step {step} of training on page B. Left: the output as ink. Right: the same output split exactly "
               "into two inks,\nmin(output, B) in sepia and max(output − B, 0) in red: red is ink the network holds that page B does not have.",
               os.path.join(G, "rubric_siren_w256.png"), k=3, crop=(0, 0, 128, 128))
        typology("siren30_w256_adam_AB_s0_n256", [0, 1, 2, 3, 5, 8, 13, 30, 2000],
                 "Broad strokes first",
                 "SIREN (width 256), Adam. It held page A, then trained on page B. Its own output at nine steps of the second training.\n"
                 "B's columns arrive in two steps while A's letters still show through them; by step 13 A is gone at every scale.",
                 os.path.join(G, "broad_strokes_siren_w256.png"), k=2, cols=3)

    if "all" in which or "hero" in which:
        H = "siren30_w256_adam_AB_s0_n512"
        if os.path.exists(os.path.join(R, H + ".npz")):
            moment(H, 2, "Rubric",
                   "SIREN (width 256), Adam, 512 × 512 samples, step {step} of training on page B. Left: the output as ink. Right: the same "
                   "output split exactly into two inks,\nmin(output, B) in sepia and max(output − B, 0) in red: red is ink the network "
                   "holds that page B does not have.",
                   os.path.join(G, "rubric_siren_n512.png"), k=2)
            slit_two_ink(H, 16, "Palimpsest",
                         "One page, {n} vertical strips; strip j is the network's output at the j-th saved step, from step {first} at the "
                         "left edge to step {last} at the right.\nSepia: ink shared with page B. Red: ink page B does not have. "
                         "SIREN (width 256), Adam, 512 × 512 samples.",
                         os.path.join(G, "palimpsest_slit_siren_n512.png"), k=2)
            typology(H, [0, 1, 2, 3, 5, 8, 13, 30, 2000], "Broad strokes first",
                     "SIREN (width 256), Adam, 512 × 512 samples. It held page A, then trained on page B. Its own output at nine steps "
                     "of the second training.",
                     os.path.join(G, "broad_strokes_siren_n512.png"), k=1, cols=3)
            band_specimen(H, [0, 2, 5, 13, 2000], "Specimen: which scales of which page, when",
                          "Rows: octave bands of the output (cycles per image); columns: steps of training on page B. Top-left 128 × 128 "
                          "samples, each row on one scale.\nHorizontal texture is page A (Lucretius), vertical is page B (Genesis). "
                          "Positive part printed as ink.",
                          os.path.join(G, "specimen_bands_siren_n512.png"), crop=(0, 0, 128, 128), k=2, lo_min=16)
