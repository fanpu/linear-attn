"""Hero plates from the 256² maps (CPU only, reads cache/). Usage:
  PYTHONPATH=. python heroes.py [maps|metrics|diptych|table] ...
Outputs in gallery/hero, gallery/metrics, gallery/diptych."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as A
import render as R

C = "cache/"
LAB = dict(tp=("temperature T", "top-p"), tr=("temperature T", "repetition penalty"))
HORIZON = 16      # tokens: beyond this, text identity is bf16-kernel dependent (see README, verification)


def axes(d):
    dx = d["xs"][1] - d["xs"][0]
    dy = d["ys"][1] - d["ys"][0]
    xl, yl = LAB[d["grid"]]
    return ((d["xs"][0] - dx / 2, d["xs"][-1] + dx / 2), (d["ys"][0] - dy / 2, d["ys"][-1] + dy / 2), xl, yl)


def put(img, name, sub, d, title, caption, dark=True):
    g, ink = ((0.05, 0.05, 0.06), (0.82, 0.8, 0.75)) if dark else (tuple(R.INK_PAPER), (0.15, 0.15, 0.15))
    R.save(R.frame(img, title, caption, ground=g, ink=ink, axes=axes(d)), name, sub)
    print(name)


def maps(names=("story_tp256", "list_tp256", "fact_tp256"), s=10):
    for n in names:
        d = A.load(C + n + ".npz")
        L = d["tokens"].shape[2]
        q = f"“{d['prompt']}”  Qwen3-0.6B, {d['tokens'].shape[0]}² grid, {L} tokens, shared uniforms"
        put(R.style_mosaic(d, s), f"{n}_mosaic.png", "hero", d, "Text-hash mosaic", q +
            "\ncolour = cell of identical text (hash, proper colouring; declared palette); brightness = token at which the text leaves greedy")
        put(R.style_glass(d, s), f"{n}_glass.png", "hero", d, "Stained glass", q +
            "\nlead = boundary between different texts; glass colour and jitter declared from the text hash")
        put(R.style_ink(d, s, w=2), f"{n}_ink.png", "hero", d, "Boundary lines", q +
            "\none line per tile edge whose two texts differ; darker = the texts part at an earlier token", dark=False)
        put(R.style_age(d, s, w=2), f"{n}_age.png", "hero", d, "Boundary age", q +
            f"\nboundary colour = token index at which the neighbours part (batlow, 0 → {L - 1})")
        if n == "story_tp256":
            fd = A.first_divergence(d["tokens"]).astype(float)
            sig = fd - (HORIZON - 0.5)
            put(R.style_split(sig, s, lead_tokens=d["tokens"][..., :HORIZON]), f"{n}_firstdiv_spectral.png", "hero", d,
                "First divergence from greedy, Spectral split",
                q + f"\nsigned = (token where the text leaves the greedy text) − {HORIZON}: purple side leaves before token {HORIZON}, red side later or never."
                f"\nEach side rank-normalised (declared; sequential quantity, labelled variant). Thin lines: boundaries born before token {HORIZON}")
            sig, thr = R.coherence_split(d)
            put(R.style_split(sig, s), f"{n}_coherence_spectral.png", "hero", d, "Coherent vs degenerate, Spectral split",
                q + f"\nsigned = mean model entropy along the text − {thr:.2f} nats (histogram valley); purple = coherent, red = degenerate")
            put(R.style_split(sig, s, pairing="aurora_ember"), f"{n}_coherence_aurora.png", "hero", d,
                "Coherent vs degenerate, aurora / ember split", q + "\nsame field, palettes.py aurora_ember pairing (declared)")
            Hm = A.mean_entropy(d)
            from scipy.stats import rankdata   # round first: bf16 batch noise (~1e-3 nats) inside one cell must not band
            fill = rankdata(np.round(Hm, 2).ravel()).reshape(Hm.shape) / Hm.size
            put(R.style_riso(d, s, shift=4, fill=fill), f"{n}_riso.png", "hero", d, "Two-ink riso", q +
                "\nblue = boundaries (full ink if born in the first half); pink coverage = rank of mean model entropy; pink misregistered 4 px (declared)", dark=False)


def metrics(n="story_tp256", s=8):
    d = A.load(C + n + ".npz")
    tk = d["tokens"]
    L = tk.shape[2]
    m = A.text_metrics(tk, d["eos_pos"])
    Hm = A.mean_entropy(d)
    Hs = A.mean_entropy(d, "H_samp")
    fields = [("rep", m["rep"], "cmc.lajolla", "repetition: fraction of 3-grams already seen in the text"),
              ("distinct2", m["distinct2"], "cmc.batlow", "distinct-2: unique bigrams / bigrams"),
              ("length", m["length"].astype(float), "cmc.oslo", f"length until EOS (capped at {L})"),
              ("Hmodel", Hm, "magma", "mean model entropy per step (nats)"),
              ("Hsamp", Hs, "magma", "mean entropy of the truncated, tempered sampling distribution (nats)"),
              ("firstdiv", A.first_divergence(tk).astype(float), "cmc.batlow", "token at which the text leaves greedy")]
    ext = [*axes(d)[0], *axes(d)[1]]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.6), facecolor="#101012")
    for ax, (k, f, cm, lab) in zip(axs.ravel(), fields):
        im = ax.imshow(f, origin="lower", extent=ext, aspect="auto", cmap=cm, interpolation="nearest",
                       vmin=np.nanpercentile(f, 0.5), vmax=np.nanpercentile(f, 99.5))
        ax.set_title(lab, color="#ddd", fontsize=9.5)
        ax.tick_params(colors="#bbb", labelsize=8)
        ax.set_xlabel("T", color="#bbb"); ax.set_ylabel("top-p", color="#bbb")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.ax.tick_params(colors="#bbb", labelsize=7)
    fig.suptitle(f"{n}: text metrics over the (T, top-p) plane, “{d['prompt']}”", color="#eee")
    fig.tight_layout()
    os.makedirs("gallery/metrics", exist_ok=True)
    fig.savefig(f"gallery/metrics/{n}_metrics_sheet.png", dpi=150, facecolor=fig.get_facecolor())
    # large single plates: Spectral split of repetition (0 vs >0 is a genuine two-sided split) and length
    rep = m["rep"]
    sig = np.where(rep > 0, rep, -(1.0 - m["distinct2"]) - 1e-6)
    put(R.style_split(sig, s), f"{n}_rep_spectral.png", "metrics", d, "Repetition, Spectral split",
        f"red = the text repeats a 3-gram (rank of repetition rate); purple = no repeated 3-gram (rank of 1 − distinct-2)"
        f"\nseam = onset of verbatim repetition; each side rank-normalised (declared)")
    # length/EOS is not rendered as a plate: story/list never emit EOS inside the horizon, fact ends at
    # token 10 in >95 % of pixels (see README).
    put(R.style_continuous(Hm, s, "magma"), f"{n}_entropy_magma.png", "metrics", d, "Mean model entropy",
        "mean over the text of the untempered next-token entropy (nats), magma, 0.5–99.5 percentile clip")


def diptych(a="story_tp128", b="story_tp128_gumbel", s=12):
    da, db = A.load(C + a + ".npz"), A.load(C + b + ".npz")
    ims = []
    for d, name in [(da, "inverse CDF, shared uniforms u_t"), (db, "Gumbel-max, shared Gumbel noise g_t")]:
        L = d["tokens"].shape[2]
        nc = len(np.unique(A.prefix_hashes(d["tokens"])[-1]))
        plate = R.style_glass(d, s)
        ims.append(R.frame(plate, name, f"{nc} distinct {L}-token texts in {d['tokens'].shape[0]}² samples",
                           axes=axes(d)))
    h = max(i.shape[0] for i in ims)
    ims = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=0.06) for i in ims]
    R.save(np.concatenate(ims, 1), "diptych_icdf_vs_gumbel_glass.png", "diptych")
    ims = [R.frame(R.style_ink(d, s, w=2), name, None, ground=tuple(R.INK_PAPER), ink=(0.15,) * 3, axes=axes(d))
           for d, name in [(da, "inverse CDF"), (db, "Gumbel-max")]]
    R.save(np.concatenate(ims, 1), "diptych_icdf_vs_gumbel_ink.png", "diptych")


def table(n="story_tp256", ps=(0.5, 0.9), T_max=1.5, rows=10):
    import hover
    out = []
    d = A.load(C + n + ".npz")
    for p in ps:
        out.append(f"<p><b>top-p = {d['ys'][np.argmin(np.abs(d['ys'] - p))]:.3f}</b>, walking along T</p>")
        out.append(hover.transect_table(C + n + ".npz", p, rows, T_max))
    open(f"cache/{n}_transects.html", "w").write("\n".join(out))
    print("\n".join(out)[:3000])


def redo():
    """Regenerate only the story plates that changed (first-divergence split, riso)."""
    maps(("story_tp256",))


if __name__ == "__main__":
    for w in sys.argv[1:] or ["maps", "metrics", "diptych", "table"]:
        globals()[w]()

