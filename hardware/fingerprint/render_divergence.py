"""render_divergence.py - Fingerprint piece 2 plates from cache/div_<prompt>.npz + .json (no GPU).

    OMP_NUM_THREADS=2 python render_divergence.py [--prompts feynman story sky] [--only raster drift texts firstdiv triptych film]

Rows = batch size B (one row per measured B), columns = generated token position t. Everything drawn is measured:
which completion each batch size produced (greedy, temperature 0), where it first departs from the B = 1 text,
how far the B = 1 logits drifted before that, and which CUDA kernels ran. Colours are declared choices.
"""
import argparse, json, os, subprocess, sys, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P
from render_invariance import short_kernel, stack_line, cat_colors, INK, PAPER, NIGHT

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6})

NIPPON = ["#cb1b45", "#005caf", "#1b813e", "#ffb11b", "#592c63", "#ca7a2c", "#86a697", "#e16b8c", "#0089a7", "#6a4c9c",
          "#f596aa", "#7b90d2", "#a8d8b9", "#fedfe1", "#b28fce", "#e9a368"]


def load(prompt):
    z = np.load(f"{CACHE}/div_{prompt}.npz")
    d = {k: z[k] for k in z.files}
    d["meta"] = json.load(open(f"{CACHE}/div_{prompt}.json"))
    d["NB"], d["L"] = d["toks"].shape
    return d


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)


def classes(d):
    """Distinct full completions, ids by count (0 = the B = 1 text), and per-row first token that departs from B = 1."""
    keys = [tuple(r) for r in d["toks"]]
    uniq = {}
    for k in keys:
        uniq[k] = uniq.get(k, 0) + 1
    order = sorted(uniq, key=lambda k: (k != keys[0], -uniq[k]))
    cid = {k: i for i, k in enumerate(order)}
    ids = np.array([cid[k] for k in keys])
    first = np.array([int(np.argmax(r != d["toks"][0])) if (r != d["toks"][0]).any() else -1 for r in d["toks"]])
    return ids, first, order, uniq


def prefix_tree_ids(d):
    """(NB, L) branch id at each position: rows sharing the same prefix up to t share an id; a branch keeps its id
    until it splits, when the branch that stays with the smaller-B row keeps it and the others get new ids."""
    NB, L = d["NB"], d["L"]
    ids = np.zeros((NB, L), int)
    cur = np.zeros(NB, int)        # branch id per row at t-1
    nxt = 1
    for t in range(L):
        new = cur.copy()
        for b in np.unique(cur):
            rows = np.nonzero(cur == b)[0]
            toks = d["toks"][rows, t]
            keep = toks[0]        # the smallest-B row of this branch keeps the id
            for v in np.unique(toks):
                if v == keep:
                    continue
                new[rows[toks == v]] = nxt; nxt += 1
        cur = new
        ids[:, t] = cur
    return ids


def kernel_sig_ids(d, which="kern_decode"):
    sigs = [tuple(k) for k in d["meta"][which]]
    vocab = []
    for s in sigs:
        if s not in vocab:
            vocab.append(s)
    labels = [" + ".join(sorted({short_kernel(k) for k in v if "Memset" not in k and "Memcpy" not in k})) for v in vocab]
    return np.array([vocab.index(s) for s in sigs]), labels


def style_colors(style, n):
    if style == "paper":
        base = [to_rgb(c) for c in NIPPON]
    elif style == "riso":
        base = [to_rgb(c) for c in ["#ff48b0", "#0078bf", "#00a95c", "#ffe800", "#e45d50", "#5c55a6", "#f6a56d", "#3d5588"]]
    else:
        base = list(plt.get_cmap("cet_glasbey_light")(np.linspace(0, 1, 24))[:, :3])
    if n <= len(base):
        return np.array(base[:n])
    extra = plt.get_cmap("cet_glasbey")(np.arange(n - len(base)) / max(1, n - len(base) - 1))[:, :3]
    return np.vstack([np.array(base), extra])


def raster_rgb(d, style):
    """Row = B, column = t. Shared prefix with B = 1 → pale ground tone; after the departure → colour of the row's final
    completion class. Rows within the batch that disagree with row 0 are hatched with a darker tick."""
    NB, L = d["NB"], d["L"]
    ids, first, order, uniq = classes(d)
    cols = style_colors(style, len(order))
    ground = to_rgb("#e4dfd3") if style in ("paper", "riso") else to_rgb("#1e1e26")
    img = np.tile(np.array(ground), (NB, L, 1))
    for j in range(NB):
        if first[j] >= 0:
            img[j, first[j]:] = cols[ids[j]]
    return img, ids, first, order, cols


def plate_raster(d, prompt, style="paper"):
    NB, L = d["NB"], d["L"]; Bs = d["Bs"]
    img, ids, first, order, cols = raster_rgb(d, style)
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    ksig, klabels = kernel_sig_ids(d)
    kcols = cat_colors(len(klabels))
    fig = plt.figure(figsize=(18, 10), facecolor=bg)
    axm = fig.add_axes([0.075, 0.12, 0.80, 0.70]); axt = fig.add_axes([0.075, 0.83, 0.80, 0.05], sharex=axm)
    axk = fig.add_axes([0.06, 0.12, 0.012, 0.70]); axr = fig.add_axes([0.88, 0.12, 0.10, 0.70])
    axm.imshow(img, aspect="auto", interpolation="nearest", extent=[-.5, L - .5, NB - .5, -.5])
    # within-batch disagreement (rows of the same batch whose greedy token differs from row 0): dark ticks
    rd = d["rows_disagree"] > 0
    yy, xx = np.nonzero(rd)
    axm.scatter(xx, yy, s=2.5, marker="|", color=fg, alpha=.8, lw=.5)
    for j in range(NB):
        if first[j] >= 0:
            axm.plot([first[j] - .5, first[j] - .5], [j - .5, j + .5], color=fg, lw=1.2)
    axm.set_yticks(range(NB)); axm.set_yticklabels([str(b) if (i % 4 == 0) else "" for i, b in enumerate(Bs)], fontsize=6.5)
    axm.set_ylabel("batch size B (each row: B identical copies of the prompt decoded together; row 0 shown)", color=fg)
    axm.set_xlabel("generated token position t", color=fg); axm.tick_params(colors=fg)
    for sp in axm.spines.values(): sp.set_edgecolor(fg)
    # top strip: B = 1 top-2 logit margin per step (where the greedy choice is fragile)
    m = d["margin"][0]
    axt.imshow(np.log10(np.maximum(m, 1e-3))[None], aspect="auto", cmap="cmc.lajolla_r" if dark else "cmc.lajolla",
               interpolation="nearest", extent=[-.5, L - .5, 0, 1], vmin=-2, vmax=1.5)
    axt.set_yticks([]); axt.tick_params(labelbottom=False)
    axt.set_ylabel("B=1\nmargin", color=fg, fontsize=7, rotation=0, ha="right", va="center")
    for sp in axt.spines.values(): sp.set_edgecolor(fg)
    # left strip: decode-step kernel signature per B
    axk.imshow(kcols[ksig][:, None], aspect="auto", interpolation="nearest", extent=[0, 1, NB - .5, -.5])
    axk.set_xticks([]); axk.set_yticks([]); axk.set_title("kern", fontsize=6, color=fg)
    for sp in axk.spines.values(): sp.set_edgecolor(fg)
    # right: first departure token per B and number of within-batch disagreeing rows
    fd = np.where(first >= 0, first, np.nan)
    axr.plot(fd, np.arange(NB), "o", color=fg, ms=2.5)
    axr.set_ylim(NB - .5, -.5); axr.set_xlim(0, L); axr.set_yticks([]); axr.tick_params(colors=fg, labelsize=7)
    axr.set_xlabel("first token ≠ B=1", color=fg, fontsize=8); axr.set_facecolor(bg)
    for sp in axr.spines.values(): sp.set_edgecolor(fg)
    n_div = int((first >= 0).sum())
    fig.text(0.06, 0.955, f"Fingerprint · greedy decoding of one prompt at {NB} batch sizes · “{d['meta']['prompt']}”", fontsize=15, color=fg, weight="bold")
    fig.text(0.06, 0.925, f"Qwen3-0.6B, temperature 0, {L} tokens. {len(order)} distinct completions across {NB} batch sizes; {n_div} of {NB} depart from the B = 1 text "
             f"(median first departure at token {int(np.nanmedian(fd)) if n_div else '-'}). Run-to-run repeats identical: {d['meta']['run_to_run']}.", fontsize=9, color=fg)
    fig.text(0.06, 0.06, "Ground = tokens identical to the B = 1 completion; colour = which distinct completion the row ends in, from its first differing token (bar). "
             "Ticks = greedy tokens of other rows in the same batch that differ from row 0. Top strip: B = 1 top-2 logit margin (log10, dark = fragile).",
             fontsize=8, color=fg)
    fig.text(0.06, 0.035, stack_line(d["meta"]) + f" | {len(klabels)} decode kernel signatures over B", fontsize=7.5, color=fg, alpha=.8)
    fig.savefig(f"{GAL}/divergence_raster_{prompt}_{style}.png", dpi=150, facecolor=bg); plt.close(fig)
    print("wrote raster", prompt, style)


def plate_drift(d, prompt, style="night"):
    """log10 max |Δ logit| of row 0 vs the B = 1 logits at every step, while the prefixes still agree (grey afterwards)."""
    NB, L = d["NB"], d["L"]; Bs = d["Bs"]
    ids, first, order, uniq = classes(d)
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    dm = np.log10(np.maximum(d["dmax"], 1e-6)).astype(float)
    for j in range(NB):
        if first[j] >= 0:
            dm[j, first[j]:] = np.nan
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 9.5), facecolor=bg, gridspec_kw=dict(height_ratios=[3.2, 1]), sharex=True)
    cm = plt.get_cmap("cet_fire" if dark else "cmc.lajolla").copy(); cm.set_bad("#3a3a44" if dark else "#c9c3b6")
    im = ax.imshow(dm, aspect="auto", cmap=cm, interpolation="nearest", vmin=-3, vmax=1, extent=[-.5, L - .5, NB - .5, -.5])
    ax.set_yticks(range(NB)); ax.set_yticklabels([str(b) if (i % 4 == 0) else "" for i, b in enumerate(Bs)], fontsize=6.5)
    ax.set_ylabel("batch size B", color=fg); ax.tick_params(colors=fg)
    for sp in ax.spines.values(): sp.set_edgecolor(fg)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01); cb.set_label("log10 max |logit(B) − logit(1)|", color=fg); cb.ax.tick_params(colors=fg)
    m = d["margin"][0]
    ax2.plot(np.arange(L), m, color=fg, lw=.9)
    ax2.set_yscale("log"); ax2.set_ylabel("B=1 top-2 margin", color=fg); ax2.set_xlabel("generated token position t", color=fg)
    ax2.tick_params(colors=fg); ax2.set_facecolor(bg)
    for sp in ax2.spines.values(): sp.set_edgecolor(fg)
    for j in range(NB):
        if first[j] >= 0:
            ax2.axvline(first[j], color=fg, lw=.4, alpha=.25)
    ax2.set_xlim(-.5, L - .5)
    fig.suptitle(f"Fingerprint · logit drift before the split · “{d['meta']['prompt']}”: how far row 0's logits are from the B = 1 logits at each step (grey = prefix already differs)",
                 color=fg, fontsize=12, x=0.02, ha="left")
    fig.text(0.01, 0.01, stack_line(d["meta"]), fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(f"{GAL}/divergence_drift_{prompt}_{style}.png", dpi=140, facecolor=bg); plt.close(fig)
    print("wrote drift", prompt, style)


def texts_table(d, T, max_rows=16):
    """Markdown rows: completion class, batch sizes, shared-prefix length, text with the departure in bold."""
    ids, first, order, uniq = classes(d)
    Bs = d["Bs"]; base = d["toks"][0]
    rows = []
    for c, k in enumerate(order[:max_rows]):
        js = [j for j in range(d["NB"]) if ids[j] == c]
        f = int(first[js[0]])
        toks = np.array(k)
        if f < 0:
            txt = T.decode(toks)
            rows.append((c, [int(Bs[j]) for j in js], -1, txt, ""))
        else:
            rows.append((c, [int(Bs[j]) for j in js], f, T.decode(toks[:f]), T.decode(toks[f:])))
    return rows


def plate_texts(d, prompt, style="paper", ntok=120):
    """The distinct completions, shared prefix grey, departure onwards in the class colour."""
    T = tok()
    ids, first, order, uniq = classes(d)
    cols = style_colors(style, len(order))
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    rows = texts_table(d, T, max_rows=14)
    fig = plt.figure(figsize=(16, 1.2 + 1.35 * len(rows)), facecolor=bg)
    y = 0.97
    fig.text(0.03, y, f"Fingerprint · the {len(order)} completions of “{d['meta']['prompt']}” (first {ntok} tokens)", fontsize=14, color=fg, weight="bold", va="top")
    y -= 0.05 * 14 / (1.2 + 1.35 * len(rows))
    dy = 1.0 / (len(rows) + 1.2)
    for c, bs, f, pre, post in rows:
        bl = ", ".join(map(str, bs)) if len(bs) <= 12 else ", ".join(map(str, bs[:10])) + f", … ({len(bs)} batch sizes)"
        head = f"B = {bl}" + (f"   departs at token {f}" if f >= 0 else "   (reference)")
        fig.text(0.03, y, "■", color=cols[c], fontsize=12, va="top")
        fig.text(0.05, y, head, fontsize=9, color=fg, va="top", weight="bold")
        pre_t = T.decode(np.array(order[c])[:min(ntok, f if f >= 0 else ntok)])
        post_t = T.decode(np.array(order[c])[f:ntok]) if 0 <= f < ntok else ""
        grey = "#8a857b" if not dark else "#7a7788"
        body = textwrap.wrap((pre_t + "⟨|⟩" + post_t).replace("\n", " ⏎ "), 150)
        split_at = next((i for i, l in enumerate(body) if "⟨|⟩" in l), len(body))
        yy = y - 0.19 * dy
        for i, line in enumerate(body[:4]):
            if i < split_at or f < 0:
                fig.text(0.05, yy, line.replace("⟨|⟩", ""), fontsize=8.2, color=grey, va="top")
            elif i == split_at:
                a, b = line.split("⟨|⟩")
                t1 = fig.text(0.05, yy, a, fontsize=8.2, color=grey, va="top")
                fig.canvas.draw()
                x2 = t1.get_window_extent().x1 / fig.bbox.width
                fig.text(x2, yy, b, fontsize=8.2, color=cols[c], va="top", weight="bold")
            else:
                fig.text(0.05, yy, line, fontsize=8.2, color=cols[c], va="top", weight="bold")
            yy -= 0.165 * dy
        y -= dy
    fig.text(0.03, 0.005, stack_line(d["meta"]), fontsize=7.5, color=fg, alpha=.8)
    fig.savefig(f"{GAL}/divergence_texts_{prompt}_{style}.png", dpi=150, facecolor=bg); plt.close(fig)
    print("wrote texts", prompt, style)


def plate_firstdiv(ds, style="paper"):
    """first departure token vs B for all prompts, with the decode kernel signature bands. Plotter idiom."""
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig, axs = plt.subplots(len(ds), 1, figsize=(16, 3.2 * len(ds) + .8), facecolor=bg, sharex=True)
    for ax, (prompt, d) in zip(np.atleast_1d(axs), ds.items()):
        ids, first, order, uniq = classes(d)
        Bs = d["Bs"]; ksig, klabels = kernel_sig_ids(d); kc = cat_colors(len(klabels))
        sw = np.r_[0, np.nonzero(np.diff(ksig))[0] + 1, len(Bs)]
        for a, b in zip(sw[:-1], sw[1:]):
            ax.axvspan(Bs[a] - .5, Bs[b - 1] + .5, color=kc[ksig[a]], alpha=.2, lw=0)
        cols = style_colors(style, len(order))
        fd = np.where(first >= 0, first, d["L"] + 8)
        ax.scatter(Bs, fd, c=cols[ids], s=18, edgecolor=fg, lw=.4, zorder=3)
        ax.axhline(d["L"] + 8, color=fg, lw=.4, ls=":"); ax.text(Bs[-1], d["L"] + 10, "identical to B = 1 →", ha="right", fontsize=7, color=fg)
        ax.set_ylabel("first token ≠ B = 1", color=fg); ax.tick_params(colors=fg); ax.set_facecolor(bg)
        ax.set_ylim(-5, d["L"] + 22); ax.set_xscale("log"); ax.set_xlim(.9, Bs[-1] * 1.05)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
        ax.text(0.005, 0.9, f"“{d['meta']['prompt']}”  ·  {len(order)} completions", transform=ax.transAxes, fontsize=10, color=fg, weight="bold")
    np.atleast_1d(axs)[-1].set_xlabel("batch size B (log)", color=fg)
    fig.text(0.01, 0.005, "Bands: decode-step CUDA kernel signature (torch.profiler). Colour: which completion (per prompt). " + stack_line(list(ds.values())[0]["meta"]), fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(f"{GAL}/divergence_firstdiv_{style}.png", dpi=140, facecolor=bg); plt.close(fig); print("wrote firstdiv", style)


def plate_triptych(ds, style="paper"):
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig, axs = plt.subplots(1, len(ds), figsize=(7 * len(ds), 9), facecolor=bg)
    for ax, (prompt, d) in zip(np.atleast_1d(axs), ds.items()):
        img, ids, first, order, cols = raster_rgb(d, style)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        for j in range(d["NB"]):
            if first[j] >= 0:
                ax.plot([first[j] - .5, first[j] - .5], [j - .5, j + .5], color=fg, lw=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"“{d['meta']['prompt']}”\n{len(order)} completions over {d['NB']} batch sizes", color=fg, fontsize=10, loc="left")
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
    fig.text(0.01, 0.01, "Rows: batch size 1 → 512 (top to bottom). Columns: token 0 → %d. Ground = same token as B = 1; colour = final completion class from its first differing token." % (list(ds.values())[0]["L"] - 1),
             fontsize=8, color=fg)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{GAL}/divergence_triptych_{style}.png", dpi=150, facecolor=bg); plt.close(fig); print("wrote triptych", style)


def film(d, prompt, style="paper", fps=12, hold=1.0):
    """The raster revealed token by token (MP4 + GIF)."""
    import imageio.v2 as iio
    NB, L = d["NB"], d["L"]
    img, ids, first, order, cols = raster_rgb(d, style)
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    T = tok(); base = d["toks"][0]
    tmp = f"{CACHE}/film_{prompt}_{style}"; os.makedirs(tmp, exist_ok=True)
    frames = []
    for t in range(1, L + 1):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=bg)
        ax = fig.add_axes([0.06, 0.20, 0.92, 0.72])
        shown = img.copy(); shown[:, t:] = to_rgb(bg)
        ax.imshow(shown, aspect="auto", interpolation="nearest", extent=[-.5, L - .5, NB - .5, -.5])
        for j in range(NB):
            if 0 <= first[j] < t:
                ax.plot([first[j] - .5, first[j] - .5], [j - .5, j + .5], color=fg, lw=1.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(bg)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
        n_cls = len({tuple(r[:t]) for r in d["toks"]})
        fig.text(0.06, 0.95, f"“{d['meta']['prompt']}”  ·  token {t}/{L}  ·  {n_cls} distinct prefixes across {NB} batch sizes", fontsize=12, color=fg)
        txt = T.decode(base[max(0, t - 40):t]).replace("\n", " ⏎ ")
        fig.text(0.06, 0.12, "B = 1: …" + txt[-150:], fontsize=9, color=fg, va="top")
        fig.text(0.06, 0.05, "rows = batch size 1…512 · ground = same token as B = 1 · colour = completion class after its first differing token", fontsize=8, color=fg, alpha=.8)
        fn = f"{tmp}/{t:04d}.png"; fig.savefig(fn, dpi=100, facecolor=bg); plt.close(fig); frames.append(fn)
    out = f"{GAL}/divergence_film_{prompt}_{style}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%04d.png",
                    "-vf", f"tpad=stop_mode=clone:stop_duration={hold},scale=1280:720", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", f"{out}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%04d.png",
                    "-vf", "fps=8,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5", f"{out}.gif"], check=True)
    print("wrote film", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=["feynman", "story", "sky"])
    ap.add_argument("--only", nargs="*", default=["raster", "drift", "texts", "firstdiv", "triptych", "summary"])
    a = ap.parse_args()
    ds = {p: load(p) for p in a.prompts if os.path.exists(f"{CACHE}/div_{p}.npz")}
    if "summary" in a.only:
        T = tok(); summ = {}
        for p, d in ds.items():
            ids, first, order, uniq = classes(d)
            ksig, klabels = kernel_sig_ids(d)
            fd = first[first >= 0]
            summ[p] = dict(prompt=d["meta"]["prompt"], NB=int(d["NB"]), L=int(d["L"]), n_completions=len(order),
                           n_depart=int((first >= 0).sum()), first_div_min=int(fd.min()) if fd.size else -1,
                           first_div_median=float(np.median(fd)) if fd.size else -1,
                           n_decode_sigs=len(klabels), within_batch_disagree_B=[int(b) for b, r in zip(d["Bs"], d["rows_disagree"]) if r.max() > 0],
                           run_to_run=d["meta"]["run_to_run"], wall_s=d["meta"]["wall_s"],
                           texts=[dict(cls=c, Bs=bs, first=f, pre=pre[-160:], post=post[:160]) for c, bs, f, pre, post in texts_table(d, T)])
            print(p, {k: v for k, v in summ[p].items() if k != "texts"})
        json.dump(summ, open(f"{CACHE}/div_summary.json", "w"), indent=1)
    for p, d in ds.items():
        if "raster" in a.only:
            for st in ("paper", "night", "riso"):
                plate_raster(d, p, st)
        if "drift" in a.only:
            plate_drift(d, p, "night"); plate_drift(d, p, "paper")
        if "texts" in a.only:
            plate_texts(d, p, "paper")
        if "film" in a.only:
            film(d, p, "paper")
    if "firstdiv" in a.only and ds:
        plate_firstdiv(ds, "paper"); plate_firstdiv(ds, "night")
    if "triptych" in a.only and len(ds) > 1:
        plate_triptych(ds, "paper"); plate_triptych(ds, "night")


if __name__ == "__main__":
    main()
