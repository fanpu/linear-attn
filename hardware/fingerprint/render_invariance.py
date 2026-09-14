"""render_invariance.py - Fingerprint piece 1 plates from cache/<tag>.npz + .json (no GPU).

    OMP_NUM_THREADS=2 python render_invariance.py [--tag inv] [--only bitmaps atlas bands position]

Rows of every bitmap = batch size B (B = 1 at the top), columns = output element of the fixed target row,
value = number of bits in which the element differs from the B = 1 result (0 = identical). All of that is
measured; the colour maps, the split, and the inks are declared aesthetic choices.
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from PIL import Image
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"
INK = "#1b1b1b"; PAPER = "#f3eee3"; NIGHT = "#0b0b0e"; SEAM = "#160f1c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6,
                     "xtick.major.width": .5, "ytick.major.width": .5})

_spec = importlib.util.spec_from_file_location("lattice_common", "/home/fzeng/ml/research/hardware/lattice/common.py")
_lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lc)
short_kernel = _lc.short_kernel

NICE = {  # op name -> (label, what)
    "mm_tm_fp32": ("torch.mm fp32", "Thinking Machines' linspace example, D = 4096"),
    "mm_tm_bf16": ("torch.mm bf16", "same matrices in bf16"),
    "lin_q_bf16": ("q_proj bf16", "Qwen3 L13 1024 → 2048"),
    "lin_up_bf16": ("up_proj bf16", "Qwen3 L13 1024 → 3072"),
    "lin_down_bf16": ("down_proj bf16", "Qwen3 L13 3072 → 1024"),
    "lin_up_fp32": ("up_proj fp32", "same weights upcast"),
    "lin_head_fp32": ("lm_head fp32", "1024 → 151936 (first 4096 shown)"),
    "lin_up_prefill_bf16": ("up_proj prefill bf16", "[B, 32, 1024] → last position"),
    "rms_hf_bf16": ("RMSNorm (HF-style) bf16", "fp32 inside, bf16 out"),
    "rms_fused_bf16": ("F.rms_norm bf16", "fused kernel"),
    "rms_fused_fp32": ("F.rms_norm fp32", "fused kernel"),
    "softmax_vocab_fp32": ("softmax over 151936, fp32", "first 4096 shown"),
    "sdpa_decode_math_bf16": ("SDPA decode, math", "1 query × 256 keys, GQA 16/8"),
    "sdpa_prefill_math_bf16": ("SDPA prefill, math", "64 × 64 causal"),
    "sdpa_decode_eff_bf16": ("SDPA decode, mem-efficient", "1 query × 256 keys"),
    "sdpa_prefill_eff_bf16": ("SDPA prefill, mem-efficient", "64 × 64 causal"),
    "sdpa_decode_flash_bf16": ("SDPA decode, flash", "1 query × 256 keys"),
    "sdpa_prefill_flash_bf16": ("SDPA prefill, flash", "64 × 64 causal"),
    "sdpa_decode_cudnn_bf16": ("SDPA decode, cuDNN", "1 query × 256 keys"),
    "sdpa_prefill_cudnn_bf16": ("SDPA prefill, cuDNN", "64 × 64 causal"),
    "sdpa_decode_default_bf16": ("SDPA decode, default dispatch", "1 query × 256 keys"),
    "sdpa_prefill_default_bf16": ("SDPA prefill, default dispatch", "64 × 64 causal"),
    "layer_decode_bf16": ("one decoder layer, decode step", "L13, KV cache of 127"),
    "layer_prefill_bf16": ("one decoder layer, prefill", "L13, 32 tokens"),
    "model_prefill_fp32logits": ("full Qwen3-0.6B prefill → fp32 logits", "24-token prompt, first 4096 logits"),
}
ORDER = list(NICE)


def load(tag="inv"):
    z = np.load(f"{CACHE}/{tag}.npz")
    meta = json.load(open(f"{CACHE}/{tag}.json"))
    d = {"Bs": z["Bs"], "names": [str(n) for n in z["names"]], "meta": meta, "tm": float(z["tm_snippet"])}
    for n in d["names"]:
        d[n] = dict(bits=z[f"{n}__bits"], ndiff=z[f"{n}__ndiff"], maxabs=z[f"{n}__maxabs"], **meta["ops"][n])
    return d


def popcount(a):
    u = a.view(np.dtype(f"u{a.dtype.itemsize}"))
    b = np.unpackbits(u.view(np.uint8).reshape(*u.shape, u.itemsize), axis=-1)
    return b.sum(-1).astype(np.uint8)


def bitdiff(op, pos=0):
    """(nB, ncol) number of differing bits vs the B = 1 output."""
    b = op["bits"][pos]
    return popcount(b ^ b[:1])


def ulpdiff(op, pos=0):
    """(nB, ncol) signed distance in representable steps (ulps) from the B = 1 output (sign-magnitude ints)."""
    b = op["bits"][pos].astype(np.int64)
    mask = (1 << (8 * op["bits"].dtype.itemsize - 1)) - 1
    o = np.where(b < 0, -(b & mask), b)
    return o - o[:1]


def kernel_ids(op):
    """Per-B kernel signature -> small ints in order of first appearance, plus the vocabulary (short labels)."""
    sigs = [tuple(k) for k in op["kernels"]]
    vocab, ids = [], []
    for s in sigs:
        if s not in vocab:
            vocab.append(s)
        ids.append(vocab.index(s))
    labels = [" + ".join(sorted({short_kernel(k) for k in v if "Memset" not in k and "Memcpy" not in k})) or "(none)"
              for v in vocab]
    return np.array(ids), labels


def switches(ids):
    return np.nonzero(np.diff(ids))[0] + 1     # index j where signature(j) != signature(j-1)


def stack_line(meta):
    s = meta["stack"]
    return (f"GB10 (sm_{s['capability'][0]}{s['capability'][1]}) | driver {s['driver_name'].split(',')[0]} | CUDA {s['cuda']} | "
            f"torch {s['torch']} | cuBLAS {s['cublas']} | cuDNN {s['cudnn']} | tf32 off | {s['date']}")


def save_rgb(img, name, row_scale=1):
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if row_scale > 1:
        a = np.repeat(a, row_scale, 0)
    Image.fromarray(a).save(f"{GAL}/{name}.png", optimize=True)
    print("wrote", name, a.shape)


# ------------------------------------------------------------------ colour mappings (declared)
def cmap_bits(nbits, style):
    """0 differing bits = ground; 1..nbits = a sequential ramp."""
    if style == "night":
        ramp = plt.get_cmap("cet_fire")(np.linspace(0.15, 1, nbits))
        ground = to_rgb(NIGHT)
    elif style == "oslo":
        ramp = plt.get_cmap("cmc.oslo")(np.linspace(0.25, 1, nbits))
        ground = to_rgb(NIGHT)
    elif style == "paper":
        ramp = plt.get_cmap("Greys")(np.linspace(0.55, 1, nbits))
        ground = to_rgb(PAPER)
    else:
        raise ValueError(style)
    cols = np.vstack([[*ground, 1], ramp])
    return ListedColormap(cols)


def rgb_bits(bd, nbits, style):
    return cmap_bits(nbits, style)(bd)[..., :3]


def rgb_spectral(ud):
    """Signed ulp distance, Sohl-Dickstein Spectral split at 0: identical = dark seam, +ulps → red/orange/yellow,
    -ulps → purple/blue/green, each side rank-normalised separately."""
    img = P.render_split(ud.astype(float), "sd_spectral", near_boundary="small")
    img[ud == 0] = to_rgb(SEAM)      # rank ties at 0 would otherwise land mid-ramp; identical = the dark seam
    return img


def rgb_riso(op, inks=("#ff48b0", "#0078bf")):
    """Two-ink overprint: pink = element differs when the target row is FIRST in the batch,
    blue = differs when it is LAST. Overprint (dark) = both."""
    a = (bitdiff(op, 0) > 0).astype(float)
    b = (bitdiff(op, 2) > 0).astype(float)
    return P.overprint([a, b], list(inks))


def cat_colors(n):
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    if n <= 20:
        return base[:n]
    extra = plt.get_cmap("cet_glasbey_dark")(np.arange(n - 20) / max(1, n - 21))[:, :3]
    return np.vstack([base, extra])


# ------------------------------------------------------------------ plates
def plate_bitmap(d, name, style, pos=0, row_scale=4):
    op = d[name]; Bs = d["Bs"]; nB = len(Bs)
    nbits = 8 * op["bits"].dtype.itemsize
    if style == "spectral":
        img = rgb_spectral(ulpdiff(op, pos))
    elif style == "riso":
        img = rgb_riso(op)
    else:
        img = rgb_bits(bitdiff(op, pos), nbits, style)
    ids, labels = kernel_ids(op)
    ncol = img.shape[1]
    # print version: native columns, rows x row_scale, kernel strip on the left (16 px + 4 px gap)
    cols = cat_colors(len(labels))
    strip = np.repeat(cols[ids][:, None, :], 16, 1)
    gap = np.ones((nB, 4, 3)) * (to_rgb(PAPER) if style in ("paper", "riso") else to_rgb(NIGHT))
    save_rgb(np.concatenate([strip, gap, img], 1), f"bitmap_{name}_{style}_print", row_scale)

    # labelled plate
    dark = style in ("night", "oslo", "spectral")
    fg = "#e8e4d8" if dark else INK
    bg = NIGHT if dark else PAPER
    fig = plt.figure(figsize=(16, 9.6), facecolor=bg)
    axk = fig.add_axes([0.045, 0.14, 0.012, 0.74]); axm = fig.add_axes([0.062, 0.14, 0.70, 0.74])
    axc = fig.add_axes([0.775, 0.14, 0.17, 0.74])
    axk.imshow(cols[ids][:, None, :], aspect="auto", interpolation="nearest", extent=[0, 1, nB + .5, .5])
    axk.set_xticks([]); axk.set_ylabel("batch size B  (target row is %s in the batch)" % ["first", "middle", "last"][pos], color=fg)
    axk.tick_params(colors=fg); axk.set_yticks(np.r_[1, 64:nB + 1:64])
    for sp in axk.spines.values(): sp.set_edgecolor(fg)
    axm.imshow(img, aspect="auto", interpolation="nearest", extent=[-.5, ncol - .5, nB + .5, .5])
    axm.set_yticks([]); axm.set_xlabel("output element (of %d%s)" % (op["D"], "" if op["D"] == ncol else f", first {ncol} shown"), color=fg)
    axm.tick_params(colors=fg)
    for sp in axm.spines.values(): sp.set_edgecolor(fg)
    for j in switches(ids):
        axm.axhline(Bs[j] - .5, color=fg, lw=.4, alpha=.6)
    # side panel: fraction of the row that differs, per B and position
    frac = op["ndiff"] / op["D"]
    for pi, (lab, ls) in enumerate(zip(["first", "middle", "last"], ["-", "--", ":"])):
        axc.plot(frac[pi], Bs, ls, color=fg, lw=.8, label=f"target {lab}")
    axc.set_ylim(nB + .5, .5); axc.set_xlim(-0.02, 1.02); axc.set_yticks([])
    axc.set_xlabel("fraction of elements ≠ B = 1", color=fg); axc.tick_params(colors=fg)
    axc.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=fg)
    axc.set_facecolor(bg)
    for sp in axc.spines.values(): sp.set_edgecolor(fg)
    for j in switches(ids):
        axc.axhline(Bs[j] - .5, color=fg, lw=.4, alpha=.6)
    lab, what = NICE.get(name, (name, ""))
    fig.text(0.045, 0.945, f"Fingerprint · {lab}", fontsize=16, color=fg, weight="bold")
    fig.text(0.045, 0.915, f"{what}. One fixed input row inside batches of size 1…{nB}: bits of its output that differ from the B = 1 result.", fontsize=9.5, color=fg)
    if style == "spectral":
        cap = "Colour: signed ulp distance from B = 1, Spectral split at 0 (dark seam = identical; red side = larger, purple side = smaller; ranks per side)."
    elif style == "riso":
        cap = "Two inks: pink = element differs when the row is first in the batch, blue = when it is last; overprint = both."
    else:
        cap = f"Colour: number of differing bits (0 = ground, 1…{nbits} = ramp). Horizontal rules and the left strip: CUDA kernel signature per B (torch.profiler)."
    fig.text(0.045, 0.082, cap, fontsize=8.5, color=fg)
    for i, l in enumerate(labels[:6]):
        fig.text(0.045 + i * 0.155, 0.055, "■", color=cols[i], fontsize=10)
        fig.text(0.058 + i * 0.155, 0.055, l[:34], color=fg, fontsize=7)
    fig.text(0.045, 0.025, stack_line(d["meta"]) + f" | checks (repeat, filler swap) all pass: {all(all(v) for v in op['checks'].values())}", fontsize=7, color=fg, alpha=.8)
    fig.savefig(f"{GAL}/bitmap_{name}_{style}.png", dpi=150, facecolor=bg)
    plt.close(fig); print("wrote", f"bitmap_{name}_{style}")


def pool_cols(img, ncol_out):
    """Max-pool columns of an (nB, ncol) array to ncol_out (keeps any nonzero visible)."""
    nB, nc = img.shape
    if nc <= ncol_out:
        return img
    f = nc // ncol_out
    return img[:, :f * ncol_out].reshape(nB, ncol_out, f).max(-1)


def plate_atlas(d, style):
    names = [n for n in ORDER if n in d["names"]]
    nB = len(d["Bs"]); dark = style != "paper"
    fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    nc, nr = 5, int(np.ceil(len(names) / 5))
    fig, axs = plt.subplots(nr, nc, figsize=(20, 3.6 * nr + 1.2), facecolor=bg)
    for ax, n in zip(axs.flat, names):
        op = d[n]; nbits = 8 * op["bits"].dtype.itemsize
        bd = pool_cols(bitdiff(op, 0), 512)
        if style == "spectral":
            ud = ulpdiff(op, 0); ud = ud[:, :512 * (ud.shape[1] // 512)] if ud.shape[1] >= 512 else ud
            if ud.shape[1] > 512:
                ud = ud.reshape(nB, 512, -1); ud = np.take_along_axis(ud, np.abs(ud).argmax(-1)[..., None], -1)[..., 0]
            img = rgb_spectral(ud)
        else:
            img = rgb_bits(bd, nbits, style)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ids, labels = kernel_ids(op)
        for j in switches(ids):
            ax.axhline(j - .5, color=fg, lw=.35, alpha=.5)
        nb = int((op["ndiff"][0] > 0).sum())
        lab, _ = NICE.get(n, (n, ""))
        ax.set_title(f"{lab}\n{nb}/{nB} batch sizes differ · {len(labels)} kernel signatures · max|Δ| = {op['maxabs'].max():.2g}",
                     fontsize=8.5, color=fg, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(fg); sp.set_linewidth(.5)
    for ax in axs.flat[len(names):]:
        ax.axis("off")
    fig.text(0.02, 0.985, "Fingerprint atlas: batch invariance of 25 ops on one GB10", fontsize=15, color=fg, weight="bold", va="top")
    fig.text(0.02, 0.962, "Each tile: rows = batch size 1…%d (top to bottom), columns = output elements of one fixed row (max-pooled to 512), "
             "colour = %s. Rules = kernel-signature changes." % (nB, "Spectral split of the signed ulp distance from B = 1" if style == "spectral" else "bits that differ from the B = 1 result"),
             fontsize=9, color=fg, va="top")
    fig.text(0.02, 0.008, stack_line(d["meta"]), fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    fig.savefig(f"{GAL}/atlas_{style}.png", dpi=130, facecolor=bg); plt.close(fig); print("wrote atlas", style)


def plate_bands(d, names, style="paper"):
    """Step plots: fraction of differing elements vs B, with kernel bands named. Plotter idiom."""
    dark = style != "paper"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    Bs = d["Bs"]
    fig, axs = plt.subplots(len(names), 1, figsize=(16, 2.6 * len(names) + 1), facecolor=bg, sharex=True)
    for ax, n in zip(np.atleast_1d(axs), names):
        op = d[n]; ids, labels = kernel_ids(op); cols = cat_colors(len(labels))
        frac = op["ndiff"] / op["D"]
        sw = np.r_[0, switches(ids), len(Bs)]
        for a, b in zip(sw[:-1], sw[1:]):
            ax.axvspan(Bs[a] - .5, Bs[b - 1] + .5, color=cols[ids[a]], alpha=.22, lw=0)
            if b - a >= 6:
                ax.text((Bs[a] + Bs[b - 1]) / 2, 1.03, labels[ids[a]][:28], ha="center", va="bottom", fontsize=6.5, color=fg, rotation=0)
        for pi, (lab, ls) in enumerate(zip(["first", "middle", "last"], ["-", "--", ":"])):
            ax.step(Bs, frac[pi], ls, where="mid", color=fg, lw=.9, label=f"target row {lab}")
        ax.set_ylim(-.03, 1.03); ax.set_xlim(.5, Bs[-1] + .5)
        ax.set_ylabel("fraction ≠ B=1", color=fg); ax.tick_params(colors=fg); ax.set_facecolor(bg)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
        ax.text(0.005, 0.92, NICE.get(n, (n, ""))[0], transform=ax.transAxes, fontsize=10, color=fg, weight="bold")
    np.atleast_1d(axs)[0].legend(loc="center right", fontsize=7, frameon=False, labelcolor=fg)
    np.atleast_1d(axs)[-1].set_xlabel("batch size B", color=fg)
    fig.text(0.01, 0.005, stack_line(d["meta"]) + " | bands = CUDA kernel signature (torch.profiler), labelled where wide enough", fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(f"{GAL}/bands_{style}.png", dpi=140, facecolor=bg); plt.close(fig); print("wrote bands", style)


def plate_position(d, name, style="night"):
    """Triptych: same op, the target row first / middle / last in the batch."""
    op = d[name]; nbits = 8 * op["bits"].dtype.itemsize; nB = len(d["Bs"])
    dark = style != "paper"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig, axs = plt.subplots(1, 3, figsize=(20, 7.2), facecolor=bg)
    for pi, ax in enumerate(axs):
        img = rgb_spectral(ulpdiff(op, pi)) if style == "spectral" else rgb_bits(bitdiff(op, pi), nbits, style)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(f"target row {['first', 'middle', 'last'][pi]} in the batch · {int((op['ndiff'][pi] > 0).sum())}/{nB} differ", color=fg, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
    same01 = np.array_equal(op["bits"][0], op["bits"][1]); same02 = np.array_equal(op["bits"][0], op["bits"][2])
    fig.suptitle(f"Fingerprint · {NICE.get(name, (name, ''))[0]}: does the row's position in the batch matter?  first==middle: {same01}, first==last: {same02}",
                 color=fg, fontsize=13, x=0.02, ha="left")
    fig.text(0.01, 0.01, stack_line(d["meta"]), fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(f"{GAL}/position_{name}_{style}.png", dpi=130, facecolor=bg); plt.close(fig); print("wrote position", name, style)


def summary(d):
    rows = []
    for n in ORDER:
        if n not in d["names"]:
            continue
        op = d[n]; ids, labels = kernel_ids(op)
        rows.append(dict(op=n, label=NICE[n][0], D=op["D"], dtype=op["dtype"],
                         nB_diff=[int((op["ndiff"][p] > 0).sum()) for p in range(3)],
                         frac_diff_median=[float(np.median(op["ndiff"][p][1:] / op["D"])) for p in range(3)],
                         maxabs=float(op["maxabs"].max()), n_signatures=len(labels), n_switches=int(len(switches(ids))),
                         pos_first_eq_last=bool(np.array_equal(op["bits"][0], op["bits"][2])),
                         checks_pass=bool(all(all(v) for v in op["checks"].values())), kernels=labels))
    json.dump(dict(tm_snippet=d["tm"], nB=int(len(d["Bs"])), ops=rows), open(f"{CACHE}/inv_summary.json", "w"), indent=1)
    print(f"{'op':28s} {'D':>7s} diffB(f/m/l)   med.frac   max|d|   sigs sw  f==l checks")
    for r in rows:
        print(f"{r['op']:28s} {r['D']:7d} {r['nB_diff'][0]:3d}/{r['nB_diff'][1]:3d}/{r['nB_diff'][2]:3d}  "
              f"{r['frac_diff_median'][0]:.3f}  {r['maxabs']:8.2g}  {r['n_signatures']:3d} {r['n_switches']:3d}  {r['pos_first_eq_last']!s:5s} {r['checks_pass']}")


HEROES = ["mm_tm_bf16", "lin_down_bf16", "lin_head_fp32", "sdpa_decode_cudnn_bf16", "layer_decode_bf16", "model_prefill_fp32logits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="inv")
    ap.add_argument("--only", nargs="*", default=["summary", "bitmaps", "atlas", "bands", "position"])
    ap.add_argument("--heroes", nargs="*", default=HEROES)
    a = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    d = load(a.tag)
    heroes = [h for h in a.heroes if h in d["names"]]
    if "summary" in a.only:
        summary(d)
    if "bitmaps" in a.only:
        for h in heroes:
            for st in ("night", "spectral", "paper", "riso", "oslo"):
                plate_bitmap(d, h, st)
    if "atlas" in a.only:
        for st in ("night", "paper", "spectral"):
            plate_atlas(d, st)
    if "bands" in a.only:
        plate_bands(d, heroes, "paper"); plate_bands(d, heroes, "night")
    if "position" in a.only:
        for h in heroes[:3]:
            plate_position(d, h, "night")


if __name__ == "__main__":
    main()
