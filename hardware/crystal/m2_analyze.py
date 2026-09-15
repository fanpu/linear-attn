"""m2_analyze.py - crystal M2 (CPU): assemble the kernel-label cube, stats, load check, timing stats, previews.

Reads cache/cube_<dtype>/k*.npz and cache/timing_bf16_g64.npz; writes
  cache/cube_<dtype>_labels.npz  labels (256,256,256) uint16 [m-1, k-1, n-1], vocab (by volume), raw-name vocab
  cache/m2_stats_<dtype>.json, cache/preview/m2_*.png
    OMP_NUM_THREADS=4 python m2_analyze.py [--dtype bf16]
"""
import argparse, collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import colorcet as cc
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, PREV = f"{HERE}/cache", f"{HERE}/cache/preview"
G = 256
STACK = "GB10 + cuBLAS 13.1.1 in torch 2.14.0+cu130"

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("lattice_common", f"{HERE}/../lattice/common.py")
LC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LC)


def kernel_label(names):  # same as common.kernel_label (no torch import here)
    return LC.kernel_label(names.replace("std::enable_if<true, void>::type ", ""))


def apps_present(info):
    return bool(info["smi_before"]["other_apps"] or info["smi_after"]["other_apps"]
                or (info.get("check") or {}).get("other_apps"))


def assemble(dtype):
    files = sorted(glob.glob(f"{CACHE}/cube_{dtype}/k*.npz"))
    ks = [int(os.path.basename(f)[1:4]) for f in files]
    raw_id = np.zeros((G, G, G), np.uint16); raw_vocab = {}; fp_classes = {}; purity = {}
    slab_rows = []
    for f, k in zip(files, ks):
        z = np.load(f); info = json.loads(str(z["info"]))
        voc = [str(x) for x in z["names_vocab"]]
        lut = np.array([raw_vocab.setdefault(x, len(raw_vocab)) for x in voc], np.uint16)
        raw_id[:, k - 1, :] = lut[z["name_id"]]
        fpk = z["fp"].reshape(-1, 6)
        _, inv = np.unique(fpk, axis=0, return_inverse=True)
        fp_classes[k] = int(inv.max() + 1)
        slab_rows.append(dict(k=k, wall=float(z["wall"]), wall_fp=float(z["wall_fp"]), wall_names=float(z["wall_names"]),
                              temp=info["smi_before"]["temp"], util=info["smi_before"]["util"], time=info["smi_before"]["time"],
                              busy=apps_present(info), apps=sorted(set(info["smi_before"]["other_apps"] +
                                                                       info["smi_after"]["other_apps"])),
                              check=info.get("check") or {}, fp_inv=inv, name_id=lut[z["name_id"]].ravel()))
    raw_list = [None] * len(raw_vocab)
    for x, i in raw_vocab.items():
        raw_list[i] = x
    lab_of_raw = [kernel_label(x) for x in raw_list]
    lab_names = sorted(set(lab_of_raw))
    lab_idx = np.array([lab_names.index(l) for l in lab_of_raw], np.uint16)
    lab = lab_idx[raw_id]
    cnt = np.bincount(lab.ravel(), minlength=len(lab_names))
    rank = np.argsort(-cnt)                        # new id 0 = largest volume
    remap = np.empty_like(rank); remap[rank] = np.arange(len(rank))
    lab = remap[lab].astype(np.uint16)
    vocab = [lab_names[i] for i in rank]; counts = cnt[rank]
    # fingerprint -> kernel purity per slab
    for r in slab_rows:
        l = remap[lab_idx[r["name_id"]]]
        tab = collections.defaultdict(collections.Counter)
        for a, b in zip(r["fp_inv"].ravel(), l):
            tab[a][b] += 1
        purity[r["k"]] = sum(c.most_common(1)[0][1] for c in tab.values()) / len(l)
        del r["fp_inv"], r["name_id"]
    return ks, lab, vocab, counts, raw_list, fp_classes, purity, slab_rows


def regions(lab, nlab):
    s6 = ndimage.generate_binary_structure(3, 1)
    out = []
    for i in range(nlab):
        comp, n = ndimage.label(lab == i, structure=s6)
        sizes = np.bincount(comp.ravel())[1:] if n else np.array([], int)
        out.append(dict(label=i, n=int(n), ge64=int((sizes >= 64).sum()), largest=int(sizes.max()) if n else 0))
    return out


def cmap_for(n):
    cols = list(cc.glasbey_dark)
    while len(cols) < n:
        cols += cols
    return ListedColormap(cols[:n])


def show(ax, img, cm, n, title, xl, yl):
    ax.imshow(img, cmap=cm, vmin=-.5, vmax=n - .5, interpolation="nearest", origin="lower",
              extent=(.5, G + .5, .5, G + .5))
    ax.set_title(title, fontsize=9); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7)


def legend_panel(ax, vocab, counts, cm, top=30):
    ax.axis("off")
    tot = counts.sum()
    for i in range(min(top, len(vocab))):
        y = 1 - (i + .5) / top
        ax.add_patch(plt.Rectangle((0, y - .012), .04, .024, color=cm.colors[i], transform=ax.transAxes))
        ax.text(.06, y, f"{vocab[i][:70]}  {100*counts[i]/tot:.2f} %", fontsize=6.5, va="center", transform=ax.transAxes)
    if len(vocab) > top:
        ax.text(0, -.01, f"+ {len(vocab)-top} more labels ({100*counts[top:].sum()/tot:.3f} % of the cube)",
                fontsize=6.5, transform=ax.transAxes)


def previews(dtype, lab, vocab, counts):
    n = len(vocab); cm = cmap_for(n)
    decl = f"colour = profiler kernel label, glasbey_dark in order of cube volume (declared); nearest-neighbour; {STACK}, {dtype}"
    ks = [2, 3, 16, 17, 64, 97, 128, 191, 256]
    fig = plt.figure(figsize=(22, 15), dpi=110)
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, .9])
    for i, k in enumerate(ks):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        show(ax, lab[:, k - 1, :], cm, n, f"k = {k} ({'odd' if k % 2 else 'even'})", "n", "m")
    legend_panel(fig.add_subplot(gs[:, 3]), vocab, counts, cm)
    fig.suptitle(f"Crystal: kernel-label slabs at fixed k, m and n in [1, 256]. {decl}", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{PREV}/m2_slabs_k_{dtype}.png"); plt.close(fig)

    for axis, fixed, xl, name in [(2, [64, 127, 128, 200, 201, 256], "k", "mk"), (0, [16, 64, 127, 128, 200, 256], "n", "kn")]:
        fig = plt.figure(figsize=(22, 10.5), dpi=110)
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .9])
        for i, x in enumerate(fixed):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            if axis == 2:
                show(ax, lab[:, :, x - 1], cm, n, f"n = {x}", "k", "m")
            else:
                show(ax, lab[x - 1, :, :], cm, n, f"m = {x}", "n", "k")
        legend_panel(fig.add_subplot(gs[:, 3]), vocab, counts, cm)
        what = "(m, k) planes at fixed n" if axis == 2 else "(k, n) planes at fixed m"
        fig.suptitle(f"Crystal: {what}. {decl}", fontsize=11)
        fig.tight_layout(); fig.savefig(f"{PREV}/m2_planes_{name}_{dtype}.png"); plt.close(fig)

    # parity split of an even-k slab, as in lattice's dispatch_mosaic_split
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.2), dpi=110)
    for a, k in zip(ax, [128, 127, 256]):
        img = np.concatenate([np.repeat(lab[:, k - 1, 0::2], 2, 1), np.full((G, 6), n - 1), np.repeat(lab[:, k - 1, 1::2], 2, 1)], 1)
        a.imshow(img, cmap=cm, vmin=-.5, vmax=n - .5, interpolation="nearest", origin="lower", aspect="auto")
        a.set_title(f"k = {k}: odd n (left) | even n (right), columns doubled", fontsize=9)
        a.set_xticks([]); a.set_ylabel("m")
    fig.suptitle(f"Parity split. {decl}", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{PREV}/m2_parity_split_{dtype}.png"); plt.close(fig)


def timing_stats():
    p = f"{CACHE}/timing_bf16_g64.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p); info = json.loads(str(z["info"]))
    ref = z["ref"]                       # (j, t, ref, one, temp, util)
    r = ref[:, 2]; h = len(r) // 2
    sh = z["shapes"].astype(float); t = z["t_med"]
    gfl = 2 * sh[:, 0] * sh[:, 1] * sh[:, 2] / t / 1e9
    st = dict(wall_min=info["wall_s"] / 60, contaminated=info["contaminated"], start_temp=info["start_temp"],
              start_util=info["start_util"], smi_before=info["smi_before"], smi_after=info["smi_after"],
              ref_median_us=float(np.median(r) * 1e6), ref_std_rel=float(r.std() / r.mean()),
              ref_range_rel=float((r.max() - r.min()) / np.median(r)),
              ref_second_over_first_half=float(np.median(r[h:]) / np.median(r[:h])),
              one_median_us=float(np.median(ref[:, 3]) * 1e6),
              temp_min=float(ref[:, 4].min()), temp_max=float(ref[:, 4].max()),
              iqr_over_median_median=float(np.median(z["iqr"])), iqr_over_median_p99=float(np.percentile(z["iqr"], 99)),
              frac_iqr_gt_5pct=float((z["iqr"] > .05).mean()), gflops_median=float(np.median(gfl)),
              gflops_max=float(gfl.max()), resumes=json.loads(str(z["resumes"])), other_seen=json.loads(str(z["other_seen"])))
    # rank correlation of ln t residual (vs ln mkn) with measurement position: drift painting structure?
    pos = np.empty(len(t)); pos[z["order"]] = np.arange(len(t))
    x = np.log(sh.prod(1)); y = np.log(t)
    res = y - np.polyval(np.polyfit(x, y, 2), x)
    rk = lambda a: np.argsort(np.argsort(a))
    st["spearman_resid_vs_order"] = float(np.corrcoef(rk(res), rk(pos))[0, 1])
    # previews
    v = z["grid"]; V = len(v)
    L = np.log10(gfl).reshape(V, V, V)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    lo, hi = np.percentile(L, [1, 99.5])
    for a, j in zip(ax[0], sorted({V // 16, V // 4, V // 2, V - 1})):
        im = a.imshow(L[:, j, :], cmap="cet_fire", vmin=lo, vmax=hi, interpolation="nearest", origin="lower")
        a.set_title(f"log10 GFLOP/s, k = {v[j]}", fontsize=9); a.set_xlabel("n index"); a.set_ylabel("m index")
    fig.colorbar(im, ax=ax[0, 3], fraction=.046)
    a = ax[1, 0]; a.plot(ref[:, 1] / 60, r / np.median(r), lw=.8); a.set_title("reference (200,200,200) / median"); a.set_xlabel("min")
    a = ax[1, 1]; a.plot(ref[:, 1] / 60, ref[:, 4], lw=.8, color="C3"); a.set_title("GPU temperature (C)"); a.set_xlabel("min")
    a = ax[1, 2]; a.hist(np.clip(z["iqr"], 0, .2), 200); a.set_yscale("log"); a.set_title("IQR / median per shape (clipped 0.2)")
    a = ax[1, 3]; a.plot(ref[:, 1] / 60, ref[:, 3] * 1e6, lw=.8, color="C2"); a.set_title("1x1x1 overhead probe (us)")
    fig.suptitle(f"Timing volume 64^3 (v_j = 1+4j+(j mod 4)), bf16, {STACK}; cet_fire on log10 GFLOP/s (declared), "
                 f"1st-99.5th percentile", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{PREV}/m2_timing_bf16.png"); plt.close(fig)
    return st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--no-cube", action="store_true")
    args = ap.parse_args()
    os.makedirs(PREV, exist_ok=True)
    st = {}
    if not args.no_cube:
        ks, lab, vocab, counts, raw_list, fp_classes, purity, slabs = assemble(args.dtype)
        st.update(n_slabs=len(ks), n_kernel_labels=len(vocab), n_raw_names=len(raw_list),
                  label_volume=[(v, int(c)) for v, c in zip(vocab, counts)],
                  none_voxels=int(counts[vocab.index("(none)")]) if "(none)" in vocab else 0,
                  fp_classes_per_k_median=float(np.median(list(fp_classes.values()))),
                  fp_classes_total_k_fp=int(sum(fp_classes.values())),
                  fp_to_kernel_purity_mean=float(np.mean(list(purity.values()))),
                  fp_to_kernel_purity_min=(min(purity, key=purity.get), float(min(purity.values()))),
                  cube_wall_h=sum(s["wall"] for s in slabs) / 3600,
                  slab_wall_s=dict(median=float(np.median([s["wall"] for s in slabs])),
                                   max=float(max(s["wall"] for s in slabs))),
                  temp_range=[min(float(s["temp"]) for s in slabs), max(float(s["temp"]) for s in slabs)])
        if len(ks) == G:
            np.savez_compressed(f"{CACHE}/cube_{args.dtype}_labels.npz", labels=lab, vocab=np.array(vocab),
                                counts=counts, raw_vocab=np.array(raw_list))
            rg = regions(lab, len(vocab))
            st["regions"] = dict(total=sum(r["n"] for r in rg), ge64=sum(r["ge64"] for r in rg),
                                 per_label=[dict(r, name=vocab[r["label"]]) for r in rg])
            # regions inside each of the four parity sub-lattices (k mod 2, n mod 2): stride-2 grids in k and n,
            # 6-connected in the sub-lattice (m step 1, k and n step 2)
            sub = {}
            for pk in (0, 1):
                for pn in (0, 1):
                    rs = regions(np.ascontiguousarray(lab[:, pk::2, pn::2]), len(vocab))
                    sub[f"k{'odd' if pk == 0 else 'even'}_n{'odd' if pn == 0 else 'even'}"] = dict(
                        total=sum(r["n"] for r in rs), ge64=sum(r["ge64"] for r in rs),
                        labels_present=sum(r["n"] > 0 for r in rs))
            st["regions_parity_sublattices"] = sub
            # the lamellae: which labels change with n parity inside even vs odd k
            par = {}
            for kp, sl in [("even_k", slice(1, G, 2)), ("odd_k", slice(0, G, 2))]:
                A = lab[16:, sl, :]                      # m >= 17 (outside the small-m rows)
                par[kp] = float((A[:, :, 0::2][:, :, :127] != A[:, :, 1::2][:, :, :127]).mean())
            st["odd_vs_even_n_label_differs"] = par
        # load check
        chk = [s for s in slabs if s["check"]]
        busy = [s for s in chk if s["busy"]]
        st["load_check"] = dict(slabs_checked=len(chk), slabs_overlapping=len(busy),
                                agree_overlap=(sum(s["check"]["agree"] for s in busy), sum(s["check"]["n"] for s in busy)),
                                agree_all=(sum(s["check"]["agree"] for s in chk), sum(s["check"]["n"] for s in chk)),
                                overlap_apps=sorted({a.split(",")[1].strip() for s in busy for a in s["apps"]}),
                                overlap_ks=[s["k"] for s in busy],
                                differ_examples=[(s["k"], s["check"]["differ"][:3]) for s in chk if s["check"]["agree"] < s["check"]["n"]][:5])
        # reproduction of M1 dense slabs (same k): kernel labels
        rep = {}
        for k in [3, 16, 64, 127, 128, 256]:
            p = f"{CACHE}/m1_slab{k}.npz"
            if os.path.exists(p) and k in ks:
                z = np.load(p); l1 = np.array([kernel_label(x) for x in z["names"]]).reshape(G, G)
                l2 = np.array(vocab)[lab[:, k - 1, :]]
                rep[k] = float((l1 == l2).mean())
        st["m1_slab_label_agreement"] = rep
        json.dump(dict(st, slabs=[{a: b for a, b in s.items()} for s in slabs]),
                  open(f"{CACHE}/m2_stats_{args.dtype}.json", "w"), indent=1, default=str)
        if len(ks) == G:
            previews(args.dtype, lab, vocab, counts)
    if args.dtype == "bf16":
        ts = timing_stats()
        if ts:
            st["timing"] = ts
            json.dump(ts, open(f"{CACHE}/m2_timing_stats.json", "w"), indent=1, default=str)
    brief = {k: v for k, v in st.items() if k not in ("label_volume", "regions")}
    print(json.dumps(brief, indent=1, default=str)[:5000])
    if "regions" in st:
        print("regions", st["regions"]["total"], "ge64", st["regions"]["ge64"])
        for r in sorted(st["regions"]["per_label"], key=lambda r: -r["largest"])[:40]:
            print(f"  {r['n']:6d} regions, {r['ge64']:5d} >=64, largest {r['largest']:9d}  {r['name'][:80]}")


if __name__ == "__main__":
    main()
