"""m1_analyze.py - crystal M1 (CPU): numbers and matplotlib previews from cache/m1_*.npz.

Writes cache/m1_stats.json, cache/m1_mapping.md (class -> kernel tables) and cache/preview/*.png.
    OMP_NUM_THREADS=4 python m1_analyze.py
"""
import collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import colorcet as cc
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, PREV = f"{HERE}/cache", f"{HERE}/cache/preview"
LATTICE = os.path.normpath(f"{HERE}/../lattice")
_spec = importlib.util.spec_from_file_location("lattice_common", f"{LATTICE}/common.py")
LC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LC)   # hardware/lattice/common.py
CUBE = 256 ** 3


def kernel_label(names):
    """lattice's kernel_label; also strips 'std::enable_if<true, void>::type ', which lattice's short_kernel
    only handles in its '!(false)' form (otherwise a gemvx kernel is labelled 'enable_if<T>')."""
    return LC.kernel_label(names.replace("std::enable_if<true, void>::type ", ""))


def load(name):
    p = f"{CACHE}/m1_{name}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    d = {k: z[k] for k in z.files}
    d["info"] = json.loads(str(d["info"]))
    return d


def fp_keys(fp):
    return [tuple(r) for r in fp]


def area_ids(labels):
    """labels -> ids sorted by count (0 = most common), vocabulary [(label, count)]."""
    c = collections.Counter(labels)
    vocab = [k for k, _ in c.most_common()]
    idx = {k: i for i, k in enumerate(vocab)}
    return np.array([idx[l] for l in labels]), [(k, c[k]) for k in vocab]


def purity(a, b):
    """fraction of items whose a-class's majority b-label equals their b-label."""
    tab = collections.defaultdict(collections.Counter)
    for x, y in zip(a, b):
        tab[x][y] += 1
    return sum(c.most_common(1)[0][1] for c in tab.values()) / len(a), tab


def gpu_state(info):
    s0, s1 = info.get("smi_before", {}), info.get("smi_after", {})
    return dict(other_apps_before=s0.get("other_apps"), other_apps_after=s1.get("other_apps"),
                util_before=s0.get("util"), util_after=s1.get("util"), time=s0.get("time"),
                temp=s0.get("temp"), loadavg=s0.get("loadavg"))


def cat_cmap(n):
    return ListedColormap(cc.glasbey_dark[:max(n, 1)] if n <= len(cc.glasbey_dark) else
                          (cc.glasbey_dark * (n // len(cc.glasbey_dark) + 1))[:n])


def main():
    os.makedirs(PREV, exist_ok=True)
    st, md = {}, []

    # ---------- rate and determinism
    r1, r2 = load("rate1"), load("rate2")
    if r1 is not None:
        st["rate"] = dict(n=len(r1["fp"]), wall1_s=float(r1["wall"]), us_per_shape1=float(r1["wall"]) / len(r1["fp"]) * 1e6,
                          state1=gpu_state(r1["info"]))
        if r2 is not None:
            same = (r1["fp"] == r2["fp"]).all(1)
            st["rate"].update(wall2_s=float(r2["wall"]), us_per_shape2=float(r2["wall"]) / len(r2["fp"]) * 1e6,
                              state2=gpu_state(r2["info"]), identical_frac=float(same.mean()),
                              n_differ=int((~same).sum()))
        us = min(st["rate"]["us_per_shape1"], st["rate"].get("us_per_shape2", 1e9))
        st["rate"]["proj_dense_fp_h"] = CUBE * us * 1e-6 / 3600
        st["rate"]["proj_stride2_fp_h"] = CUBE / 8 * us * 1e-6 / 3600

    # ---------- slice check against lattice g256
    sl = load("slice")
    if sl is not None:
        z = np.load(f"{LATTICE}/cache/time_bf16_g256_k4096.npz")
        G = len(z["grid"]); lat = z["fp"].reshape(G, G, 6)
        m, n = sl["shapes"][:, 0], sl["shapes"][:, 2]
        ref = lat[m - 1, n - 1]
        eq = (ref == sl["fp"]).all(1)
        st["slice"] = dict(n=len(eq), identical=int(eq.sum()), frac=float(eq.mean()),
                           lattice_npz="hardware/lattice/cache/time_bf16_g256_k4096.npz",
                           lattice_date=json.loads(str(z["info"]))["stack"]["date"],
                           n_classes_crystal=len(set(fp_keys(sl["fp"]))), n_classes_lattice_sub=len(set(fp_keys(ref))),
                           state=gpu_state(sl["info"]))
        v = np.unique(m)
        idc, voc = area_ids(fp_keys(sl["fp"]))
        idl = np.array([dict((k, i) for i, (k, _) in enumerate(voc)).get(t, -1) for t in fp_keys(ref)])
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), dpi=110)
        cm = cat_cmap(len(voc))
        for a, arr, t in [(ax[0], idl, "lattice g256 (2026-09-13), sub-grid"),
                          (ax[1], idc, "crystal k=4096 (this run)")]:
            a.imshow(np.ma.masked_less(arr.reshape(64, 64), 0), cmap=cm, vmin=-.5, vmax=len(voc) - .5,
                     interpolation="nearest", origin="lower"); a.set_title(t, fontsize=10)
        ax[2].imshow(eq.reshape(64, 64), cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="lower")
        ax[2].set_title(f"bitwise equal: {eq.sum()}/{len(eq)}", fontsize=10)
        for a in ax:
            tk = np.arange(0, 64, 8); a.set_xticks(tk, v[tk], fontsize=7); a.set_yticks(tk, v[tk], fontsize=7)
            a.set_xlabel("n"); a.set_ylabel("m")
        fig.suptitle("Slice check: fingerprint classes at k = 4096 on m, n = 1+4j+(j mod 4) (glasbey, classes by area; "
                     "GB10 + cuBLAS 13.1.1, torch 2.14.0+cu130, bf16)", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{PREV}/slice_k4096_check.png"); plt.close(fig)

    # ---------- 2000-shape profiler sample
    kn = load("kern")
    if kn is not None:
        labs = [kernel_label(x) for x in kn["names"]]
        fk = [(int(k),) + t for k, t in zip(kn["shapes"][:, 1], fp_keys(kn["fp"]))]
        c = collections.Counter(labs)
        # classes as (k, fingerprint): fingerprint values are not comparable across k (different masters)
        tab = collections.defaultdict(collections.Counter)
        for f, l in zip(fk, labs):
            tab[f][l] += 1
        multi = {str(f): dict(v) for f, v in tab.items() if len(v) > 1}
        # the same raw 6-tuple at different k
        raw = collections.defaultdict(set)
        for f, l in zip(fk, labs):
            raw[f[1:]].add(f[0])
        st["kern"] = dict(n=len(labs), n_kernel_labels=len(c), n_classes_k_fp=len(tab),
                          n_classes_with_repeats=sum(sum(v.values()) > 1 for v in tab.values()),
                          multi_kernel_classes=multi, fp_tuple_at_several_k=sum(len(s) > 1 for s in raw.values()),
                          wall_fp_s=float(kn["wall_fp"]), wall_names_s=float(kn["wall_names"]),
                          names_us_per_shape=float(kn["wall_names"]) / len(labs) * 1e6,
                          label_counts=c.most_common(), state=gpu_state(kn["info"]))
        md.append("## 2000 random shapes in [1,256]^3: kernel labels\n\n| kernel label | shapes |\n|---|---|")
        md += [f"| `{l}` | {n} |" for l, n in c.most_common()]
        md.append("")

    # ---------- dense slabs and planes
    st["slabs"], st["planes"] = {}, {}
    ks_all = collections.Counter()
    for p in sorted(glob.glob(f"{CACHE}/m1_slab*.npz")) + sorted(glob.glob(f"{CACHE}/m1_plane*.npz")):
        name = os.path.basename(p)[3:-4]
        d = load(name)
        labs = np.array([kernel_label(x) for x in d["names"]])
        is_slab = name.startswith("slab")
        if is_slab:
            fk = fp_keys(d["fp"])
        else:
            fk = [(int(k),) + t for k, t in zip(d["shapes"][:, 1], fp_keys(d["fp"]))]
        p_fk, tab = purity(fk, labs)
        p_kf, _ = purity(labs, fk)
        ids_f, voc_f = area_ids(fk)
        ids_k, voc_k = area_ids(list(labs))
        multi = [(f, dict(v)) for f, v in tab.items() if len(v) > 1]
        multi.sort(key=lambda x: -sum(x[1].values()))
        # shapes whose class shares its fingerprint with another kernel (minority members)
        minority = sum(sum(v.values()) - max(v.values()) for _, v in multi)
        ks_all.update(labs.tolist())
        rec = dict(n=len(labs), n_fp_classes=len(voc_f), n_kernel_labels=len(voc_k), fp_to_kernel_purity=p_fk,
                   kernel_to_fp_purity=p_kf, n_multi_kernel_classes=len(multi), minority_shapes=int(minority),
                   fp_us_per_shape=float(d["wall_fp"]) / len(labs) * 1e6,
                   names_us_per_shape=float(d["wall_names"]) / len(labs) * 1e6, state=gpu_state(d["info"]))
        (st["slabs"] if is_slab else st["planes"])[name] = rec
        # mapping table
        md.append(f"## {name}: {rec['n_fp_classes']} fingerprint classes, {rec['n_kernel_labels']} kernel labels, "
                  f"purity fp->kernel {p_fk:.4f}, kernel->fp {p_kf:.4f}\n")
        if is_slab:
            md.append("| class (by area) | shapes | kernel label(s) : shapes |\n|---|---|---|")
            for i, (f, cnt) in enumerate(voc_f):
                md.append(f"| {i} | {cnt} | " + "; ".join(f"`{l}` : {q}" for l, q in tab[f].most_common()) + " |")
        else:
            md.append(f"(m, k) plane, classes are (k, fingerprint); {len(multi)} classes hold > 1 label. Largest:\n")
            md.append("| k | shapes | kernel label(s) : shapes |\n|---|---|---|")
            for f, v in multi[:15]:
                md.append(f"| {f[0]} | {sum(v.values())} | " + "; ".join(f"`{l}` : {q}" for l, q in
                                                                        collections.Counter(v).most_common()) + " |")
        md.append("")
        # preview
        G = 256
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.4), dpi=100)
        axis2 = "n" if is_slab else "k"
        titles = []
        if is_slab:
            ax[0].imshow(ids_f.reshape(G, G), cmap=cat_cmap(len(voc_f)), vmin=-.5, vmax=len(voc_f) - .5,
                         interpolation="nearest", origin="lower", extent=(.5, G + .5, .5, G + .5))
            titles.append(f"fingerprint classes ({len(voc_f)}, glasbey by area)")
        else:
            # along m within each k row: mark where the fingerprint changes (classes are per k)
            F = np.array([hash(t) for t in fp_keys(d["fp"])]).reshape(G, G)
            edge = np.zeros((G, G), bool); edge[1:, :] = F[1:, :] != F[:-1, :]; edge[:, 1:] |= False
            ax[0].imshow(edge, cmap="gray_r", interpolation="nearest", origin="lower", extent=(.5, G + .5, .5, G + .5))
            titles.append("fingerprint changes between m-1 and m (per k)")
        cmk = cat_cmap(len(voc_k))
        ax[1].imshow(ids_k.reshape(G, G), cmap=cmk, vmin=-.5, vmax=len(voc_k) - .5, interpolation="nearest",
                     origin="lower", extent=(.5, G + .5, .5, G + .5))
        titles.append(f"profiler kernel label ({len(voc_k)}, glasbey by area)")
        for a, t in zip(ax, titles):
            a.set_title(t, fontsize=10); a.set_xlabel(axis2); a.set_ylabel("m")
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmk.colors[i]) for i in range(min(12, len(voc_k)))]
        ax[1].legend(handles, [f"{l[:60]} ({c})" for l, c in voc_k[:12]], fontsize=6, loc="upper left",
                     bbox_to_anchor=(1.01, 1), frameon=False)
        fixed = f"k = {name[4:]}" if is_slab else f"n = {name[5:]}"
        fig.suptitle(f"{name}: torch.mm, {fixed}, m and {axis2} in [1, 256] (GB10 + cuBLAS 13.1.1, torch 2.14.0+cu130, "
                     f"bf16, 2026-09-15). fp->kernel purity {p_fk:.3f}, kernel->fp {p_kf:.3f}", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{PREV}/{name}.png"); plt.close(fig)
    st["dense_kernel_vocab"] = len(ks_all)

    # ---------- load dependence
    loads = {}
    for p in sorted(glob.glob(f"{CACHE}/m1_load*.npz")) + ([f"{CACHE}/m1_kern.npz"] if kn is not None else []):
        name = os.path.basename(p)[3:-4]; d = load(name)
        loads[name] = d
    if loads:
        base = next(iter(loads))
        st["load"] = {}
        for name, d in loads.items():
            same = (d["fp"] == loads[base]["fp"]).all(1)
            s = gpu_state(d["info"])
            st["load"][name] = dict(identical_to=base, frac=float(same.mean()), n_differ=int((~same).sum()),
                                    busy_other_process=bool(s["other_apps_before"] or s["other_apps_after"]),
                                    burn_calls=int(d["burn_calls"]) if "burn_calls" in d else None, **s)

    # ---------- decision
    if "rate" in st:
        us_fp = st["rate"]["us_per_shape1"]
        us_nm = np.median([r["names_us_per_shape"] for r in st["slabs"].values()]) if st["slabs"] else None
        st["projection"] = dict(fp_only_dense_h=CUBE * us_fp * 1e-6 / 3600,
                                fp_plus_names_dense_h=(CUBE * (us_fp + us_nm) * 1e-6 / 3600) if us_nm else None,
                                fp_plus_names_stride2_h=(CUBE / 8 * (us_fp + us_nm) * 1e-6 / 3600) if us_nm else None)
    json.dump(st, open(f"{CACHE}/m1_stats.json", "w"), indent=1, default=str)
    open(f"{CACHE}/m1_mapping.md", "w").write("\n".join(md) + "\n")
    brief = {k: v for k, v in st.items() if k not in ("kern",)}
    print(json.dumps(brief, indent=1, default=str)[:6000])
    if "kern" in st:
        print(json.dumps({k: v for k, v in st["kern"].items() if k != "label_counts"}, default=str)[:3000])


if __name__ == "__main__":
    main()
