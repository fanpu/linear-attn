"""common.py - load cached sweeps, name kernels, build categorical maps (no GPU)."""
import collections, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"


def load(kind, tag):
    z = np.load(f"{CACHE}/{kind}_{tag}.npz")
    d = {k: z[k] for k in z.files}
    d["info"] = json.loads(str(d["info"]))
    d["G"] = len(d["grid"])
    return d


def targs(s):
    """Top-level template arguments of the first template in s."""
    i = s.find("<")
    if i < 0:
        return s, []
    depth, cur, out = 0, "", []
    for ch in s[i + 1:]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                out.append(cur.strip()); break
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    return s[:i], out


def short_kernel(full):
    """One GPU kernel name -> compact label that keeps the numeric/boolean template parameters."""
    m = re.search(r"(cutlass_\w+|nvjet_\w+)", full)
    if m:
        return m.group(1).replace("cutlass_", "cutlass:").replace("nvjet_sm121_", "nvjet:")
    head, args = targs(full.replace("std::enable_if<!(false), void>::type ", "").replace("void ", ""))
    head = head.split("::")[-2] if head.endswith("::kernel") else head.split("::")[-1]
    keep = [a[0].upper() if a in ("true", "false") else a for a in args if re.fullmatch(r"-?\d+|true|false", a)]
    return f"{head}<{','.join(keep)}>" if keep else head


def kernel_label(names):
    ks = [short_kernel(x) for x in names.split(" | ") if x and "Memset" not in x]
    main = [k for k in ks if not k.startswith("splitKreduce")]
    lab = "+".join(main) if main else "+".join(ks)
    if any(k.startswith("splitKreduce") for k in ks):
        lab += " +splitK"
    return lab or "(none)"


def categorize(labels, G, min_frac=0.0):
    """labels (G*G) -> id map sorted by area (0 = largest), and ordered vocabulary with counts."""
    c = collections.Counter(labels)
    vocab = [k for k, _ in c.most_common()]
    idx = {k: i for i, k in enumerate(vocab)}
    return np.array([idx[l] for l in labels]).reshape(G, G), [(k, c[k]) for k in vocab]


def fp_labels(fp):
    return [tuple(r) for r in fp]


def gflops(d):
    k = d["info"]["stack"]["k"]
    return (2.0 * d["m"] * d["n"] * k / d["t_med"] / 1e9).reshape(d["G"], d["G"])


def stack_line(info, extra=""):
    s = info["stack"]
    return (f"GB10 ({s['arch']}) + cuBLAS {s['cublas']} (cublasLt {s['cublasLt']}) | driver {s['gpu'].split(', ')[-1]} | "
            f"CUDA {s['cuda']} | torch {s['torch']} | {s['dtype']}{' allow_tf32' if s['allow_tf32'] else ''} | "
            f"k={s['k']} | {s['date']}{extra}")
