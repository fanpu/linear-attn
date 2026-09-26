"""Cross-architecture table of massive activations from cache/acts_*.json/npz."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "olmo2-1b", "deltanet-1.3b", "gla-1.3b", "gdn-1.3b",
         "rwkv7-1.5b", "qwen3-0.6b-randinit"]
KIND = {"qwen3-0.6b": "softmax", "qwen3-1.7b": "softmax", "qwen3-4b": "softmax", "olmo2-1b": "softmax",
        "deltanet-1.3b": "linear (DeltaNet)", "gla-1.3b": "linear (GLA)", "gdn-1.3b": "linear (Gated DeltaNet)",
        "rwkv7-1.5b": "recurrent (RWKV-7)", "qwen3-0.6b-randinit": "softmax, untrained"}


def summarize(name):
    r = json.load(open(os.path.join(CACHE, f"acts_{name}.json")))
    L = r["layers"]
    ratios = np.array([l["top1"] / l["median"] for l in L])
    li = int(ratios.argmax())
    best = L[li]
    out = {"name": name, "kind": KIND.get(name, ""), "nll": r["mean_nll"], "n_tokens": r["n_tokens"],
           "peak_layer": li, "n_layers": len(L), "ratio": float(ratios[li]), "top1": best["top1"],
           "median": best["median"], "ratio_excl_pos0": best["top1_no0"] / best["median_no0"],
           "top_tok": best["tops"][0]["tok"], "top_pos": best["tops"][0]["pos"], "top_dim": best["tops"][0]["dim"],
           "ratios": ratios.tolist()}
    z = np.load(os.path.join(CACHE, f"acts_{name}.npz"))
    if "tokmax" in z:
        tm = z["tokmax"][li]
        med = best["median"]
        mass = tm >= 1000 * med  # Sun et al.-style: ~1000x the median
        out["n_massive_tokens"] = int(mass.sum())
        out["frac_massive_tokens"] = float(mass.mean())
        pos = z["tokpos"][mass]
        out["massive_at_pos0"] = int((pos == 0).sum())
        toks = z["tokstr"][mass]
        u, c = np.unique(toks, return_counts=True)
        o = np.argsort(-c)[:12]
        out["massive_token_types"] = [(str(u[k]), int(c[k])) for k in o]
        out["massive_dims"] = sorted(set(int(x) for x in z["tokarg"][li][mass]))[:20]
    return out


if __name__ == "__main__":
    rows = []
    for n in ORDER:
        if os.path.exists(os.path.join(CACHE, f"acts_{n}.json")):
            s = summarize(n)
            rows.append(s)
            print(f"{n:22s} {s['kind']:24s} nll {s['nll']:.3f}  peak L{s['peak_layer']:2d}/{s['n_layers']}  "
                  f"max/median {s['ratio']:8.0f}  (excl pos0 {s['ratio_excl_pos0']:7.0f})  top {s['top1']:7.1f} at "
                  f"pos {s['top_pos']} {s['top_tok']!r} dim {s['top_dim']}  "
                  f"massive tokens {s.get('n_massive_tokens', '?')} (pos0: {s.get('massive_at_pos0', '?')}) "
                  f"{s.get('massive_token_types', '')[:6] if 'massive_token_types' in s else ''}")
    json.dump(rows, open(os.path.join(CACHE, "summary.json"), "w"), indent=1)
