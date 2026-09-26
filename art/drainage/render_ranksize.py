"""Rank–size of basins: Qwen3-0.6B (full context), its one-token-memory null, uniform random
mappings, Qwen3-1.7B (16,384 starts), and bf16 vs fp32 on shard 0. Ink on cream."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
for f in ["/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf", "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Italic.ttf"]:
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Yrsa", "font.size": 15, "axes.edgecolor": "#1b1a17", "text.color": "#1b1a17",
                     "axes.labelcolor": "#1b1a17", "xtick.color": "#1b1a17", "ytick.color": "#1b1a17"})
PAPER = "#f3eee2"; INK = "#1b1a17"; RUB = "#a83024"; GREY = "#8a8478"
def sizes(d):
    return np.array([b["count"] for b in json.load(open(d + "/basins.json")) if b["key"] != "EOS"])
fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
full = sizes("cache/q06b"); n_full = 151643
ax.loglog(np.arange(1, len(full) + 1), full / n_full, color=INK, lw=2.2, label=f"Qwen3-0.6B, whole vocabulary ({len(full):,} loops)")
b17 = sizes("cache/q17b"); ax.loglog(np.arange(1, len(b17) + 1), b17 / 16384, color=INK, lw=1.2, ls=(0, (5, 2)), label=f"Qwen3-1.7B, 16,384 starts ({len(b17):,})")
fp = sizes("cache/q06b_fp32"); bf = sizes("cache/proto")
ax.loglog(np.arange(1, len(bf) + 1), bf / 18956, color=GREY, lw=1.0, label=f"0.6B shard 0, bf16 ({len(bf):,})")
ax.loglog(np.arange(1, len(fp) + 1), fp / 18956, color=GREY, lw=1.0, ls=":", label=f"0.6B shard 0, fp32 ({len(fp):,})")
m1 = sizes("cache/q06b_markov1"); ax.loglog(np.arange(1, len(m1) + 1), m1 / n_full, color=RUB, lw=2.2, marker="o", ms=5, label=f"null: one token of memory ({len(m1)})")
nl = json.load(open("cache/q06b/nulls.json"))["random_mapping"]
for k, r in enumerate(nl):
    t = np.array(r["top10"]); t = t[t > 0]
    ax.loglog(np.arange(1, len(t) + 1), t / n_full, color=RUB, lw=0.6, alpha=0.5, label="null: uniform random maps (20 draws)" if k == 0 else None)
r = np.arange(1, len(full) + 1)
sel = (r >= 10) & (r <= 3000)
slope, icpt = np.polyfit(np.log(r[sel]), np.log(full[sel] / n_full), 1)
rr = r[sel]
ax.loglog(rr, np.exp(icpt) * rr ** slope, color=RUB, lw=0.9, ls="--",
          label=f"least-squares fit, ranks 10–3000: slope {slope:.2f}")
print("rank-size slope", slope)
ax.set_xlabel("rank of the loop (1 = largest basin)"); ax.set_ylabel("share of starting tokens that drain into it")
ax.set_title("Basin sizes, largest first", loc="left", fontstyle="italic", fontsize=22)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="lower left")
fig.tight_layout(); fig.savefig("gallery/ranksize.png", facecolor=PAPER)
print("ok")
