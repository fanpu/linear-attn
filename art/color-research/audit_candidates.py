"""Run the twitter_pipeline metrics over real gallery images, not synthetic renders.

PIPELINE.md measures palettes on one controlled field. This asks the operational question instead:
for each image actually being considered for posting, how much does the platform cost it?
"""
import sys, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import twitter_pipeline as T

ROOT = "/home/fzeng/ml/research/"
CANDIDATES = [
    "art/trainability-fractal/gallery/hero_deep_swirl_spectral_print.png",
    "art/trainability-fractal/gallery/hero_deep_swirl_riso_print.png",
    "art/gd-bifurcation/gallery/basin_wide_riso.png",
    "art/gd-bifurcation/gallery/basin_wide_spectral.png",
    "art/gd-bifurcation/gallery/hero_riso_lyapunov.png",
    "art/signal-propagation/gallery/frontier_N100_poster_spectral.png",
    "art/signal-propagation/gallery/frontier_N100_poster_riso.png",
    "art/game-chaos/gallery/lyap_z3_shrimp_spectral.png",
    "art/game-chaos/gallery/lyap_z3_shrimp_verdigris.png",
    "hardware/lattice/gallery/lattice_spectral_split_bf16_s2048_k4096.png",
    "hardware/dither/gallery/hero_sd_idle_zoom_riso.png",
    "hardware/precision-divergence/gallery/hero_float16_c1_rivers.png",
    "art/decode-map/gallery/hero/story_tp256_glass.png",
    "art/combed/gallery/hair/hair_diptych_N256.png",
    "art/grokking/gallery/specimen_plotter.png",
    "hardware/float-ruler/gallery/nautilus_float16_observatory.png",
    "art/weight-spectrum/gallery/riso_mp_vs_esd_mlp_bs16_s1_fluo_pink_blue.png",
    "hardware/fingerprint/gallery/divergence_texts_feynman_paper.png",
    "hardware/roofline/gallery/roofline_night.png",
]

rows = []
for rel in CANDIDATES:
    src = np.asarray(Image.open(ROOT + rel).convert("RGB"))
    src = T.lanczos(src, 2048)
    m = T.measure(src.astype(np.float64) / 255.0)
    m["name"] = rel.split("/")[-1]
    m["project"] = rel.split("/")[0] + "/" + rel.split("/")[1]
    rows.append(m)
    print(f"{m['name'][:46]:46s} chroma {m['chroma_fraction']:.3f}  dE {m['dE_medium']:5.2f}  "
          f"detail {m['detail_retained']:.3f}  L* {m['mean_L']:4.1f}  dark {m['dark_edge']:4.1f}  light {m['light_edge']:4.1f}")
json.dump(rows, open("candidate_audit.json", "w"), indent=1)
