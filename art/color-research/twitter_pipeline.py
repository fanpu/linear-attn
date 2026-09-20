"""Does a palette survive a social-media image pipeline?

The catalogue (COLOR_SCHEMES.md) measures each scheme in CAM02-UCS: uniformity, lightness
reversals, greyscale retention, CVD retention. Those say whether a scheme is readable in
principle. They do not say whether it is still readable after X/Twitter has had it.

X's pipeline does three things to an upload, and each one attacks a different property:

  1. resize        4096^2 render -> 2048 (`name=large`) -> ~1200 (`name=medium`) -> ~440 in-timeline.
                   Attacks *spatial* structure: fractal detail below the new Nyquist is gone,
                   and regular screens (halftone) alias into moire.
  2. JPEG 4:2:0    chroma planes are downsampled 2x before quantisation. Attacks *hue* contrast
                   specifically. Luminance contrast is kept at full resolution.
  3. framing       the image sits on #000000 (dark) or #ffffff (light) chrome.

Step 2 is the interesting one for this project, because the Sohl-Dickstein split is *designed*
to put hue contrast where lightness contrast is low: the two ramps meet dark-to-dark at the seam
and separate by hue. That is the exact signal 4:2:0 throws away first.

Hypothesis (H1): schemes that carry structure in chroma lose more to the codec than schemes
that carry it in lightness, and the loss is predicted by the chroma-carried-contrast fraction.

Everything below is measured. Aesthetic choices (which data, which crop) are declared in main().

    python twitter_pipeline.py            # full run, writes gallery/ + PIPELINE.md
    python twitter_pipeline.py --quick    # 1024^2, fewer schemes
"""
import io
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P
import sources as S

GAL = "/home/fzeng/ml/research/art/color-research/gallery/"

# X/Twitter serving sizes (long edge, px). `large` is what a click gives you; `medium` is the
# inline card on desktop; `thumb` is roughly the mobile timeline.
STAGES = {"large": 2048, "medium": 1200, "thumb": 440}
JPEG_Q = 75          # X re-encodes uploads; 75 is the commonly reported working point
SUBSAMPLING = 2      # PIL code for 4:2:0


# ---------------------------------------------------------------- pipeline
def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def lanczos(img_u8, size):
    im = Image.fromarray(img_u8)
    if max(im.size) == size:
        return img_u8
    w, h = im.size
    s = size / max(w, h)
    return np.asarray(im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS))


def jpeg(img_u8, quality=JPEG_Q, subsampling=SUBSAMPLING):
    """Round-trip through JPEG with the given chroma subsampling."""
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def pipeline(img_u8, stage):
    """Resize + re-encode, the way an upload actually arrives at a reader's screen."""
    x = lanczos(img_u8, STAGES["large"])
    x = jpeg(x)                                  # X re-encodes at upload
    if stage != "large":
        x = lanczos(x, STAGES[stage])
        x = jpeg(x)                              # and again when it serves the smaller variant
    return x


def ideal(img_u8, stage):
    """The same resize with no codec: isolates what the *codec* cost, not what the resize cost."""
    return lanczos(lanczos(img_u8, STAGES["large"]), STAGES[stage])


# ---------------------------------------------------------------- metrics
def lab(img_u8):
    return P.rgb_to_lab(img_u8.astype(np.float64) / 255.0)


def chroma_carried_fraction(img_u8):
    """Share of local contrast that lives in chroma rather than lightness.

    Sobel gradient magnitude per channel in CIELAB, summed over the image. 4:2:0 keeps L at full
    resolution and halves a*,b*, so this fraction predicts codec fragility. Units: dimensionless
    in [0,1]; L and (a,b) are commensurate because CIELAB is (roughly) perceptually scaled.
    """
    L = lab(img_u8)
    g = [np.hypot(ndimage.sobel(L[..., c], 0), ndimage.sobel(L[..., c], 1)) for c in range(3)]
    eL = g[0].sum()
    eC = np.hypot(g[1], g[2]).sum()
    return float(eC / (eL + eC + 1e-12))


def structure_energy(img_u8):
    """Total L* gradient magnitude per pixel: how much visible detail is present at this size."""
    L = lab(img_u8)[..., 0]
    return float(np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1)).mean())


def dE(a_u8, b_u8):
    d = P.deltaE2000(lab(a_u8), lab(b_u8))
    return float(np.mean(d)), float(np.percentile(d, 95))


def border_contrast(img_u8, chrome_hex):
    """dE00 between the image's outer 8-px frame and the page chrome behind it.

    Low = the image has no edge against the timeline and reads as a hole (dark art on dark mode)
    or bleeds into the page (cream art on light mode).
    """
    b = 8
    edge = np.concatenate([img_u8[:b].reshape(-1, 3), img_u8[-b:].reshape(-1, 3),
                           img_u8[:, :b].reshape(-1, 3), img_u8[:, -b:].reshape(-1, 3)])
    e_lab = P.rgb_to_lab(edge.astype(np.float64) / 255.0)
    c_lab = P.rgb_to_lab(np.array(P.hex2rgb(chrome_hex), dtype=np.float64).reshape(1, 3))
    return float(np.mean(P.deltaE2000(e_lab, np.repeat(c_lab, len(e_lab), 0))))


def measure(img_float):
    """Full report for one rendered image."""
    src = to_u8(img_float)
    out = {"chroma_fraction": chroma_carried_fraction(lanczos(src, STAGES["medium"]))}
    for stage in STAGES:
        got, want = pipeline(src, stage), ideal(src, stage)
        m, p95 = dE(got, want)
        out[f"dE_{stage}"] = m
        out[f"dE95_{stage}"] = p95
        if stage == "thumb":
            out["detail_retained"] = structure_energy(want) / (structure_energy(lanczos(src, STAGES["large"])) + 1e-12)
            out["dark_edge"] = border_contrast(got, "#000000")
            out["light_edge"] = border_contrast(got, "#ffffff")
            out["mean_L"] = float(lab(got)[..., 0].mean())
    return out


# ---------------------------------------------------------------- renderers under test
SPLITS = ["sd_spectral", "aurora_ember", "indigo_madder", "hubble_sho", "klimt_lapis",
          "verdigris_copper", "cyanotype_vandyke", "synthwave", "cinestill", "crt_phosphor",
          "malachite_rhodochrosite", "morandi", "riso_pink_blue", "hokusai_sunset"]
SEQUENTIALS = ["klimt_gold", "bioluminescence", "ember", "crt_amber", "cyanotype", "platinum"]


def render_all(x, dens, quick=False):
    """name -> (rgb float image, family). All on the same real data."""
    out = {}
    for k in (SPLITS[:5] if quick else SPLITS):
        out[k] = (P.render_split(x, k, near_boundary="large"), "split")
    for k in (SEQUENTIALS[:2] if quick else SEQUENTIALS):
        out[k] = (P.as_cmap(k)(P.rank_normalize(dens))[..., :3], "sequential")
    # The two riso treatments, as a controlled pair: same inks, same data, different technique.
    import render_sheets as R
    n = x.shape[0]
    out["riso_halftone"] = (R.riso_render(x, "large", [P.RISO["fluo_pink"], P.RISO["blue"]],
                                          P.RISO_PAPER, size=n), "riso")
    return out


def diff_map(a_u8, b_u8, full_scale=10.0):
    """Per-pixel CIEDE2000 on a black-to-ember ramp. Full scale is a declared dE00 of 10."""
    d = np.clip(P.deltaE2000(lab(a_u8), lab(b_u8)) / full_scale, 0, 1)
    return P.as_cmap("ember")(d)[..., :3]


def main(quick=False):
    n = 1024 if quick else 2048
    d = S.basin_signed(tag="z1T_4096", stride=2)
    x = d["x"]
    c = x.shape[0] // 2
    h = int(x.shape[0] * 0.31)
    x = x[c - h:c + h, c - h:c + h]                       # declared crop: the fractal edge region
    x = np.asarray(Image.fromarray(x.astype(np.float32), "F").resize((n, n), Image.BILINEAR))
    dens = np.abs(x)

    imgs = render_all(x, dens, quick)
    rows = []
    for name, (img, family) in imgs.items():
        m = measure(img)
        m.update(name=name, family=family)
        rows.append(m)
        print(f"{name:26s} chroma {m['chroma_fraction']:.3f}  dE_med {m['dE_medium']:5.2f}  "
              f"dE_thumb {m['dE_thumb']:5.2f}  detail {m['detail_retained']:.3f}")

    with open(GAL + "../twitter_pipeline.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows, imgs


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
