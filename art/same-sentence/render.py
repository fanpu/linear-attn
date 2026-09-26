"""The Same Sentence -- plates.

Each line: Article 1 of the UDHR in one language, shaped with HarfBuzz, set in Noto, and scaled
horizontally (FreeType transform) so that its length is exactly  n_units x u  px, where n_units
is the number of tokens (or, for the null plates, characters / UTF-8 bytes).  Consecutive units
alternate between black and red ink; a glyph cut by a unit boundary is cut in the image too.

usage: python render.py [plates...]   (default: all)
"""
import json, os, sys, collections
import numpy as np
from PIL import Image
import typo

HERE = os.path.dirname(os.path.abspath(__file__))
T = {t["f"]: t for t in json.load(open(os.path.join(HERE, "cache/article1.json")))}
R = json.load(open(os.path.join(HERE, "cache/tokens.json")))
TAB = {r["f"]: r for r in json.load(open(os.path.join(HERE, "cache/table.json")))}
PLATE = [f for f, r in TAB.items() if r["plate"]]
G = os.path.join(HERE, "gallery"); C = os.path.join(HERE, "cache")

# declared palette: cream stock, black ink, one rubric red (two-colour letterpress convention)
CREAM = np.array([244, 238, 225], np.float32)
BLACK = np.array([27, 25, 23], np.float32)
RED = np.array([176, 50, 34], np.float32)
GREY = np.array([150, 142, 130], np.float32)


def units(f, measure):
    t = T[f]
    if measure in R:
        sp = R[measure]["rows"][f]["spans"]
    elif measure == "chars":
        b = np.cumsum([0] + [len(c.encode("utf-8")) for c in t["text"]])
        sp = [(int(b[i]), int(b[i + 1])) for i in range(len(t["text"]))]
    elif measure == "bytes":
        sp = [(i, i + 1) for i in range(t["bytes"])]
    return sp


class Canvas:
    def __init__(s, W, H):
        s.img = np.empty((H, W, 3), np.uint8); s.img[:] = CREAM.astype(np.uint8)
        s.W, s.H = W, H

    def ink(s, alpha, colour, x, y):
        """alpha (h,w) in [0,1]; colour (h,w,3) or (3,)"""
        h, w = alpha.shape
        x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, s.W), min(y + h, s.H)
        if x1 <= x0 or y1 <= y0:
            return
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
        col = colour if np.ndim(colour) == 1 else colour[y0 - y:y1 - y, x0 - x:x1 - x]
        reg = s.img[y0:y1, x0:x1].astype(np.float32)
        # multiply-like: ink darkens, never lightens (overprint)
        out = reg * (1 - a) + np.minimum(reg, col) * a
        s.img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def rule(s, x0, x1, y0, y1, colour, a=1.0):
        s.ink(np.full((y1 - y0, x1 - x0), a, np.float32), colour, x0, y0)

    def text(s, txt, x, y, size, colour=BLACK, feats=None, anchor="l", script="Latn", bcp="en", kx=1.0):
        """y = baseline. anchor l / r / c."""
        g, wn = typo.shape(txt, script, bcp, "ltr", size, feats)
        H = int(size * 2.2); base = int(size * 1.4)
        a, _ = typo.raster(g, [(0, len(txt.encode()))], size, kx, H, base)
        w = wn * kx
        xo = {"l": 0, "r": -w, "c": -w / 2}[anchor]
        s.ink(a, colour, int(round(x + xo)), int(y - base))
        return w

    def save(s, path, maxw=None):
        im = Image.fromarray(s.img)
        if maxw and im.width > maxw:
            im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
        im.save(path, optimize=True)
        return im.size


# declared optical-size correction: Noto Sans Thaana draws its letters at roughly x-height of a
# Latin em, so at the common em it reads far smaller than its neighbours; it is set 1.4x larger.
OPTICAL = {"Thaa": 1.4}


def strip(f, measure, size, u):
    t = T[f]
    sp = units(f, measure)
    size = size * OPTICAL.get(t["script"], 1.0)
    g, wn = typo.shape(t["text"], t["script"], t["bcp47"], t["dir"], size)
    L = len(sp) * u
    kx = L / wn
    s0 = size / OPTICAL.get(t["script"], 1.0)
    H = int(s0 * 3.0); base = int(s0 * 1.9)
    a, tok = typo.raster(g, sp, size, kx, H, base, rtl=t["dir"] == "rtl")
    col = np.where((tok % 2 == 0)[..., None], BLACK, RED).astype(np.float32)
    return a, col, base, L, len(sp), kx


SC = {"smcp": True, "c2sc": True}
NUM = {"onum": True, "pnum": True}


def plate(order, measure, size, u, out, title, subtitle, caption, label_measure=None, ref=None,
          pitch=float(os.environ.get('SS_PITCH', 1.9)), lm=None, preview=6000, eng_ruler=True, ticks=(1, 2, 4, 8, 16)):
    """order: list of text keys, top to bottom."""
    ref = ref or measure
    n_eng = len(units("eng", ref))
    Leng = n_eng * u if ref == measure else None
    lab = size * 0.62
    lm = lm or int(size * 13.5)
    top = int(size * 7.5)
    P = int(size * pitch)
    Ls = [len(units(f, measure)) * u for f in order]
    W = int(lm + max(Ls) + size * 7)
    H = int(top + P * len(order) + size * 6.5)
    cv = Canvas(W, H)
    x0 = lm
    # header
    cv.text(title, x0, int(size * 2.6), size * 1.55, BLACK, SC)
    cv.text(subtitle, x0, int(size * 4.1), size * 0.7, BLACK)
    # scale: English length and multiples (measured reference)
    if eng_ruler:
        Le = len(units("eng", measure)) * u
        yt, yb = top - int(size * 1.2), top + P * len(order) - int(size * 0.3)
        for m in ticks:
            x = int(x0 + m * Le)
            if x > W - size:
                continue
            if m == 1:
                cv.rule(x, x + 1, yt, yb, GREY, 0.9)
            else:
                cv.rule(x, x + 1, yt, yt + int(size * 0.5), GREY, 0.9)
                cv.rule(x, x + 1, yb - int(size * 0.5), yb, GREY, 0.9)
            cv.text(f"×{m}", x + 4, yt - 4, lab * 0.9, GREY, NUM, anchor="l")
        cv.text("the English line", x0 + Le + 4, yt + int(lab * 1.6), lab * 0.8, GREY)
    y = top
    for f, L in zip(order, Ls):
        a, col, base, L, n, kx = strip(f, measure, size, u)
        yy = y + int(size * 0.55)  # baseline position within pitch
        cv.ink(a, col, x0, yy - base)
        nm = T[f]["name"]
        cv.text(nm, x0 - size * 0.8, yy, lab, BLACK, SC, anchor="r")
        lm_ = label_measure or measure
        nn = len(units(f, lm_))
        cv.text(f"{nn}", x0 + L + size * 0.6, yy, lab, BLACK, NUM)
        y += P
    # caption
    cy = H - int(size * 3.2)
    for i, line in enumerate(caption):
        cv.text(line, x0, cy + int(i * lab * 1.55), lab * 0.95, BLACK)
    full = os.path.join(C, out)
    cv.save(full)
    sz = cv.save(os.path.join(G, out), maxw=preview)
    print(out, (W, H), "->", sz)
    return cv


def order_by(measure, keys):
    return sorted(keys, key=lambda f: (len(units(f, measure)), T[f]["bytes"], T[f]["name"]))


def main(which):
    S = 26
    u = S * 0.5 * 170 / 33 / 2.24          # declared: one token = 29.9 px (3.5 mm at 8,600 px/m);
    # chosen so that the median language (2.24x English under Qwen3) sets at roughly its natural width.
    q_order = order_by("Qwen3", PLATE)
    n = len(PLATE)
    src = "Source: UDHR in XML (eric-muller/udhr @ 588b3f4, successor of Unicode’s UDHR in Unicode), stage ≥ 4 texts, one per ISO 639-1 language and script; Article 1, NFC."
    if "qwen3" in which:
        plate(q_order, "Qwen3", S, u, "plate_qwen3.png", "The Same Sentence",
              "All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience "
              "and should act towards one another in a spirit of brotherhood.",
              [f"Article 1 of the Universal Declaration of Human Rights in {n} languages, each line set exactly as long as the number of tokens the Qwen3 tokenizer (151,643-entry byte-level BPE) needs to read it: one token = {u:.1f} px.",
               "Tokens alternate black and red; where a token boundary falls inside a letter, the letter is cut. Glyphs are condensed or extended to fit, never resampled. Number at the line end: token count.",
               src])
    if "gpt2" in which:
        plate(q_order, "GPT-2", S, u, "plate_gpt2.png", "The Same Sentence, 2019",
              "The same lines, in the same order, measured by GPT-2’s tokenizer (50,257-entry byte-level BPE).",
              [f"Order is that of the Qwen3 plate, so rank changes appear as a ragged edge. One token = {u:.1f} px, as on the Qwen3 plate.",
               "Tokens alternate black and red; where a token boundary falls inside a letter, the letter is cut. Number at the line end: token count.", src])
    if "nulls" in which:
        # four measures, same order, scaled so the English line has the same length on every plate
        S2 = 18
        Le = S2 * 0.5 * 170 / 2.24  # same English length as on the main plate, in this size
        for meas, nm, desc in [("chars", "characters", "one unit per Unicode character (a character-level reader)"),
                               ("bytes", "bytes", "one unit per UTF-8 byte (a byte-level reader such as ByT5)"),
                               ("Qwen3", "Qwen3 tokens", "one unit per Qwen3 token"),
                               ("GPT-2", "GPT-2 tokens", "one unit per GPT-2 token")]:
            ne = len(units("eng", meas))
            plate(q_order, meas, S2, Le / ne, f"null_{meas.lower().replace('-', '')}.png",
                  f"The Same Sentence, in {nm}",
                  f"Each line is as long as its {nm}; {desc}. The English line has the same length on all four plates.",
                  [f"Same {n} texts and order as the Qwen3 plate. Units alternate black and red. English = {ne} {nm}.", src],
                  preview=4000)
    if "pairs" in which:
        pairs()


# one language per script (declared: for scripts with several languages in the set, the most widely
# used one, chosen by hand; English stands for Latin)
REP = ["eng", "rus", "arb", "hin", "cmn_hans", "cmn_hant", "jpn", "kor", "ell_monotonic", "heb", "amh", "bod", "tha",
       "mal", "tam", "ben", "guj", "pan", "kan", "tel", "sin", "khm", "lao", "mya", "kat", "hye", "div", "iii", "ike",
       "jav_java", "san_gran", "vie_han"]


def pairs():
    S = 24
    u = S * 0.5 * 170 / 33 / 2.24
    keys = [k for k in REP if k in T]
    missing = [k for k in REP if k not in T]
    if missing:
        print("missing reps", missing)
    keys.sort(key=lambda f: -R["GPT-2"]["rows"][f]["n"] / R["Qwen3"]["rows"][f]["n"])
    lab = S * 0.62; lm = int(S * 12); top = int(S * 8); P = int(S * 2.1); gap = int(S * 1.5)
    Lmax = max(R["GPT-2"]["rows"][f]["n"] for f in keys) * u
    W = int(lm + Lmax + S * 9); H = int(top + len(keys) * (2 * P + gap) + S * 6)
    cv = Canvas(W, H); x0 = lm
    cv.text("What a Larger Vocabulary Buys", x0, int(S * 2.6), S * 1.55, BLACK, SC)
    cv.text("Each language twice: above, as GPT-2 reads it (50,257 tokens in the vocabulary, 2019); below, as Qwen3 reads it (151,643, 2025).",
            x0, int(S * 4.1), S * 0.7)
    cv.text("Ordered by how much the bigger vocabulary shortens the line; the factor is in the margin.", x0, int(S * 5.2), S * 0.7)
    y = top
    for f in keys:
        ng, nq = R["GPT-2"]["rows"][f]["n"], R["Qwen3"]["rows"][f]["n"]
        for j, meas in enumerate(["GPT-2", "Qwen3"]):
            a, col, base, L, n, kx = strip(f, meas, S, u)
            if j == 0:  # declared: the GPT-2 line printed at 45% ink so the Qwen3 line reads as the present
                a = a * 0.45
            yy = y + j * P + int(S * 0.55)
            cv.ink(a, col, x0, yy - base)
            cv.text(f"{n}", x0 + L + S * 0.6, yy, lab, BLACK if j else GREY, NUM)
            if j == 0:
                cv.text(T[f]["name"], x0 - S * 0.8, yy, lab, BLACK, SC, anchor="r")
            else:
                cv.text(f"÷{ng / nq:.2f}", x0 - S * 0.8, yy, lab, RED, NUM, anchor="r")
        y += 2 * P + gap
    cy = H - int(S * 3.0)
    for i, line in enumerate([f"One token = {u:.1f} px on both lines. Tokens alternate black and red; letters cut by a token boundary are cut. One language per script (hand-chosen); English stands for Latin.",
                              "Source: UDHR in XML (eric-muller/udhr @ 588b3f4), Article 1, NFC."]):
        cv.text(line, x0, cy + int(i * lab * 1.55), lab * 0.95)
    cv.save(os.path.join(C, "plate_pairs.png"))
    print("pairs", cv.save(os.path.join(G, "plate_pairs.png"), maxw=6000))


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3", "gpt2", "nulls", "pairs"])
