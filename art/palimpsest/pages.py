"""Render the pages. Ink density in [0,1] (1 = ink, 0 = bare vellum).

A  under-text: Lucretius, De rerum natura I.1-, chancery hand (Z003), lines horizontal.
B  over-text : Vulgate Genesis 1, bold book hand (C059 Bold), lines vertical (rotated 90 deg,
               as the Archimedes Palimpsest's prayer book runs across Archimedes), larger pitch.
C  unrelated : Iliad I.1- in Greek (P052), lines at -33 deg, a third pitch. Null control (b).
D1..D4       : decoys. Same hand, size, angle and line positions as A, words of A shuffled.
               They share A's layout and letter statistics but not its letters. Null control (c).

All pages are drawn once at 1024 px and box-downsampled, so every resolution sees the same page.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = "/usr/share/fonts/opentype/urw-base35/"
BASE = 1024

LUCRETIUS = (
    "Aeneadum genetrix, hominum divomque voluptas, alma Venus, caeli subter labentia signa "
    "quae mare navigerum, quae terras frugiferentis concelebras, per te quoniam genus omne "
    "animantum concipitur visitque exortum lumina solis: te, dea, te fugiunt venti, te nubila "
    "caeli adventumque tuum, tibi suavis daedala tellus summittit flores, tibi rident aequora "
    "ponti placatumque nitet diffuso lumine caelum. nam simul ac species patefactast verna diei "
    "et reserata viget genitabilis aura favoni, aeriae primum volucris te, diva, tuumque "
    "significant initum perculsae corda tua vi. inde ferae pecudes persultant pabula laeta "
    "et rapidos tranant amnis: ita capta lepore te sequitur cupide quo quamque inducere pergis. "
    "denique per maria ac montis fluviosque rapacis frondiferasque domos avium camposque "
    "virentis omnibus incutiens blandum per pectora amorem efficis ut cupide generatim saecla "
    "propagent. quae quoniam rerum naturam sola gubernas nec sine te quicquam dias in luminis "
    "oras exoritur neque fit laetum neque amabile quicquam, te sociam studeo scribendis "
    "versibus esse, quos ego de rerum natura pangere conor Memmiadae nostro."
)
GENESIS = (
    "In principio creavit Deus caelum et terram. Terra autem erat inanis et vacua, et tenebrae "
    "erant super faciem abyssi: et spiritus Dei ferebatur super aquas. Dixitque Deus: Fiat lux. "
    "Et facta est lux. Et vidit Deus lucem quod esset bona: et divisit lucem a tenebris. "
    "Appellavitque lucem Diem, et tenebras Noctem: factumque est vespere et mane, dies unus. "
    "Dixit quoque Deus: Fiat firmamentum in medio aquarum: et dividat aquas ab aquis. Et fecit "
    "Deus firmamentum, divisitque aquas, quae erant sub firmamento, ab his, quae erant super "
    "firmamentum. Et factum est ita. Vocavitque Deus firmamentum, Caelum: et factum est vespere "
    "et mane, dies secundus."
)
ILIAD = (
    "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην, ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε, πολλὰς δ᾽ "
    "ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν οἰωνοῖσί τε πᾶσι, "
    "Διὸς δ᾽ ἐτελείετο βουλή, ἐξ οὗ δὴ τὰ πρῶτα διαστήτην ἐρίσαντε Ἀτρεΐδης τε ἄναξ ἀνδρῶν "
    "καὶ δῖος Ἀχιλλεύς. τίς τ᾽ ἄρ σφωε θεῶν ἔριδι ξυνέηκε μάχεσθαι; Λητοῦς καὶ Διὸς υἱός· "
    "ὃ γὰρ βασιλῆϊ χολωθεὶς νοῦσον ἀνὰ στρατὸν ὄρσε κακήν, ὀλέκοντο δὲ λαοί."
)

# (text, font file, font px at 1024, line pitch px at 1024, angle deg, x-offset px)
SPECS = {
    "A": (LUCRETIUS, "Z003-MediumItalic.otf", 46, 50, 0.0, 0),
    "B": (GENESIS, "C059-Bold.otf", 50, 69, 90.0, 0),
    "C": (ILIAD, "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 36, 59, -33.0, 0),
}


def _shuffle_words(text, seed):
    w = text.split()
    random.Random(seed).shuffle(w)
    return " ".join(w)


def render(text, font_file, size, pitch, angle, xoff=0):
    """Fill a large canvas with justified-left lines of text, rotate, centre-crop to BASE."""
    big = int(BASE * 1.6)
    img = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype((font_file if font_file.startswith("/") else FONTS + font_file), size)
    words = text.split()
    wi = 0
    y = 0
    margin = 12 + xoff
    while y < big:
        x = margin
        line = []
        while True:
            w = words[wi % len(words)]
            tw = d.textlength((" ".join(line + [w])), font=font)
            if margin + tw > big - 12 and line:
                break
            line.append(w)
            wi += 1
        d.text((margin, y), " ".join(line), fill=255, font=font)
        y += pitch
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    o = (big - BASE) // 2
    img = img.crop((o, o, o + BASE, o + BASE))
    return np.asarray(img, dtype=np.float32) / 255.0


def all_pages():
    pages = {}
    for k, (t, f, s, p, a, x) in SPECS.items():
        # start the visible text some way into the passage so the crop is full of words
        pages[k] = render(t, f, s, p, a, x)
    t, f, s, p, a, x = SPECS["A"]
    for j in range(1, 5):
        pages[f"D{j}"] = render(_shuffle_words(t, 100 + j), f, s, p, a, x)
    return pages


def downsample(x, n):
    return np.asarray(Image.fromarray(x).resize((n, n), Image.BOX), dtype=np.float32)


def load(n):
    path = os.path.join(HERE, "cache", f"pages_{n}.npz")
    if not os.path.exists(path):
        base = os.path.join(HERE, "cache", f"pages_{BASE}.npz")
        if os.path.exists(base):
            P = dict(np.load(base))
        else:
            P = all_pages()
            os.makedirs(os.path.dirname(base), exist_ok=True)
            np.savez_compressed(base, **P)
        if n != BASE:
            P = {k: downsample(v, n) for k, v in P.items()}
        np.savez_compressed(path, **P)
    return dict(np.load(path))


if __name__ == "__main__":
    import sys
    for n in (256, 512, BASE):
        P = load(n)
    P = load(256)
    for k, v in P.items():
        print(k, v.shape, round(float(v.mean()), 3))
    # pairwise correlations at 256 (null (c): pages should be near-uncorrelated)
    ks = list(P)
    Z = {k: (P[k] - P[k].mean()) / P[k].std() for k in ks}
    for a in ks:
        print(a, " ".join(f"{float((Z[a]*Z[b]).mean()):+.3f}" for b in ks))
    row = np.concatenate([P[k] for k in ks], 1)
    Image.fromarray((255 * (1 - row)).astype(np.uint8)).save(os.path.join(HERE, "cache", "pages_preview.png"))
