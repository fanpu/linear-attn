"""Shared look: stock, inks, type sizes, token transcription.  All values are declared choices."""
import unicodedata
from PIL import Image, ImageDraw
import typeset as ts

CREAM = (243, 237, 224)
INK = (27, 26, 23)
MADDER = (163, 45, 38)          # second ink: what the model said
GREY = (132, 124, 110)          # apparatus: rules, ids, markers
PALE = (196, 188, 172)          # sayable tokens in census plates

W, H = 3440, 4864                # 40 cm wide at 8,600 px/m, 1:1.414


def sheet(w=W, h=H):
    im = Image.new("RGB", (w, h), CREAM)
    return im, ImageDraw.Draw(im)


def visible_parts(s):
    """Split a string into (text, kind) parts, kind in {"t" text, "m" marker, "s" special token}.
    Declared transcription: space -> '␣' (only leading/trailing/doubled), newline -> '↵', tab -> '⇥',
    other controls -> ⟨U+..⟩; ⟨0xNN⟩ (undecodable byte) passes through as a marker; a special/control
    token (wrapped in U+E000..U+E001 by analyze.py) is drawn boxed."""
    out = []

    def add(t, m):
        if out and out[-1][1] == m and m != "s":
            out[-1] = (out[-1][0] + t, m)
        else:
            out.append((t, m))
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\ue000":
            j = s.find("\ue001", i)
            j = n if j < 0 else j
            add(s[i + 1:j], "s"); i = j + 1; continue
        if ch == "\u27e8" and s[i + 1:i + 3] == "0x" and s[i + 5:i + 6] == "\u27e9":
            add(s[i:i + 6], "m"); i += 6; continue
        if ch == " ":
            lead = s[:i].strip() == ""
            trail = s[i:].strip() == ""
            dbl = (i + 1 < n and s[i + 1] == " ") or (i > 0 and s[i - 1] == " ")
            mk = lead or trail or dbl
            add("␣" if mk else " ", "m" if mk else "t")
        elif ch == "\n":
            add("↵", "m")
        elif ch == "\t":
            add("⇥", "m")
        elif ch == "\r":
            add("␍", "m")
        elif unicodedata.category(ch) in ("Cc", "Cf") and ch != "\u200d":
            add(f"⟨U+{ord(ch):04X}⟩", "m")
        else:
            add(ch, "t")
        i += 1
    return out


SCALE = {"t": 1.0, "m": 0.72, "s": 0.62}


def _part_w(t, k, size, chain):
    if k == "s":
        return ts.width(t, size * SCALE[k], chain) + size * 0.34
    return ts.width(t, size * SCALE[k], chain)


def _draw_part(d, x, y, t, k, size, chain, fill, marker_fill):
    if k == "t":
        return ts.draw(d, (x, y), t, size, chain, fill)
    if k == "m":
        return ts.draw(d, (x, y), t, size * 0.72, chain, marker_fill)
    # special token: small text inside a hairline box, in the reply's own ink
    ss = size * SCALE["s"]
    w = ts.width(t, ss, chain)
    pad = size * 0.1
    lw = max(1, int(size / 36))
    d.rectangle([x + pad * 0.6, y - ss * 0.95, x + pad * 0.6 + w + 2 * pad, y + ss * 0.22], outline=fill, width=lw)
    ts.draw(d, (x + pad * 0.6 + pad, y), t, ss, chain, fill)
    return x + w + size * 0.34


def draw_visible(d, xy, s, size, fill, marker_fill=GREY, chain=None, max_w=None):
    chain = chain or ts.SERIF_CHAIN
    x, y = xy
    for t, k in visible_parts(s):
        w = _part_w(t, k, size, chain)
        if max_w is not None and x + w > xy[0] + max_w:
            if k == "t":
                acc = ""
                for ch in t:
                    if x + ts.width(acc + ch + "…", size, chain) > xy[0] + max_w:
                        break
                    acc += ch
                x = ts.draw(d, (x, y), acc, size, chain, fill)
            x = ts.draw(d, (x, y), "…", size, chain, marker_fill)
            return x, True
        x = _draw_part(d, x, y, t, k, size, chain, fill, marker_fill)
    return x, False


def vis_width(s, size, chain=None):
    chain = chain or ts.SERIF_CHAIN
    return sum(_part_w(t, k, size, chain) for t, k in visible_parts(s))


def uname(s):
    t = s.strip() or s
    if len(t) == 1:
        return unicodedata.name(t, f"U+{ord(t):04X}")
    return None


def smallcaps(d, xy, text, size, fill, anchor="ls", track=0.08):
    """Letter-spaced capitals in the text face (declared stand-in for true small caps)."""
    f = ts.font(ts.F["serif"], size)
    x, y = xy
    tot = sum(f.getlength(c) + size * track for c in text) - size * track
    if anchor[0] == "r":
        x -= tot
    elif anchor[0] == "m":
        x -= tot / 2
    for c in text:
        d.text((x, y), c, font=f, fill=fill, anchor="l" + anchor[1])
        x += f.getlength(c) + size * track
    return x
