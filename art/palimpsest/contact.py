"""gallery/contact_sheet.png: the best plates, scaled to a common height, on vellum."""
import os
import sys

from PIL import Image

import look

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "gallery")


def main(names, out=os.path.join(G, "contact_sheet.png"), cols=3, cell=900, pad=40):
    ims = [Image.open(os.path.join(G, n)).convert("RGB") for n in names]
    rows = (len(ims) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad + 60
    sheet = look.sheet(W, H)
    look.text(sheet, (pad, 18), "Palimpsest — contact sheet", 30)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        im.thumbnail((cell, cell), Image.LANCZOS)
        x = pad + c * (cell + pad) + (cell - im.width) // 2
        y = 60 + pad + r * (cell + pad) + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1:])
