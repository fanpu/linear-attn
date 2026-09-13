"""Hover-able HTML map (gallery/<name>.html): move over a cell to read the text that every
(T, p) in it produces. Also writes a markdown table of texts along one horizontal transect."""
import base64
import html
import json
import os
import sys

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

import analysis as A
import render as Rr

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)


def decode_row(r):
    return tok.decode([int(x) for x in r if x >= 0])


def build(path, name, s=4, title=""):
    d = A.load(path)
    tk = d["tokens"]
    H, W, L = tk.shape
    hs = A.prefix_hashes(tk)
    lab = A.labels_from_hash(hs[-1])
    nl = lab.max() + 1
    rep = np.zeros(nl, int)
    rep[lab.ravel()] = np.arange(H * W)
    flat = tk.reshape(-1, L)
    texts = [decode_row(flat[i]) for i in rep]
    fd = A.first_divergence(tk).ravel()[rep]
    area = np.bincount(lab.ravel(), minlength=nl)
    img = Rr.style_mosaic(d, 1)                                  # one px per sample, CSS scales it
    Image.fromarray((img * 255).astype(np.uint8)).save("/tmp/_hover.png")
    b64 = base64.b64encode(open("/tmp/_hover.png", "rb").read()).decode()
    lab_b64 = base64.b64encode(lab[::-1].astype(np.int32).tobytes()).decode()   # top row first
    xs, ys = d["xs"], d["ys"]
    ylab = "top-p" if d["grid"] == "tp" else "repetition penalty"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{background:#0e0e10;color:#ddd;font:14px/1.45 Georgia,serif;margin:24px}}
#wrap{{display:flex;gap:24px;align-items:flex-start}} canvas{{image-rendering:pixelated;width:{W*s}px;height:{H*s}px;cursor:crosshair}}
#txt{{max-width:520px;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;background:#18181b;padding:12px;border-radius:4px;min-height:200px}}
.meta{{color:#9a9;margin-bottom:8px;font-family:Georgia,serif}}</style></head><body>
<h2 style="font-weight:normal">{html.escape(title)}</h2>
<p class="meta">Prompt: “{html.escape(d['prompt'])}” · {L} tokens · x: temperature {xs[0]-(xs[1]-xs[0])/2:g}–{xs[-1]+(xs[1]-xs[0])/2:g} · y: {ylab} {ys[0]-(ys[1]-ys[0])/2:g}–{ys[-1]+(ys[1]-ys[0])/2:g} · {nl} distinct outputs.
Brightness = how late the output departs from greedy decoding.</p>
<div id="wrap"><canvas id="c" width="{W}" height="{H}"></canvas><div><div class="meta" id="m">hover a cell</div><div id="txt"></div></div></div>
<script>
const W={W},H={H};const texts={json.dumps(texts)};const fd={json.dumps(fd.tolist())};const area={json.dumps(area.tolist())};
const xs={json.dumps([float(x) for x in xs])},ys={json.dumps([float(y) for y in ys])};
const lab=new Int32Array(Uint8Array.from(atob("{lab_b64}"),c=>c.charCodeAt(0)).buffer);
const cv=document.getElementById('c'),cx=cv.getContext('2d');const im=new Image();let base=null;
im.onload=()=>{{cx.drawImage(im,0,0);base=cx.getImageData(0,0,W,H);}};im.src="data:image/png;base64,{b64}";
let last=-1;cv.onmousemove=e=>{{const r=cv.getBoundingClientRect();const j=Math.floor((e.clientX-r.left)/r.width*W),i=Math.floor((e.clientY-r.top)/r.height*H);
if(i<0||j<0||i>=H||j>=W||!base)return;const c=lab[i*W+j];
document.getElementById('m').textContent=`T=${{xs[j].toFixed(4)}}  {ylab}=${{ys[H-1-i].toFixed(4)}}  ·  cell of ${{area[c]}} samples  ·  departs from greedy at token ${{fd[c]}}`;
if(c===last)return;last=c;document.getElementById('txt').textContent=texts[c];
const o=new ImageData(new Uint8ClampedArray(base.data),W,H);for(let k=0;k<W*H;k++){{if(lab[k]===c){{o.data[4*k]=255;o.data[4*k+1]=255;o.data[4*k+2]=255;}}}}cx.putImageData(o,0,0);}};
</script></body></html>"""
    out = os.path.join(Rr.GAL, name + ".html")
    open(out, "w").write(page)
    print(out, os.path.getsize(out) / 1e6, "MB", nl, "cells")
    return d, lab, texts


def transect_table(path, p_value, max_rows=8, T_max=None):
    """Distinct outputs met walking along temperature at fixed y = p_value."""
    d = A.load(path)
    tk = d["tokens"]
    i = int(np.argmin(np.abs(d["ys"] - p_value)))
    row = tk[i]
    xs = d["xs"]
    segs = []
    j = 0
    while j < len(xs):
        k = j
        while k + 1 < len(xs) and np.array_equal(row[k + 1], row[j]):
            k += 1
        segs.append((j, k))
        j = k + 1
    esc = lambda x: html.escape(x).replace("\n", " ⏎ ")
    lines = ['<table><tr><th>T range</th><th>tokens shared with the cell to the left</th>'
             '<th>output (first 48 tokens; the part that differs from the cell to the left in bold)</th></tr>']
    prev = None
    dx = xs[1] - xs[0]
    for (a, b) in segs[:max_rows]:
        if T_max is not None and xs[a] > T_max:
            break
        r = row[a][:48]
        if prev is None:
            shared = 0
        else:
            dif = np.flatnonzero(prev != row[a])
            shared = int(dif[0]) if len(dif) else len(r)
        shared = min(shared, len(r))
        head, tail = esc(decode_row(r[:shared])), esc(decode_row(r[shared:]))
        lines.append(f"<tr><td>{xs[a]-dx/2:.3f}–{xs[b]+dx/2:.3f}</td><td>{shared if prev is not None else '–'}</td>"
                     f"<td><span style='color:#888'>{head}</span><b>{tail}</b></td></tr>")
        prev = row[a]
    lines.append("</table>")
    return "\n".join(lines)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4,
          sys.argv[4] if len(sys.argv) > 4 else "Decode map")
