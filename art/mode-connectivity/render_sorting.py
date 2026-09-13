"""Film 'Sorting': weight matching re-lists B's first-layer units, sweep by sweep.

Measured: B's first-layer receptive fields (784 weights per unit), the layer-1 permutation after every
coordinate-descent sweep of Git Re-Basin weight matching, and the test loss along A -> pi_k(B) after each
sweep (analyze_units.py, 13 points).  Aesthetic: tile flight paths (smoothstep ease between sweeps),
Crameri berlin diverging colours scaled per tile by max |w|, slot order of A (sorted by final matched cosine).

usage: python render_sorting.py mnist
"""
import argparse, shutil, torch
from PIL import Image, ImageDraw, ImageFont
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--tag', default='')
ap.add_argument('--still', action='store_true')
args = ap.parse_args()
U = load(f'units_{args.ds}{args.tag}.npz')
Wt = torch.load(os.path.join(CACHE, f'hero_{args.ds}{args.tag}_weights.pt'), weights_only=False)
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
SP = U['sweep_perm0']  # [S+1, n], slot i holds B unit SP[k, i]
S1, n = SP.shape
order = np.argsort(-U['matchcos0'])  # slot j shows A unit order[j]
cm = plt.get_cmap('cmc.berlin')
TS = 22
side = 28


def tiles(Wm):
    T = Wm.numpy().reshape(-1, side, side)
    T = T / np.abs(T).max(axis=(1, 2), keepdims=True)
    rgb = (cm((T + 1) / 2)[..., :3] * 255).astype(np.uint8)
    return [np.asarray(Image.fromarray(t).resize((TS, TS), Image.LANCZOS)) for t in rgb]


TA, TB = tiles(Wt['A']['W0']), tiles(Wt['B']['W0'])
ncol, nrow = 16, int(np.ceil(n / 16))
pitch = TS + 3
W, H = 1920, 1080
PA = (150, 150)
PB = (150 + ncol * pitch + 80, 150)
slot_of_Aunit = np.empty(n, int); slot_of_Aunit[order] = np.arange(n)


def slot_xy(slot, origin):
    r, c = divmod(slot, ncol)
    return origin[0] + c * pitch, origin[1] + r * pitch


def b_positions(k):
    """xy of every B unit at sweep state k: B unit SP[k,i] sits next to A unit i."""
    pos = np.zeros((n, 2))
    for i in range(n):
        pos[SP[k, i]] = slot_xy(slot_of_Aunit[i], PB)
    return pos


font_path = matplotlib.font_manager.findfont('DejaVu Serif')
mono_path = matplotlib.font_manager.findfont('DejaVu Sans Mono')
F_title, F_lab, F_small, F_mono = [ImageFont.truetype(p, s) for p, s in
                                    [(font_path, 34), (font_path, 26), (font_path, 17), (mono_path, 20)]]
BG, INKc, SUB, ACC = (8, 8, 12), (232, 226, 212), (143, 136, 124), (244, 162, 89)
sl = U['sweep_lams']
SL = U['sweep_loss']
bar = np.array([(x - ((1 - sl) * x[0] + sl * x[-1])).max() for x in SL])


def background(k):
    fig = fig_px(W, H, dpi=100, bg='#08080c')
    ax = ax_px(fig, 1380, 380, 440, 300, W, H)
    ymax = SL.max() * 1.1
    for j in range(k + 1):
        ax.plot(sl, SL[j], color='#e8e2d4', lw=0.8, alpha=0.12 + 0.0 * j)
    ax.plot(sl, SL[k], color='#f4a259', lw=2.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
    ax.plot([0, 1], [0, 0], color='#8f887c', lw=0.6)
    arr = fig_to_array(fig)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.text((W / 2, 55), 'SORTING', font=F_title, fill=INKc, anchor='mm')
    d.text((PA[0] + ncol * pitch / 2, 118), 'A', font=F_lab, fill=INKc, anchor='mm')
    lab = 'B' if k == 0 else f'π{"₁₂₃₄₅₆₇₈₉"[min(k, 9) - 1] if k < 10 else str(k)}(B)'
    d.text((PB[0] + ncol * pitch / 2, 118), 'B → π(B)', font=F_lab, fill=INKc, anchor='mm')
    d.text((1380, 180), f'sweep {k:2d} / {S1 - 1}', font=F_mono, fill=INKc)
    d.text((1380, 215), f'⟨A, π(B)⟩ = {U["sweep_obj"][k]:8.1f}', font=F_mono, fill=INKc)
    d.text((1380, 250), f'test barrier {bar[k]:.3f} nats', font=F_mono, fill=ACC)
    d.text((1380, 330), 'test loss along A → π(B)', font=F_small, fill=SUB)
    d.text((1380, 690), 'A', font=F_small, fill=SUB, anchor='mt'); d.text((1820, 690), 'π(B)', font=F_small, fill=SUB, anchor='mt')
    txt = [f'{n} first-layer units of two {DSNAME} MLPs,', 'each tile = one unit\'s 28×28 weights.',
           'Weight matching solves one assignment', 'problem per layer per sweep; tiles fly to', 'the slot beside the A unit they are',
           'paired with (all 3 layers are matched;', 'layer 1 is shown). Slots ordered by final', 'matched similarity. Flight paths are',
           'decorative; positions at each sweep are', 'exact. Colour: berlin, per-tile |w| max.']
    for i, t in enumerate(txt):
        d.text((1380, 760 + 26 * i), t, font=F_small, fill=SUB)
    return np.asarray(im).copy()


def smooth(a):
    return a * a * (3 - 2 * a)


fd = os.path.join(HERE, 'frames', f'sort_{args.ds}')
shutil.rmtree(fd, ignore_errors=True); os.makedirs(fd)
plan = [(0, 0, 0.0)] * 60
for k in range(S1 - 1):
    nfly = 150 if k == 0 else 40
    plan += [(k, k + 1, smooth(f / (nfly - 1))) for f in range(nfly)] + [(k + 1, k + 1, 0.0)] * (30 if k == 0 else 12)
plan += [(S1 - 1, S1 - 1, 0.0)] * 150
if args.still:
    plan = [(S1 - 1, S1 - 1, 0.0)]
bgs = {}
posc = {}
for fno, (k0, k1, a) in enumerate(plan):
    kb = k1 if a > 0.5 or k0 == k1 else k0
    if kb not in bgs:
        bgs[kb] = background(kb)
    for kk in (k0, k1):
        if kk not in posc:
            posc[kk] = b_positions(kk)
    canvas = bgs[kb].copy()
    for j in range(n):
        x, y = slot_xy(slot_of_Aunit[j], PA)
        canvas[y:y + TS, x:x + TS] = TA[j]
    pos = posc[k0] * (1 - a) + posc[k1] * a
    moving = np.abs(posc[k0] - posc[k1]).sum(1) > 0
    for u in list(np.nonzero(~moving)[0]) + list(np.nonzero(moving)[0]):
        x, y = int(round(pos[u, 0])), int(round(pos[u, 1]))
        canvas[y:y + TS, x:x + TS] = TB[u]
    Image.fromarray(canvas).save(os.path.join(fd, f'{fno:05d}.png'))
if args.still:
    save_rgb(canvas, f'sorting_{args.ds}{args.tag}_final.png')
else:
    print(encode_video(fd, f'sorting_{args.ds}{args.tag}', fps=30, gif_width=960, gif_fps=15))
shutil.rmtree(fd)
