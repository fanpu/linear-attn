"""Film 'Walk': 900 test images (90 per class, sorted) coloured by what the network at each point of the
path predicts, along the naive line A->B, the Bezier curve and the matched line A->pi(B).

Measured: argmax prediction and max softmax confidence of the interpolated network for every image at
121 path points (compute_hero.py), train loss profile (201 points).  Aesthetic: glasbey_light class colours
(nominal), tint = colour x digit ink x (0.35 + 0.65 conf); frames between the 121 measured points are
cross-faded (declared).  Loops by ping-pong.

usage: python render_walk.py mnist
"""
import argparse, shutil, torchvision, colorcet as cc
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--tag', default='')
ap.add_argument('--sub', type=int, default=3, help='frames per measured step')
ap.add_argument('--still', action='store_true')
args = ap.parse_args()
D = load(f'hero_{args.ds}{args.tag}.npz')
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
dset = {'mnist': torchvision.datasets.MNIST, 'fmnist': torchvision.datasets.FashionMNIST}[args.ds](
    '/home/fzeng/ml/research/art/data', train=False, download=False)
imgs = dset.data.numpy()[D['pred_idx']].astype(np.float32) / 255.  # [900, 28, 28]
lab = D['pred_labels']
COL = np.array([to_rgb(c) for c in cc.glasbey_light[:10]])
T = D['pred_t']
NAMES = [('naive', 'straight line A → B'), ('bezier', 'Bézier curve A → C → B'), ('matched', 'straight line A → π(B)')]
G = 30
TS = 18  # thumbnail px
from PIL import Image
thumbs = np.stack([np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((TS, TS), Image.LANCZOS)) / 255. for im in imgs])


def mosaic(pred, conf):
    col = COL[pred] * (0.35 + 0.65 * conf[:, None])  # [900,3]
    tiles = thumbs[..., None] * col[:, None, None, :] + (1 - thumbs[..., None]) * col[:, None, None, :] * 0.10
    M = tiles.reshape(G, G, TS, TS, 3).transpose(0, 2, 1, 3, 4).reshape(G * TS, G * TS, 3)
    return M


W, H = 1920, 1080
BG = '#08080c'
frames_dir = os.path.join(HERE, 'frames', f'walk_{args.ds}')
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir)
lams = D['lams']
ymax = D['path_naive'][:, 0].max() * 1.1
seq = []
for i in range(len(T) - 1):
    for s in range(args.sub):
        seq.append((i, s / args.sub))
seq.append((len(T) - 1, 0.0))
hold = [seq[0]] * 30
full = hold + seq + [seq[-1]] * 30 + seq[::-1]
if args.still:
    full = [(60, 0.0)]


def draw(k, i, a):
    fig = fig_px(W, H, dpi=100, bg=BG)
    j = min(i + 1, len(T) - 1)
    t = T[i] * (1 - a) + T[j] * a
    for c, (key, title) in enumerate(NAMES):
        x0 = 90 + c * 600
        M0 = mosaic(D['pred_' + key][i], D['conf_' + key][i])
        M1 = mosaic(D['pred_' + key][j], D['conf_' + key][j])
        ax = ax_px(fig, x0, 130, 540, 540, W, H)
        ax.imshow(np.clip(M0 * (1 - a) + M1 * a, 0, 1), interpolation='antialiased')
        acc = (D['pred_' + key][i] == lab).mean() * (1 - a) + (D['pred_' + key][j] == lab).mean() * a
        fig.text((x0 + 270) / W, 1 - 105 / H, title, ha='center', color='#e8e2d4', fontsize=15, style='italic')
        axp = ax_px(fig, x0, 720, 540, 230, W, H)
        r = D['path_' + key][:, 0]
        axp.fill_between(lams, 0, r, color='#e8e2d4', alpha=0.13, lw=0)
        axp.plot(lams, r, color='#e8e2d4', lw=1.2)
        Lt = np.interp(t, lams, r)
        axp.plot([t, t], [0, ymax], color='#e8e2d4', lw=0.6, alpha=0.5)
        axp.plot(t, Lt, 'o', color=BG, mec='#ffffff', mew=1.6, ms=8)
        axp.set_xlim(-0.02, 1.02); axp.set_ylim(-0.03 * ymax, ymax)
        axp.text(0, -0.13 * ymax, 'A', color='#e8e2d4', ha='center', va='top', fontsize=12)
        axp.text(1, -0.13 * ymax, 'π(B)' if key == 'matched' else 'B', color='#e8e2d4', ha='center', va='top', fontsize=12)
        fig.text((x0 + 270) / W, 1 - 1000 / H, f'train loss {Lt:.3f}    shown images correct {100 * acc:.1f}%',
                 ha='center', color='#9b9588', fontsize=11, family='DejaVu Sans Mono')
    fig.text(0.5, 1 - 40 / H, f'ONE BASIN  ·  walking between two {DSNAME} networks   t = {t:.3f}', ha='center', va='center',
             color='#e8e2d4', fontsize=18)
    fig.text(0.5, 1 - 1055 / H, '900 test images (90 per class, rows grouped by true class), each coloured by the class the network at t predicts; '
             'brightness = its confidence.  Colours between the 121 measured t are cross-faded.', ha='center', color='#77726a', fontsize=10)
    arr = fig_to_array(fig)
    Image.fromarray(arr).save(os.path.join(frames_dir, f'{k:05d}.png'))
    return arr


for k, (i, a) in enumerate(full):
    arr = draw(k, i, a)
if args.still:
    save_rgb(arr, f'walk_{args.ds}{args.tag}_midpoint.png')
else:
    print(encode_video(frames_dir, f'walk_{args.ds}{args.tag}', fps=30, gif_width=960, gif_fps=12))
    shutil.rmtree(frames_dir)
