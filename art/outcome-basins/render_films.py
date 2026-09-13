"""Render films from cached frames: MP4 (H.264, yuv420p, 1080², silent) + GIF (palettegen, <15 MB).

  python render_films.py f3_etafilm newton   |  f3_zoomfilm newton  |  f3_zoomfilm spectral  |  xor_etafilm raw
"""
import glob, json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_lib import *

tag, style = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
src = f'cache/frames/{tag}'
args = json.load(open(f'{src}/args.json'))
out = f'cache/render_{tag}_{style}'
os.makedirs(out, exist_ok=True)
fs = sorted(glob.glob(f'{src}/[0-9]*.npz'))
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 26)
except Exception:
    font = ImageFont.load_default()
TQ = (1.0, 3.3)       # fixed log10-step brightness range for every frame (no flicker); declared
for i, f in enumerate(fs):
    p = f'{out}/{i:04d}.png'
    if os.path.exists(p):
        continue
    d = np.load(f)
    st, tc = d['status'], d['tconv']
    if args['problem'] == 'fact3':
        cols = F3_COLORS if style != 'canon' else F3_CANON
        lab = d['raw'] if style != 'canon' else d['canon']
    else:
        lab = d['raw'] if style in ('raw', 'spectral') else d['canon']
        cols = xor_raw_colors(lab) if style == 'raw' else xor_canon_colors(lab)
    if style == 'spectral':
        rgb = style_spectral(st, tc)
    else:
        rgb, _ = style_newton(lab, st, tc, cols, tq=TQ, vmin=0.3)
    img = Image.fromarray((np.clip(rgb[::-1], 0, 1) * 255 + 0.5).astype(np.uint8))
    if img.size[0] < 1080:     # declared: integer nearest-neighbour upscale, centred on a dark 1080^2 mat (XOR film)
        k = 1080 // img.size[0]
        img = img.resize((img.size[0] * k, img.size[1] * k), Image.NEAREST)
        mat = Image.new('RGB', (1080, 1080), (20, 18, 23)); o = (1080 - img.size[0]) // 2
        mat.paste(img, (o, o)); img = mat; mat_lbl = True
    else:
        mat_lbl = False
    dr = ImageDraw.Draw(img)
    eta = float(d['eta']); w = float(d['win'][1] - d['win'][0])
    label = f'η = {eta:.3f}' if args['mode'] == 'eta' else f'width {w:.2e}   ×{(args["half0"] * 2) / w:.1e}'
    if mat_lbl:
        dr.text((o, 1080 - o + 12), label, fill=(232, 226, 214), font=font)
    else:
        dr.text((28, 1080 - 50), label, fill=(232, 226, 214) if style != 'spectral' else (30, 30, 30), font=font)
    img.save(p)
    if i % 50 == 0:
        print('frame', i, flush=True)
mp4 = f'gallery/{tag}_{style}.mp4'; gif = f'gallery/{tag}_{style}.gif'
hold = ['-vf', f'tpad=stop_mode=clone:stop_duration=2,tpad=start_mode=clone:start_duration=1']
for crf in (18, 22, 26, 30):   # keep MP4 under the 20 MB commit limit
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{out}/%04d.png', *hold,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'slow', mp4], check=True)
    if os.path.getsize(mp4) < 19e6:
        break
for gsz, gfps, ncol in ((540, 15, 192), (440, 12, 128), (360, 10, 96), (300, 8, 64)):   # GIF under ~15 MB
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{out}/%04d.png', '-vf',
                    f'fps={gfps},scale={gsz}:{gsz}:flags=lanczos,split[a][b];[a]palettegen=max_colors={ncol}[p];[b][p]paletteuse=dither=sierra2_4a',
                    '-loop', '0', gif], check=True)
    if os.path.getsize(gif) < 15e6:
        break
print(mp4, os.path.getsize(mp4) / 1e6, 'MB;', gif, os.path.getsize(gif) / 1e6, 'MB', 'gif size', gsz, 'crf', crf)
raise SystemExit
gsz = 540
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{out}/%04d.png', '-vf',
                f'fps={min(fps, 15)},scale={gsz}:{gsz}:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a',
                '-loop', '0', gif], check=True)
print(mp4, os.path.getsize(mp4) / 1e6, 'MB;', gif, os.path.getsize(gif) / 1e6, 'MB')
