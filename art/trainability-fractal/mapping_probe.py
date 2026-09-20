"""Probe tonal mappings for the deep_swirl hero on the cached 1024^2 field.
Scores each on (a) key commitment |mean-0.5| and (b) whether the measured
converge/diverge boundary is actually the most visible thing in the frame."""
import numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use('Agg')
import matplotlib as mpl
import styles as S
from common_render import cdf_img, edges, speed01
from render_windows import load

w = load('deep_zoomA4_1024_f64'); M = w['M']
E = edges(M)                                   # true boundary mask (2x2 sign change)
Ep = np.pad(E, ((0,1),(0,1)), constant_values=False)
print(f"field {M.shape}, boundary pixels {E.sum()} ({E.mean()*100:.2f}%)")

def lum(rgb):  # perceptual-ish luminance 0..1
    a = rgb.astype(np.float32)/255
    return 0.2126*a[...,0]+0.7152*a[...,1]+0.0722*a[...,2]

# ---- candidate mappings -------------------------------------------------
def cand_spectral(M):                      # CURRENT hero mapping
    return S.spectral(M)

def cand_fire_ice(M):                      # exists in styles.py, never used for the hero
    return S.dark_fire_ice(M)

def cand_magma(M):
    return S.dark_magma(M)

def _phase_gamma(M, gamma, cmap_conv, cmap_div, floor=0.0):
    """Keep the phase split, but map WITHIN each phase by a gamma on the
    speed rank instead of histogram-equalising. Slow (near boundary) -> dark."""
    s, conv = speed01(M)                   # s: 1=fast, 0=slow(near boundary)
    s = np.clip(s, 0, 1) ** gamma
    s = floor + (1-floor)*s
    out = np.zeros(M.shape+(3,), np.float32)
    cc = mpl.colormaps[cmap_conv]; cd = mpl.colormaps[cmap_div]
    out[conv]  = cc(s[conv])[...,:3]
    out[~conv] = cd(s[~conv])[...,:3]
    return (np.clip(out,0,1)*255+0.5).astype(np.uint8)

def cand_gamma_hot_ice(M):   return _phase_gamma(M, 0.55, 'inferno', 'bone')
def cand_gamma_ember(M):     return _phase_gamma(M, 0.70, 'afmhot', 'cividis')
def cand_gamma_hard(M):      return _phase_gamma(M, 1.60, 'inferno', 'bone')

def cand_two_ink(M):
    """Three chosen colours, not a 256-entry ramp: warm ink for diverged,
    cool ink for converged, paper for fast interiors, boundary left black."""
    s, conv = speed01(M)
    s = np.clip(s,0,1) ** 0.6
    warm = np.array([0.78,0.22,0.16]); cool = np.array([0.16,0.28,0.55])
    paper= np.array([0.94,0.92,0.86])
    base = np.where(conv[...,None], cool, warm)
    out  = base*(1-s[...,None]) + paper*s[...,None]
    out[Ep] = 0.04                                   # boundary = the darkest thing
    return (np.clip(out,0,1)*255+0.5).astype(np.uint8)

CANDS = [("current spectral",cand_spectral),("dark_fire_ice",cand_fire_ice),
         ("dark_magma",cand_magma),("gamma hot/ice .55",cand_gamma_hot_ice),
         ("gamma ember .70",cand_gamma_ember),("gamma hard 1.6",cand_gamma_hard),
         ("two-ink + black edge",cand_two_ink)]

results=[]
imgs=[]
for name, fn in CANDS:
    rgb = fn(M); L = lum(rgb)
    key = L.mean(); commit = abs(key-0.5)
    # boundary visibility: |L(boundary) - L(local surround)|
    from scipy.ndimage import uniform_filter
    surround = uniform_filter(L, 9)
    bvis = float(np.abs(L[Ep]-surround[Ep]).mean())
    # thumbnail survival
    th = np.asarray(Image.fromarray(rgb).convert('L').resize((200,200)),np.float32)/255
    results.append((name,key,commit,bvis,float(th.std())))
    imgs.append((name,rgb))
    print(f"{name:22} key {key:.3f}  commit {commit:.3f}  boundary-vis {bvis:.4f}  thumb-sd {th.std():.3f}")

# contact sheet
cell=560; cols=4; rows=(len(imgs)+cols-1)//cols; pad=16; lab=34
sheet=Image.new('RGB',(cols*cell+pad*(cols+1), rows*(cell+lab)+pad*(rows+1)),(24,24,26))
d=ImageDraw.Draw(sheet)
try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
except: f=ImageFont.load_default()
for i,(name,rgb) in enumerate(imgs):
    r,c=divmod(i,cols)
    x=pad+c*(cell+pad); y=pad+r*(cell+lab+pad)
    sheet.paste(Image.fromarray(rgb).resize((cell,cell),Image.LANCZOS),(x,y))
    st=[t for t in results if t[0]==name][0]
    d.text((x,y+cell+6), f"{name}   commit {st[2]:.2f}  bvis {st[3]:.3f}", fill=(235,235,230), font=f)
sheet.save('gallery/study_mapping_probe.png')
print("\nsheet -> gallery/study_mapping_probe.png")
