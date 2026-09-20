import sys, os
sys.path.insert(0,"/home/fzeng/ml/research/art/trainability-fractal")
os.chdir("/home/fzeng/ml/research/art/trainability-fractal")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use('Agg')
import styles as S
from common_render import edges, speed01

M = np.load('cache/windows/deep_zoomA4_1024_f64.npz')['measure']   # numpy only, no GPU
E = np.pad(edges(M), ((0,1),(0,1)), constant_values=False)

before = S.spectral(M)

s, conv = speed01(M); s = np.clip(s,0,1)**0.6
warm=np.array([0.80,0.20,0.13]); cool=np.array([0.10,0.20,0.46]); paper=np.array([0.95,0.93,0.87])
after = np.where(conv[...,None],cool,warm)*(1-s[...,None]) + paper*s[...,None]
after[E]=0.03
after=(np.clip(after,0,1)*255+0.5).astype(np.uint8)

C=900; pad=18; lab=64
sheet=Image.new('RGB',(2*C+3*pad, C+lab+2*pad),(20,20,22)); d=ImageDraw.Draw(sheet)
try:
    f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",26)
    fs=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",19)
except: f=fs=ImageFont.load_default()
for i,(nm,sub,img) in enumerate([
      ("BEFORE  (current hero)","boundary contrast 0.026 · key commitment 0.07",before),
      ("AFTER  (two-ink, boundary forced black)","boundary contrast 0.237 · key commitment 0.17",after)]):
    x=pad+i*(C+pad)
    sheet.paste(Image.fromarray(img).resize((C,C),Image.LANCZOS),(x,pad))
    d.text((x,pad+C+10), nm, fill=(240,240,235), font=f)
    d.text((x,pad+C+42), sub, fill=(150,150,148), font=fs)
sheet.save('gallery/study_before_after.png')
print('saved gallery/study_before_after.png', sheet.size)
