"""Riso and isoline print variants of the native 1024^2 deep plate. CPU only: python render_deep_styles.py"""
import numpy as np
from PIL import Image
import styles as S
from render_windows import load
M = load('deep_zoomA4_1024_f64')['M']
out = S.isolines(M, px=2048)
if out is not None:
    Image.fromarray(np.asarray(out)).save('gallery/hero_deep_swirl_isolines_print.png')
Image.fromarray(S.riso_two_ink(M, scale=2, period=3.0)).save('gallery/hero_deep_swirl_riso_print.png')
print('deepstyles done')
