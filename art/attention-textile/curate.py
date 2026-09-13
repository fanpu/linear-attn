"""Move weaker / duplicate renders to gallery/extra/ (run after the render scripts)."""
import os, shutil

EXTRA = ['qwen_tapestry_dark_P12R4.png', 'book_series_dark.png', 'book_series_indigo.png',
         'selfsim_plate_dark.png', 'selfsim_plate_weave.png', 'selfsim_cantor_riso.png',
         'word_thue-morse_dark.png', 'word_fibonacci_dark.png', 'toy_strip_indigo.png', 'toy_loom_weave.png']
os.makedirs('gallery/extra', exist_ok=True)
for f in EXTRA:
    if os.path.exists(f'gallery/{f}'):
        shutil.move(f'gallery/{f}', f'gallery/extra/{f}')
        print('moved', f)
