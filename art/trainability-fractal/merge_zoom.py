"""Merge zoomA (half-decade keyframes 0..5, down to 10^2.5) and zoomB (decade keyframes
from 10^3.5, centred on zoomA kf5's boundary) into one nested sequence cache/zoom_zoomAB."""
import glob, json, os, shutil
import numpy as np
out = 'cache/zoom_zoomAB'
os.makedirs(out, exist_ok=True)
meta = dict(args=dict(note='zoomA kf0-5 (dec 0.5) + zoomB (dec 1.0)'), keyframes=[])
k = 0
for tag, keep in (('zoomA', range(0, 6)), ('zoomB', None)):
    pj = json.load(open(f'cache/zoom_{tag}/path.json'))['keyframes']
    for f in sorted(glob.glob(f'cache/zoom_{tag}/kf_*.npz')):
        i = int(f[-7:-4])
        if keep is not None and i not in keep:
            continue
        shutil.copy(f, f'{out}/kf_{k:03d}.npz')
        e = dict(pj[i]) if i < len(pj) else {}
        e['k'] = k; e['source'] = f'{tag}:{i}'
        meta['keyframes'].append(e)
        k += 1
json.dump(meta, open(f'{out}/path.json', 'w'), indent=1)
print('merged', k, 'keyframes')
