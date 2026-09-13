"""Build the nested deep-zoom sequence cache/zoom_zoomAB from
  zoomA  (half-decade keyframes 0..5, 10^0 .. 10^2.5, auto-chosen centres),
  zoomB  (decade keyframes from 10^3.5, auto-chosen centres, started inside zoomA kf5),
  fill   (half-decade in-betweens: window of half-width hw_prev/sqrt(10) centred on the
          NEXT deeper keyframe's centre, so every window contains the next one).

  python merge_zoom.py fillpath   # writes cache/fill_path.json (then run zoom_compute --tag fill --path ...)
  python merge_zoom.py            # merges everything that exists, sorted by magnification
"""
import glob, json, math, os, shutil, sys
import numpy as np


def kf_list(tag, keep=None):
    pj = json.load(open(f'cache/zoom_{tag}/path.json'))['keyframes'] if os.path.exists(f'cache/zoom_{tag}/path.json') else []
    out = []
    for f in sorted(glob.glob(f'cache/zoom_{tag}/kf_*.npz')):
        i = int(f[-7:-4])
        if keep is not None and i not in keep:
            continue
        d = np.load(f)
        e = dict(pj[i]) if i < len(pj) else {}
        e.update(c0=float(d['c0']), c1=float(d['c1']), hw=float(d['hw']), file=f, source=f'{tag}:{i}')
        out.append(e)
    return out


main = kf_list('zoomA', range(0, 6)) + kf_list('zoomB')
if len(sys.argv) > 1 and sys.argv[1] == 'fillpath':
    fills = []
    for a, b in zip(main[:-1], main[1:]):
        if a['hw'] / b['hw'] > 5:           # a decade gap
            hw = a['hw'] / math.sqrt(10)
            ok = abs(b['c0'] - a['c0']) + hw <= a['hw'] and abs(b['c1'] - a['c1']) + hw <= a['hw']
            print(f"fill between {a['source']} and {b['source']}: hw={hw:.3e} nested={ok}")
            fills.append(dict(k=len(fills), c0=b['c0'], c1=b['c1'], hw=hw, res=256))
    json.dump(dict(keyframes=fills), open('cache/fill_path.json', 'w'), indent=1)
    raise SystemExit
allk = sorted(main + kf_list('fill'), key=lambda e: -e['hw'])
out = 'cache/zoom_zoomAB'
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
meta = dict(args=dict(note='zoomA kf0-5 (dec 0.5) + zoomB (dec 1.0) + half-decade fills'), keyframes=[])
for k, e in enumerate(allk):
    shutil.copy(e.pop('file'), f'{out}/kf_{k:03d}.npz')
    e['k'] = k
    meta['keyframes'].append(e)
json.dump(meta, open(f'{out}/path.json', 'w'), indent=1)
print('merged', len(allk), 'keyframes:', ' '.join(f"{math.log10(4.5/e['hw']):.1f}" for e in allk))
