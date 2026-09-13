"""Print the markdown box-counting table for README from cache/verify_zoomAB.json and
cache/verify_null_quadratic.json (run verify.py first)."""
import json
rows = json.load(open('cache/verify_zoomAB.json'))
nul = {round(r['zoom_decades'], 1): r for r in json.load(open('cache/verify_null_quadratic.json'))}
print('| plate | magnification | trainable | boundary px | D (network), b=2..32 px | 1-ulp flips (boundary px) | D (quadratic null, same magnification) |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    f = r['fit']; z = round(r['zoom_decades'], 1)
    n = nul.get(z); nf = n['fit'] if n else None
    print(f"| {r['k']+1} | 10^{z:.1f} | {100*r['conv']:.0f}% | {100*r['edge_frac']:.1f}% | "
          f"{f['D']:.2f} ± {f['se']:.2f} | {100*(r['flip_frac_edge'] or 0):.1f}% | "
          f"{(f'{nf[chr(68)]:.2f}' if nf else '-')} |")
