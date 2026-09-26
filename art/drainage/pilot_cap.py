import glob, numpy as np
from cycles import first_detection
S = []
for f in sorted(glob.glob("cache/pilot/shard00/b*.npz")):
    d = np.load(f); off = np.concatenate([[0], np.cumsum(d["length"])])
    S += [d["tokens"][off[i]:off[i+1]] for i in range(len(d["starts"]))]
for conf in (32, 64):
    L = np.array([first_detection(s, 128, 16, conf)[0] for s in S]); L[L == 0] = 10**6
    for cap in (192, 256, 320, 384, 512):
        stop = np.minimum(L, cap + 1)
        # attention work ~ sum of t over steps; weights ~ steps
        print(f"confirm {conf} cap {cap}: cycled {np.mean(L <= cap + 1):.3f}  mean steps {stop.mean():.0f}  attn-work {np.mean(stop**2)/2/1e3:.1f}k")
