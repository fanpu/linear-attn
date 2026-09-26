"""Simulate the production stop rule (cap 256 or 512, confirm 32) on the no-stop pilot, for analysis tests."""
import glob, os, sys, numpy as np
from cycles import first_detection
cap = int(sys.argv[1]); out = f"cache/pilot_sim{cap}/shard00"; os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob("cache/pilot/shard00/b*.npz")):
    d = np.load(f); off = np.concatenate([[0], np.cumsum(d["length"])])
    L, R, T = [], [], []
    for i in range(len(d["starts"])):
        s = d["tokens"][off[i]:off[i+1]][:cap + 1]
        eos = np.nonzero(np.isin(s, [151643, 151645]))[0]
        l1, p1 = first_detection(s, 80, 16, 32)
        if len(eos) and (l1 == 0 or eos[0] + 1 < l1):
            n, r = eos[0] + 1, 2
        elif l1:
            n, r = l1, 1
        else:
            n, r = len(s), 0
        L.append(n); R.append(r); T.append(s[:n])
    np.savez(os.path.join(out, os.path.basename(f)), starts=d["starts"], length=np.array(L, np.int32), reason=np.array(R, np.int8), tokens=np.concatenate(T))
