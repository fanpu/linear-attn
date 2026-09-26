import torch, time, numpy as np, torch.nn.functional as F, sys
sys.path.insert(0, "/home/fzeng/ml/research/art/one-road")
from models import ModTF, ModMLP
d=np.load('cache/data_modadd.npz'); x=torch.tensor(d['xtr']).cuda(); y=torch.tensor(d['ytr']).cuda()
for name, M in (("tf1", ModTF(97,1).cuda()), ("mlp", ModMLP(97).cuda())):
    for cap in (False, True):
        opt=torch.optim.AdamW(M.parameters(), 1e-3, weight_decay=1, capturable=cap)
        for i in range(3):
            l=F.cross_entropy(M(x),y); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize(); t=time.time()
        for i in range(100):
            l=F.cross_entropy(M(x),y); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize(); print(name, "capturable", cap, (time.time()-t)/100, flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for i in range(5):
            l=F.cross_entropy(M(x),y); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
