"""Microbenchmark: raw ConvNet4 forward throughput (images/s) under precision/layout options."""
import time, json, torch
import torch.nn.functional as F
from dl_common import ConvNet4

torch.backends.cudnn.benchmark = True
net = ConvNet4().cuda().eval()
for tf32 in (False, True):
    torch.backends.cudnn.allow_tf32 = tf32; torch.backends.cuda.matmul.allow_tf32 = tf32
    for dt in (torch.float32, torch.bfloat16):
        if dt == torch.bfloat16 and tf32:
            continue
        for cl in (False, True):
            for N in (2048, 8192):
                n = net.to(dt)
                x = torch.randn(N, 3, 32, 32, device="cuda", dtype=dt)
                if cl:
                    n = n.to(memory_format=torch.channels_last); x = x.contiguous(memory_format=torch.channels_last)
                with torch.no_grad():
                    for _ in range(3): n(x)
                    torch.cuda.synchronize(); t = time.time(); R = 10
                    for _ in range(R): n(x)
                    torch.cuda.synchronize(); dtm = (time.time() - t) / R
                print(json.dumps(dict(tf32=tf32, dtype=str(dt), channels_last=cl, N=N, img_per_s=N / dtm,
                                      TFLOPs=N * 3.4e6 * 2 / dtm / 1e12)), flush=True)
                net = net.float().to(memory_format=torch.contiguous_format)
