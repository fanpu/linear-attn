"""check the Triton kernels (hfk) against the torch reference (solid.march_fast / shadow_fast, hf)."""
import os, time, json, torch
import pasar_job
pasar_job.apply_memory_limit()
import solid, hf, hfk
from dense import field, Z0, make_cam
dev = "cuda"
F = field("B1", 4); Hm, GX, GY = solid.prep(F, 4, 0.4, Z0(), dev)
L = solid.light_dir(200, 30)
torch.cuda.synchronize(); t = time.time(); s_ref = hf.shadow_map(Hm, L, softness=18, step_texels=0.75); torch.cuda.synchronize(); t_ref = time.time() - t
hfk.shadow(Hm, L)  # compile
torch.cuda.synchronize(); t = time.time(); s_k = hfk.shadow(Hm, L, softness=18, step_texels=0.75); torch.cuda.synchronize(); t_k = time.time() - t
d = (s_ref - s_k).abs()
print(json.dumps(dict(shadow_ref_s=t_ref, shadow_triton_s=t_k, mean_abs=float(d.mean()), frac_gt_0p05=float((d > 0.05).float().mean()))), flush=True)
cam = make_cam("0,0.05,0.1,1.1,-90,50,50", 1280, 720)
os.environ["NO_TRITON"] = "1"
torch.cuda.synchronize(); t = time.time(); G0 = solid.gbuffer(Hm, cam, ss=1, device=dev); torch.cuda.synchronize(); t0 = time.time() - t
del os.environ["NO_TRITON"]
solid.gbuffer(Hm, cam, ss=1, device=dev)
torch.cuda.synchronize(); t = time.time(); G1 = solid.gbuffer(Hm, cam, ss=1, device=dev); torch.cuda.synchronize(); t1 = time.time() - t
m = G0["m"] & G1["m"]
dp = (G0["p"] - G1["p"]).norm(dim=-1)[m]
print(json.dumps(dict(march_torch_s=t0, march_triton_s=t1, hit_agree=float((G0["m"] == G1["m"]).float().mean()),
                      pos_med=float(dp.median()), frac_gt_0p01=float((dp > 0.01).float().mean()))), flush=True)
cam = make_cam("0,0.05,0.1,1.1,-90,50,50", 2560, 1440)
torch.cuda.synchronize(); t = time.time(); G = solid.gbuffer(Hm, cam, ss=2, device=dev); torch.cuda.synchronize()
print(json.dumps(dict(march_triton_2560ss2_s=time.time() - t)), flush=True)
