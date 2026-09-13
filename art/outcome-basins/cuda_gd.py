"""Per-pixel GD as raw CUDA kernels: one kernel thread = one pixel = one full training run
(all T steps, with per-pixel early exit).  On a shared GPU this matters: thousands of tiny torch
kernel launches per image get time-sliced to death, a few long launches do not.

All arithmetic is IEEE double.
status : 0 converged, 1 not converged by T, 2 diverged.
converged: loss < tol and it stays < tol for HOLD more steps (a transient dip near an unstable
           minimum is not counted).
tconv  : first step below tol, smoothed by log-linear interpolation between the bracketing steps
         (declared continuous-colouring choice, the analogue of a smoothed escape count).
diverged: max|theta| > thr or non-finite; tconv is the smoothed escape step (log-linear in max|theta|).
"""
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', os.path.join(HERE, 'cache', 'torch_ext'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1')
HOLD = 64

CUDA_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <algorithm>
#define HOLD %d
#define MAXD 8
#define BS 256

__device__ static inline double clamp01(double v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

// f = 1/4 (prod_i x_i - 1)^2
__global__ void fact_g(int64_t off, int64_t P, int D, const double* th0, double* th, double eta, int T,
                     double tol, double thr, double* tconv, int8_t* status, double* lossv) {
    int64_t p = off + (int64_t)blockIdx.x * BS + threadIdx.x; if (p >= P) return;
    double x[MAXD], g[MAXD];
    for (int i = 0; i < D; i++) x[i] = th0[p*D + i];
    double prevL = 1e300, prevM = 0, tc = T, tcb = 0, L = 0; int st = 1, below = -1;
    for (int i = 0; i < D; i++) prevM = fmax(prevM, fabs(x[i]));
    for (int t = 0; t < T; t++) {
        double pr = 1; for (int i = 0; i < D; i++) pr *= x[i];
        double r = pr - 1; L = 0.25 * r * r;
        if (L < tol) {
            if (below < 0) { below = t;
                double a = log(prevL), b = log(fmax(L, 1e-300));
                tcb = t == 0 ? 0 : (t - 1) + clamp01((a - log(tol)) / fmax(a - b, 1e-12)); }
            if (t - below >= HOLD) { st = 0; tc = tcb; break; }
        } else below = -1;
        for (int i = 0; i < D; i++) { double o = 1; for (int j = 0; j < D; j++) if (j != i) o *= x[j]; g[i] = 0.5 * r * o; }
        double M = 0; bool bad = false;
        for (int i = 0; i < D; i++) { x[i] -= eta * g[i]; double a = fabs(x[i]); if (!std::isfinite(a)) bad = true; M = fmax(M, a); }
        if (bad || M > thr) {
            double fr = 1;
            if (!bad && M > prevM) fr = clamp01((log(thr) - log(prevM)) / (log(M) - log(prevM)));
            st = 2; tc = t + fr; break;
        }
        prevM = M; prevL = L;
    }
    for (int i = 0; i < D; i++) th[p*D + i] = x[i];
    tconv[p] = tc; status[p] = (int8_t)st; lossv[p] = L;
}

// 2 -> H tanh -> 1 linear ; loss = 1/(2N) sum_n (y_n - Y_n)^2
// theta: W[k][0], W[k][1] (k<H) ; b[k] ; a[k] ; c
__global__ void mlp_g(int64_t off, int64_t P, int H, int N, const double* X, const double* Y, const double* th0, double* th,
                    double eta, int T, double tol, double thr, double* tconv, int8_t* status, double* lossv, int stall) {
    int64_t p = off + (int64_t)blockIdx.x * BS + threadIdx.x; if (p >= P) return;
    int D = 4*H + 1;
    double w[4*MAXD+1], gr[4*MAXD+1], h[32][MAXD], e[32];
    for (int i = 0; i < D; i++) w[i] = th0[p*D + i];
    double prevL = 1e300, prevM = 0, tc = T, tcb = 0, L = 0; int st = 1, below = -1;
    for (int i = 0; i < D; i++) prevM = fmax(prevM, fabs(w[i]));
    double bestL = 1e300; int bestT = 0;
    for (int t = 0; t < T; t++) {
        L = 0;
        for (int n = 0; n < N; n++) {
            double y = w[4*H];
            for (int k = 0; k < H; k++) { h[n][k] = tanh(w[2*k]*X[2*n] + w[2*k+1]*X[2*n+1] + w[2*H+k]); y += w[3*H+k]*h[n][k]; }
            e[n] = (y - Y[n]) / N; L += 0.5 * N * e[n] * e[n];
        }
        if (L < tol) {
            if (below < 0) { below = t;
                double a = log(prevL), b = log(fmax(L, 1e-300));
                tcb = t == 0 ? 0 : (t - 1) + clamp01((a - log(tol)) / fmax(a - b, 1e-12)); }
            if (t - below >= HOLD) { st = 0; tc = tcb; break; }
        } else below = -1;
        // stall: best loss has not improved by 1e-6 (relative) for `stall` steps while loss > 1e-4
        if (L < bestL * (1 - 1e-6)) { bestL = L; bestT = t; }
        if (stall > 0 && t - bestT > stall && bestL > 1e-4) { st = 1; tc = T; break; }
        for (int i = 0; i < D; i++) gr[i] = 0;
        for (int n = 0; n < N; n++) {
            gr[4*H] += e[n];
            for (int k = 0; k < H; k++) {
                gr[3*H+k] += e[n] * h[n][k];
                double dz = e[n] * w[3*H+k] * (1 - h[n][k]*h[n][k]);
                gr[2*k] += dz * X[2*n]; gr[2*k+1] += dz * X[2*n+1]; gr[2*H+k] += dz;
            }
        }
        double M = 0; bool bad = false;
        for (int i = 0; i < D; i++) { w[i] -= eta * gr[i]; double a = fabs(w[i]); if (!std::isfinite(a)) bad = true; M = fmax(M, a); }
        if (bad || M > thr) {
            double fr = 1;
            if (!bad && M > prevM) fr = clamp01((log(thr) - log(prevM)) / (log(M) - log(prevM)));
            st = 2; tc = t + fr; break;
        }
        prevM = M; prevL = L;
    }
    for (int i = 0; i < D; i++) th[p*D + i] = w[i];
    tconv[p] = tc; status[p] = (int8_t)st; lossv[p] = L;
}

// launch in slices of `per` pixels (one kernel launch each) so no single launch hogs the shared GPU
void fact_run(torch::Tensor th0, torch::Tensor th, double eta, int64_t T, double tol, double thr,
              torch::Tensor tconv, torch::Tensor status, torch::Tensor loss, int64_t per) {
    const at::cuda::CUDAGuard guard(th0.device());
    int64_t P = th0.size(0); int D = th0.size(1);
    for (int64_t off = 0; off < P; off += per) {
        int64_t n = std::min(per, P - off); int nb = (n + BS - 1) / BS;
        fact_g_launch_placeholder
    }
}
void mlp_run(torch::Tensor X, torch::Tensor Y, int64_t H, torch::Tensor th0, torch::Tensor th, double eta, int64_t T, double tol, double thr,
             torch::Tensor tconv, torch::Tensor status, torch::Tensor loss, int64_t per, int64_t stall) {
    const at::cuda::CUDAGuard guard(th0.device());
    int64_t P = th0.size(0); int N = X.size(0);
    for (int64_t off = 0; off < P; off += per) {
        int64_t n = std::min(per, P - off); int nb = (n + BS - 1) / BS;
        mlp_g_launch_placeholder
    }
}
'''
LAUNCH = '<<<'  # CUDA launch syntax assembled below
FACT_L = ('fact_g' + LAUNCH + 'nb, BS>>>(off, std::min(off + n, P), D, th0.data_ptr<double>(), th.data_ptr<double>(), eta, (int)T, tol, thr, '
          'tconv.data_ptr<double>(), status.data_ptr<int8_t>(), loss.data_ptr<double>()); torch::cuda::synchronize();')
MLP_L = ('mlp_g' + LAUNCH + 'nb, BS>>>(off, std::min(off + n, P), (int)H, N, X.data_ptr<double>(), Y.data_ptr<double>(), th0.data_ptr<double>(), '
         'th.data_ptr<double>(), eta, (int)T, tol, thr, tconv.data_ptr<double>(), status.data_ptr<int8_t>(), loss.data_ptr<double>(), (int)stall); torch::cuda::synchronize();')
CPP_SRC = r'''
#include <torch/extension.h>
void fact_run(torch::Tensor th0, torch::Tensor th, double eta, int64_t T, double tol, double thr, torch::Tensor tconv, torch::Tensor status, torch::Tensor loss, int64_t per);
void mlp_run(torch::Tensor X, torch::Tensor Y, int64_t H, torch::Tensor th0, torch::Tensor th, double eta, int64_t T, double tol, double thr, torch::Tensor tconv, torch::Tensor status, torch::Tensor loss, int64_t per, int64_t stall);
'''

_mod = None


def module():
    global _mod
    if _mod is None:
        src = (CUDA_SRC % HOLD) \
            .replace('fact_g_launch_placeholder', FACT_L).replace('mlp_g_launch_placeholder', MLP_L)
        src = src.replace('#include <cmath>', '#include <cmath>\n#include <torch/cuda.h>')
        os.makedirs(os.environ['TORCH_EXTENSIONS_DIR'], exist_ok=True)
        _mod = load_inline('outcome_gd2', cpp_sources=CPP_SRC, cuda_sources=src,
                           functions=['fact_run', 'mlp_run'], extra_cuda_cflags=['-O3'], verbose=False)
    return _mod


def _alloc(th0):
    th0 = torch.as_tensor(np.ascontiguousarray(th0), dtype=torch.float64).cuda()
    P = th0.shape[0]
    return th0, torch.empty_like(th0), torch.empty(P, dtype=torch.float64, device='cuda'), \
        torch.empty(P, dtype=torch.int8, device='cuda'), torch.empty(P, dtype=torch.float64, device='cuda')


def run(kind, th0, eta, T, tol=1e-12, thr=1e4, per=1 << 16, X=None, Y=None, H=None, chunk=1 << 22, stall=0):
    """kind 'fact' (depth = th0.shape[1]) or 'mlp'. Returns numpy dict like engine.run_gd."""
    m = module()
    P = th0.shape[0]
    out = dict(theta=np.empty(th0.shape), tconv=np.empty(P), status=np.empty(P, np.int8), loss=np.empty(P))
    for s in range(0, P, chunk):
        a, th, tc, st, lo = _alloc(th0[s:s + chunk])
        if kind == 'fact':
            m.fact_run(a, th, float(eta), int(T), float(tol), float(thr), tc, st, lo, int(per))
        else:
            Xt = torch.as_tensor(X, dtype=torch.float64).cuda().contiguous()
            Yt = torch.as_tensor(Y, dtype=torch.float64).cuda().contiguous()
            m.mlp_run(Xt, Yt, int(H), a, th, float(eta), int(T), float(tol), float(thr), tc, st, lo, int(per), int(stall))
        e = s + a.shape[0]
        out['theta'][s:e] = th.cpu().numpy(); out['tconv'][s:e] = tc.cpu().numpy()
        out['status'][s:e] = st.cpu().numpy(); out['loss'][s:e] = lo.cpu().numpy()
    return out
