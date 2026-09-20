// One thread warp trains one network for all `steps` steps entirely on-chip.
//
// Why this shape: the whole job per pixel is a 272x16 dataset X (shared by every
// pixel, 34 KB, staged once per block into shared memory) and a 16x16 + 16x1
// parameter set (2.2 KB, lives in registers). PyTorch's (N, P*n) GEMM layout
// materialises the (272, P, 16) activations in HBM 500 times per chunk; here they
// never leave the register file and the only global write is one double per pixel.
//
// Warp layout: lane = 16*half + e, so lane (half, e) owns column e of W0 and the
// gradient accumulator for that column, and processes rows i = half, half+2, ...
// Two rows in flight per warp means all 32 lanes compute a *distinct* tanh, which
// matters because fp64 tanh is the single most expensive op in the loop.
//
//   z[i,e]  = sum_d X[i,d] W0[d,e]        16 FMA/lane, no cross-lane traffic
//   out[i]  = sum_e h[i,e] W1[e] / n      4-step butterfly inside each 16-lane half
//   gW1[e] += h[i,e] r'[i]                lane-local
//   gW0[d,e]+= X[i,d] dZ[i,e]             16 FMA/lane, no cross-lane traffic
//
// Only the two half-warps have to be combined, once per step (17 shuffles).
//
// Early exit is per-warp: a diverged network just leaves the loop, so unlike the
// batched engine there is no compaction and no shape change.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#define MAXV 1.0e6
#define LASTK 20
#define WD 16            // network width; the whole lane layout assumes 16

// nonlinearity ids match tfractal's strings
enum { NL_TANH = 0, NL_RELU = 1, NL_SIN = 2, NL_IDENT = 3 };

template <int NL, int BLOCK>
__launch_bounds__(BLOCK)
__global__ void train_kernel(
    const double* __restrict__ Xg,
    const double* __restrict__ Yg,
    const double* __restrict__ W0g,
    const double* __restrict__ W1g,
    const double* __restrict__ lr0g,
    const double* __restrict__ lr1g,
    const double* __restrict__ sig0g,
    const double* __restrict__ sig1g,
    const double* __restrict__ wdg,
    double* __restrict__ outg,
    int P, int N, int steps, int check_every, double exit_loss, int early_exit)
{
    extern __shared__ double sm[];
    double* Xs = sm;              // N x 16, row major
    double* Ys = sm + (size_t)N * WD;

    for (int i = threadIdx.x; i < N * WD; i += BLOCK) Xs[i] = Xg[i];
    for (int i = threadIdx.x; i < N; i += BLOCK) Ys[i] = Yg[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int e    = lane & (WD - 1);
    const int half = lane >> 4;
    const int pix  = blockIdx.x * (BLOCK / 32) + (threadIdx.x >> 5);
    if (pix >= P) return;

    const double a0 = 1.0 / sqrt((double)WD);
    const double s2 = sqrt(2.0);
    const double rscale = 2.0 / (double)N / (double)WD;   // matches tfractal's 2.0/N/n
    const unsigned FULL = 0xffffffffu;

    double w0[WD], g0[WD];
#pragma unroll
    for (int d = 0; d < WD; ++d) w0[d] = W0g[d * WD + e];
    double w1 = W1g[e];
    if (sig0g) { double s = sig0g[pix]; 
#pragma unroll
        for (int d = 0; d < WD; ++d) w0[d] *= s; }
    if (sig1g) w1 *= sig1g[pix];

    const double lr0 = lr0g[pix], lr1 = lr1g[pix];
    const double wd  = wdg ? wdg[pix] : 0.0;

    double S = 0.0, Sinv = 0.0, Win = 0.0, v0 = 0.0;
    const int Twin = steps - LASTK;
    const int rowpairs = (N + 1) >> 1;

    for (int t = 0; t < steps; ++t) {
#pragma unroll
        for (int d = 0; d < WD; ++d) g0[d] = 0.0;
        double gw1 = 0.0, lsum = 0.0;

        for (int k = 0; k < rowpairs; ++k) {
            const int i  = 2 * k + half;
            const int ic = (i < N) ? i : 0;            // clamp; masked out below
            const double* __restrict__ xr = Xs + (size_t)ic * WD;

            double z = 0.0;
#pragma unroll
            for (int d = 0; d < WD; ++d) z = fma(xr[d], w0[d], z);
            z *= a0;

            double h, dphi;
            if (NL == NL_TANH)       { h = tanh(z * s2);  dphi = s2 * (1.0 - h * h); }
            else if (NL == NL_RELU)  { h = (z > 0.0 ? z : 0.0) * s2; dphi = (z > 0.0) ? s2 : 0.0; }
            else if (NL == NL_SIN)   { h = sin(z * s2);   dphi = s2 * cos(z * s2); }
            else                     { h = z;             dphi = 1.0; }

            double p = h * w1;
#pragma unroll
            for (int m = 1; m < WD; m <<= 1) p += __shfl_xor_sync(FULL, p, m);
            // 1/16 is exact, so this is bit-identical to torch's `.sum(-1) / n`
            double r = p * (1.0 / (double)WD) - Ys[ic];
            if (i >= N) r = 0.0;                        // pad row contributes nothing

            lsum += r * r;
            const double rr = r * rscale;
            gw1 = fma(h, rr, gw1);
            const double dz = dphi * (rr * w1) * a0;
#pragma unroll
            for (int d = 0; d < WD; ++d) g0[d] = fma(xr[d], dz, g0[d]);
        }

        // combine the two half-warps
        double loss = (lsum + __shfl_xor_sync(FULL, lsum, 16)) / (double)N;
        double G1   = gw1 + __shfl_xor_sync(FULL, gw1, 16);
#pragma unroll
        for (int d = 0; d < WD; ++d) g0[d] += __shfl_xor_sync(FULL, g0[d], 16);

        // Sohl-Dickstein's convergence measure, accumulated in place
        const double l = isfinite(loss) ? loss : MAXV;
        if (t == 0) v0 = l;
        double v = l / v0;
        if (v > MAXV) v = MAXV;                         // nan propagates, as torch.clamp does
        S += v;
        Sinv += 1.0 / v;
        if (t >= Twin) Win += v;

        if (t == steps - 1) break;                      // last update is never used

        if (early_exit && ((t + 1) % check_every) == 0 &&
            (!isfinite(loss) || loss > exit_loss)) {
            const int rem = steps - (t + 1);
            S += rem * v;
            Sinv += rem / v;
            int started = (t + 1) - Twin;
            if (started < 0) started = 0;
            Win += (double)(LASTK - started) * v;
            break;
        }

#pragma unroll
        for (int d = 0; d < WD; ++d) w0[d] -= lr0 * (g0[d] + wd * w0[d]);
        w1 -= lr1 * (G1 + wd * w1);
    }

    if (lane == 0) outg[pix] = ((Win / (double)LASTK) < 1.0) ? -S : Sinv;
}

// ---------------------------------------------------------------------------

template <int NL>
static void launch(const double* X, const double* Y, const double* W0, const double* W1,
                   const double* lr0, const double* lr1, const double* s0, const double* s1,
                   const double* wd, double* out, int P, int N, int steps, int check_every,
                   double exit_loss, int early_exit, int block, size_t shmem)
{
    const int warps = block / 32;
    const int grid = (P + warps - 1) / warps;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define LAUNCH(B)                                                                  \
    do {                                                                           \
        cudaFuncSetAttribute(train_kernel<NL, B>,                                  \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);              \
        train_kernel<NL, B><<<grid, B, shmem, st>>>(X, Y, W0, W1, lr0, lr1, s0, s1,\
            wd, out, P, N, steps, check_every, exit_loss, early_exit);             \
    } while (0)
    switch (block) {
        case 64:  LAUNCH(64);  break;
        case 128: LAUNCH(128); break;
        case 160: LAUNCH(160); break;
        case 192: LAUNCH(192); break;
        case 224: LAUNCH(224); break;
        case 256: LAUNCH(256); break;
        default: TORCH_CHECK(false, "block must be 64/128/160/192/224/256");
    }
#undef LAUNCH
}

static const double* dp(const c10::optional<torch::Tensor>& t) {
    if (!t.has_value() || !t->defined()) return nullptr;
    return t->data_ptr<double>();
}

torch::Tensor train(torch::Tensor X, torch::Tensor Y, torch::Tensor W0, torch::Tensor W1,
                    torch::Tensor lr0, torch::Tensor lr1,
                    c10::optional<torch::Tensor> sig0, c10::optional<torch::Tensor> sig1,
                    c10::optional<torch::Tensor> wd,
                    int64_t steps, int64_t nonlin, bool early_exit, int64_t check_every,
                    double exit_loss, int64_t block)
{
    TORCH_CHECK(X.scalar_type() == torch::kFloat64, "float64 only");
    TORCH_CHECK(X.size(1) == WD, "this kernel is specialised to width 16");
    TORCH_CHECK(X.is_contiguous() && Y.is_contiguous() && W0.is_contiguous() && W1.is_contiguous());
    TORCH_CHECK(lr0.is_contiguous() && lr1.is_contiguous());
    const int N = (int)X.size(0);
    const int P = (int)lr0.numel();
    const size_t shmem = (size_t)(N * WD + N) * sizeof(double);
    auto out = torch::empty({P}, X.options());
    if (P == 0) return out;

    auto f = [&](auto tag) {
        constexpr int NL = decltype(tag)::value;
        launch<NL>(X.data_ptr<double>(), Y.data_ptr<double>(), W0.data_ptr<double>(),
                   W1.data_ptr<double>(), lr0.data_ptr<double>(), lr1.data_ptr<double>(),
                   dp(sig0), dp(sig1), dp(wd), out.data_ptr<double>(), P, N, (int)steps,
                   (int)check_every, exit_loss, early_exit ? 1 : 0, (int)block, shmem);
    };
    switch (nonlin) {
        case NL_TANH:  f(std::integral_constant<int, NL_TANH>{});  break;
        case NL_RELU:  f(std::integral_constant<int, NL_RELU>{});  break;
        case NL_SIN:   f(std::integral_constant<int, NL_SIN>{});   break;
        case NL_IDENT: f(std::integral_constant<int, NL_IDENT>{}); break;
        default: TORCH_CHECK(false, "unknown nonlin id");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("train", &train, "fused per-warp trainability kernel (fp64)");
}
