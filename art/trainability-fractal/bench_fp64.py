"""Measure the machine's actual fp64 ceilings: dependent-chain FMA, ILP FMA, and
tanh. hardware/roofline only measured bf16, and the whole plan for this engine
depends on where the fp64 roof really is.
"""
import os, time, torch
from torch.utils.cpp_extension import load_inline

SRC = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>

template<int ILP>
__global__ void fma_kernel(double* out, long iters, double a) {
  double x[ILP], b = a * 1.0000001;
  #pragma unroll
  for (int i = 0; i < ILP; ++i) x[i] = a + i;
  for (long t = 0; t < iters; ++t) {
    #pragma unroll
    for (int i = 0; i < ILP; ++i) x[i] = fma(x[i], b, a);
  }
  double s = 0;
  #pragma unroll
  for (int i = 0; i < ILP; ++i) s += x[i];
  if (s == 1234.5) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void tanh_kernel(double* out, long iters, double a) {
  double x = a;
  for (long t = 0; t < iters; ++t) x = tanh(x) + a;
  if (x == 1234.5) out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void tanh_ilp_kernel(double* out, long iters, double a) {
  double x[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = a + i * 0.01;
  for (long t = 0; t < iters; ++t) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = tanh(x[i]) + a;
  }
  double s = x[0] + x[1] + x[2] + x[3];
  if (s == 1234.5) out[0] = s;
}

void fma1(torch::Tensor o, long it, double a){ fma_kernel<1><<<o.numel()/256,256>>>(o.data_ptr<double>(), it, a); }
void fma8(torch::Tensor o, long it, double a){ fma_kernel<8><<<o.numel()/256,256>>>(o.data_ptr<double>(), it, a); }
void tanh1(torch::Tensor o, long it, double a){ tanh_kernel<<<o.numel()/256,256>>>(o.data_ptr<double>(), it, a); }
void tanh4(torch::Tensor o, long it, double a){ tanh_ilp_kernel<<<o.numel()/256,256>>>(o.data_ptr<double>(), it, a); }
'''

DECL = '''
#include <torch/extension.h>
void fma1(torch::Tensor o, long it, double a);
void fma8(torch::Tensor o, long it, double a);
void tanh1(torch::Tensor o, long it, double a);
void tanh4(torch::Tensor o, long it, double a);
'''
import os
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1')
m = load_inline('fp64bench', cpp_sources=DECL, cuda_sources=SRC,
                functions=['fma1', 'fma8', 'tanh1', 'tanh4'],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False)

props = torch.cuda.get_device_properties(0)
SM = props.multi_processor_count
print(f'{props.name} sm_{props.major}{props.minor} SMs={SM} clock={props.clock_rate/1e6:.3f} GHz')

def bench(fn, threads, iters, per_iter, label, reps=5):
    o = torch.zeros(threads, dtype=torch.float64, device='cuda')
    fn(o, 100, 1.0000001); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(o, iters, 1.0000001)
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
    ops = threads * iters * per_iter
    print(f'{label:34s} {best*1e3:8.2f} ms  {ops/best/1e9:9.2f} G-op/s  '
          f'({ops/best/SM/(props.clock_rate*1e3):6.3f} op/SM/clk)')
    return ops / best

# 48 SMs * 1536 threads = 73728 resident threads; use 4x that to be safe
T = SM * 1536
bench(m.fma1, T, 20000, 1, 'fp64 FMA, dependent chain')
bench(m.fma8, T, 20000, 8, 'fp64 FMA, ILP 8')
f = bench(m.fma8, T // 4, 40000, 8, 'fp64 FMA, ILP 8, quarter occupancy')
t1 = bench(m.tanh1, T, 2000, 1, 'fp64 tanh, dependent chain')
t4 = bench(m.tanh4, T, 2000, 4, 'fp64 tanh, ILP 4')
print(f'\ntanh cost  ~ {f/t4:.1f} fp64 FMA-equivalents')
