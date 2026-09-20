"""Fused CUDA engine for tfractal: one warp trains one network, all 500 steps on-chip.

Drop-in replacement for tfractal.train_chunk on the full-batch path (no minibatch,
no checkpoints). See cuda/tfkernel.cu for the layout and why it is shaped that way.

  import tfast; tfast.available()            -> compiles on first call
  tfast.train_chunk(prob, lr0, lr1, ...)      -> dict(measure=(P,) float64)
"""
import os
import torch

_MOD = None
_ERR = None
NONLIN_ID = {'tanh': 0, 'relu': 1, 'sin': 2, 'identity': 3}
_HERE = os.path.dirname(os.path.abspath(__file__))


def load(verbose=False):
    """Build (once) and return the extension module, or raise."""
    global _MOD, _ERR
    if _MOD is None:
        if _ERR is not None:
            raise _ERR
        from torch.utils.cpp_extension import load as _load
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1')
        try:
            _MOD = _load(name='tfkernel',
                         sources=[os.path.join(_HERE, 'cuda', 'tfkernel.cu')],
                         extra_cuda_cflags=['-O3', '-lineinfo'],
                         verbose=verbose)
        except Exception as e:                       # pragma: no cover
            _ERR = e
            raise
    return _MOD


def available(prob=None, **kw):
    """True if the fused kernel can serve this call."""
    if prob is not None:
        if prob.get('width', 16) != 16 or prob['nonlin'] not in NONLIN_ID:
            return False
        if prob['X'].device.type != 'cuda' or prob['X'].dtype != torch.float64:
            return False
    if kw.get('minibatch') or kw.get('checkpoints'):
        return False
    try:
        load()
    except Exception:
        return False
    return True


# One launch per call; keep each launch a few seconds so a display-attached GPU is
# never held for an unbounded time and progress stays inspectable.
LAUNCH = 65536
BLOCK = 128


def train_chunk(prob, lr0, lr1, steps=500, sigma0=None, sigma1=None, wd=None,
                minibatch=None, early_exit=True, check_every=25, exit_loss=1e100,
                checkpoints=None, compiled=True, block=None, launch=None):
    if minibatch or checkpoints:
        raise ValueError('fused engine handles the full-batch, single-T case only')
    m = load()
    nl = NONLIN_ID[prob['nonlin']]
    X, Y = prob['X'].contiguous(), prob['Y'].contiguous()
    W0, W1 = prob['W0'].contiguous(), prob['W1'].contiguous()
    lr0 = lr0.contiguous(); lr1 = lr1.contiguous()
    P = lr0.numel()
    L = launch or LAUNCH
    out = torch.empty(P, dtype=torch.float64, device=lr0.device)
    for s in range(0, P, L):
        e = min(P, s + L)
        sl = lambda v: None if v is None else v[s:e].contiguous()
        out[s:e] = m.train(X, Y, W0, W1, lr0[s:e].contiguous(), lr1[s:e].contiguous(),
                           sl(sigma0), sl(sigma1), sl(wd),
                           int(steps), int(nl), bool(early_exit), int(check_every),
                           float(exit_loss), int(block or BLOCK))
    return dict(measure=out)
