"""Declared, measurable phase boundaries (computed per seed from cache/analysis.npz)."""
import numpy as np


def phases(A, s):
    tr_acc = A["tr_acc"][:, s]
    ev = A["ev_steps"]; te_acc = A["te_acc"][:, s]
    fs = A["full_steps"]; exc = A["excluded"][:, s]
    t_mem = int(np.argmax(tr_acc >= 0.9999))                     # train accuracy hits 100%
    # circuit formation begins when excluded loss (train set, key frequencies ablated) turns upward
    after = fs >= 0
    i_min = int(np.argmin(np.where(fs >= t_mem * 0.5, exc, np.inf)))
    t_circ = int(fs[i_min])
    t_clean = int(ev[np.argmax(te_acc >= 0.10)])                  # test accuracy leaves chance (>=10%)
    t_grok = int(ev[np.argmax(te_acc >= 0.50)])                   # 50% test accuracy
    t_done = int(ev[np.argmax(te_acc >= 0.999)]) if (te_acc >= 0.999).any() else -1
    return dict(t_mem=t_mem, t_circ=t_circ, t_clean=t_clean, t_grok=t_grok, t_done=t_done)
