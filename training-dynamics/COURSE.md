# training-dynamics: course charter and curriculum map

A self-study course on neural-network training dynamics for Fan Pu, built from `BRIEF.md` (his brief,
2026-10-29) in the format of `../docs/course-template.md`. Nine assignments plus a capstone, all built
at once, core track CPU-only so it can be done offline on a flight. Worked solutions sit in plain sight.

**New session? Read `../docs/course-template.md`, then this file, then `STATE.md`.**

## 1. Course-specific decisions

- **Anchor model and notation** (from the post and the brief, kept for every assignment):
  $f_{a,b}(x)=abx$, $L(a,b)=\tfrac12(ab-1)^2$ after absorbing $\tfrac1n\sum x_i^2=1$, gradient flow
  $\dot a=-\partial_aL$, $\dot b=-\partial_bL$; GD step $\eta$; residual $r=ab-1$; product $p=ab$;
  imbalance $c=a^2-b^2$; weight decay $\gamma$ in the convention $L_\gamma=L+\gamma(a^2+b^2)$ so the
  critical value is $\gamma=\tfrac12$; label-noise variance $\sigma^2$; sharpness
  $\lambda_{\max}(\nabla^2L)$. Matrices $W_1,W_2$; width $m$; target scalar $s$.
- **Growth order of the anchor** (brief): scalar two layers (A1) → weight decay (A2) → depth 3 (A2) →
  finite steps (A3) → noise (A4) → matrices and a width-$m$ ReLU layer (A5) → softmax output (A6) →
  width scaling (A7).
- **Compute**: everything in the core track runs on a laptop CPU; per-assignment wall time for the
  checks is listed below and measured at build time. `[GPU]` extensions are optional and go through
  pasar when he is home.
- **Environment**: one `uv` project at this directory, CPU torch wheels, numpy, scipy, matplotlib,
  pytest. Datasets predownloaded into `data/` as `.npz` (MNIST, a 5k CIFAR-10 subset); no torchvision,
  no run-time downloads.
- **Documents**: Typst, visual style of `../day4/handout/template.typ`; `template/build.sh` compiles.
  Handouts 5-8 pages, solutions as long as they need to be.
- **Papers**: all in `papers/` (`<firstauthor><year>-<slug>.pdf`), plus the source post as
  `papers/ang2026-training-dynamics-post.{html,md}`.

## 2. Curriculum map

Study order is top to bottom. Hours are estimates for Fan Pu, derivations included. "Check CPU" is
the total wall time of the assignment's numerical checks on a laptop.

| # | Assignment | Brief area | What he derives and builds | Papers used (attribution points) | Hours | Check CPU | GPU ext. |
|---|---|---|---|---|---|---|---|
| A0 | `a0-prereqs` Warm-up: the five tools | Prereqs, framing | Gradient flow as the $\eta\to0$ limit and when it fails; GD on a quadratic converges iff $\lambda<2/\eta$; fixed points, linearization, saddle vs stable, phase portrait of a 2-D ODE; Hessian, top eigenvalue and trace by Hessian-vector products and power iteration; SVD turns linear regression into independent modes; an Itô SDE as drift plus noise, the OU process and its stationary variance. Implements `gd`, `power_iteration`, `classify_fixed_point`, `ou_simulate`. Reads Simon et al. §1-2 and Biderman et al. as framing. | Simon 2026 (map), Biderman 2026 (motivation) | 4 | < 1 min | none |
| A1 | `a1-symmetries` Symmetries and conservation laws | Area 1 | Noether-style argument for $a^2-b^2$; solve the logistic ODE $\dot u=2u(s-u)$ and the time-to-learn as a function of $s$ and init scale; exact endpoint on $ab=1$ from $c$; prove and check $\tfrac{d}{dt}(W_1W_1^\top-W_2^\top W_2)=0$; Saxe mode decomposition of a deep linear net and the plateau-then-drop curve; how finite steps ($1-\eta^2r^2$ factor), momentum and weight decay break the law; modified gradient flow (Kunin) predicts the drift and the check measures it. | Saxe 2014, Du-Hu-Lee 2018, Kunin 2021 | 5 | ~15 s | none |
| A2 | `a2-weight-decay` Weight decay and phase transitions | Area 5 | All stationary points of $L_\gamma$ and their classification; bifurcation at $\gamma=\tfrac12$ via the Hessian at the origin; $ab$ vs $\gamma$ curve (continuous); critical slowing down, relaxation time diverging as $\gamma\to\tfrac12$; depth 3: origin is a minimum for every $\gamma>0$, the nonzero solution appears at a finite $\gamma^*$ and $ab$ jumps (discontinuous); basins of the zero solution; finite-step oscillation and divergence regions in the $(\eta,\gamma)$ plane. | Ziyin 2022 | 4 | ~3 min | none |
| A3 | `a3-edge-of-stability` Finite steps and the edge of stability | Area 3 | Backward error analysis to first order and the IGR term $\tfrac\eta4\|\nabla L\|^2$ for the anchor by hand; GD vs gradient flow on $L$ and on $\tilde L$; stability of minima on $ab=1$ ($\lambda_{\max}=a^2+b^2<2/\eta$), the $\eta=1$, $(2,2)$ divergence, the largest $\eta$ with a stable minimum; simulate GD from an init whose flow endpoint is unstable and watch sharpness settle at $2/\eta$; progressive sharpening and EoS on a small MLP (shrunk: 2-layer MLP, 1k-example CIFAR-10 subset, full-batch GD, power-iteration sharpness every $k$ steps); scalar RMSProp on the anchor via the central-flow view: the adaptive step seeks low curvature. Smith et al. on SGD's extra implicit term as a derivation. | Barrett-Dherin 2021, Smith 2021, Cohen 2021, Cohen 2025 | 6 | ~25 min | EoS on 5k CIFAR-10, deeper MLP, and full RMSProp central-flow comparison (~1 GB10 h) |
| A4 | `a4-sgd-noise` Noise and drift along the solution manifold | Area 4 | Minibatch noise vanishes at any exact minimum of the anchor, so SGD sits at $(2,\tfrac12)$; label-noise SGD implemented and started there, drift to $(1,1)$ observed; the OU picture on the manifold: fast transverse oscillation with variance $\propto\eta\sigma^2$, slow drift along $ab=1$ down $a^2+b^2$; drift time scaling in $\eta$ and $\sigma^2$ derived and measured; Blanc's regularizer and $\operatorname{tr}\nabla^2L$ both equal $a^2+b^2$; Li-Wang-Arora's manifold SDE as the general statement; label noise plus weight decay (the post's extension). | Blanc 2020, Damian 2021, Li-Wang-Arora 2022 | 5 | ~5 min | none |
| A5 | `a5-lazy-rich` Implicit bias: lazy vs rich | Area 2 | NTK as gradient inner products; function-space ODE $\dot f=-K(f-y)$; anchor kernel $K=(a^2+b^2)xx'$ with $a^2+b^2=\sqrt{c^2+4p^2}$, kernel movement as a function of $c$ (lazy at large $|c|$, rich at small balanced init); Woodworth's diagonal linear net on sparse regression, init-scale sweep, test error vs scale, $\ell_2$-like vs $\ell_1$-like bias; Get-rich-quick: width-$m$ two-layer ReLU, kernel movement for overall vs relative init scale; grokking as a delayed lazy-to-rich transition on a small modular-addition MLP with a misaligned initial NTK. | Jacot 2018, Chizat 2019, Woodworth 2020, Kunin 2024, Kumar 2024 | 6 | ~15 min | grokking at full width and training length, Get-rich-quick Figure 1 teacher-student (~1 GB10 h) |
| A6 | `a6-function-space` Function-space view and finetuning dynamics | Area 6 | eNTK of a finite net; the one-step decomposition $\Delta\log\pi\approx-\eta\,\mathcal A\mathcal K\mathcal G$ derived, with the anchor as the scalar squared-loss special case ($\mathcal A=1$, $\mathcal K$ the anchor kernel, $\mathcal G=r$); softmax Jacobian; the squeezing formula $\Delta\log p_j\approx\eta(p_j+p_k-\sum_ip_i^2)$ derived and simulated on a 10-class softmax (entropy collapse); MNIST MLP: eNTK between a handful of examples predicts which test predictions move after one SGD step on one training example, checked against the actual change; SFT then off-policy DPO on a tiny transformer over a synthetic vocabulary, logging chosen, rejected and argmax log-probs. | Ren-Sutherland 2025 | 6 | ~20 min | SFT then DPO on a real small LM (~2 GB10 h) |
| A7 | `a7-parameterization` Parameterization as a theory of hyperparameters | Area 7 | SP, NTK, mean-field and muP as width-scaling rules for init and learning rate; by hand, the size of the hidden-feature change after one SGD step in a width-$m$ two-layer net under SP vs muP as $m\to\infty$, and the feature-learning vs kernel dichotomy; coordinate check on an MLP for widths 64 to 2048 under both (few steps each); LR sweep at three widths on an MLP on a MNIST subset under SP and muP, where the optimum sits. | Yang-Hu 2021, Yang et al. 2022 | 5 | ~30 min | coordinate check and LR sweep on a small Transformer, widths to 2048 (~3 GB10 h) |
| C | `capstone` Predict the minimum | all | Given a chosen combination of init $(a_0,b_0)$, step size $\eta$, label-noise $\sigma^2$ and weight decay $\gamma$ (and depth 2 or 3), predict before running: converge or diverge, oscillate or not, which point on the manifold, how fast, how it drifts; then run and compare, for a fixed set of six scenarios plus three of his own. Reread Simon et al., map each assignment onto its five bodies of work and the open directions. | Simon 2026 | 3 | ~5 min | none |

Total: about 44 hours of his time, under 2 hours of CPU checks in total, about 7 GB10 hours of
optional extensions.

### Dependency graph

```
A0 ─► A1 ─► A2 ─► A3 ─► A4
            │     ▲
            └──► A5 ─► A6 ─► A7 ─► C
```

A1 first (brief). A2, A3 and A5 only need A1 and are written to be interchangeable in order; the
numbering above keeps the anchor's growth order (weight decay before depth 3 before matrices). A4 needs
A3's Hessian and stability work. A6 and A7 need A5's kernel picture. The capstone needs everything.

### Where the brief's hooks landed

Every exercise hook in `BRIEF.md` is a problem in exactly one assignment: Area 1 → A1; Area 5 → A2;
Area 3 → A3 (including the Adam/RMSProp exercise the notes ask for); Area 4 → A4 (plus the label noise
with weight decay extension from the post); Area 2 → A5; Area 6 → A6; Area 7 → A7; the capstone → C;
the two framing papers → A0 and C.

## 3. Compute tiers and shrinks

| Experiment | Tier | Core-track version | What the full version adds |
|---|---|---|---|
| Everything on the scalar or matrix anchor, diagonal linear nets, OU simulations | Light | as written | nothing |
| EoS on an MLP (A3) | Medium | 2-layer tanh MLP, width 200, 1k CIFAR-10 examples, full-batch GD, ~3000 steps, sharpness by power iteration every 20 steps; ~20 min | Cohen's 5k-example setting, deeper nets, several $\eta$ (GPU ext.) |
| Width-$m$ ReLU kernel movement (A5) | Light | $m\le 512$, 1-D or low-D data, ~2 min | teacher-student at Kunin 2024 scale (GPU ext.) |
| Grokking toy (A5) | Medium | modular addition mod 23, one-hidden-layer MLP, ~10 min | larger modulus and width (GPU ext.) |
| MNIST eNTK (A6) | Light | 2-layer MLP, 2k training examples, eNTK over 32 probes, ~3 min | nothing needed |
| SFT then DPO (A6) | Medium | 2-layer transformer, $d=64$, synthetic 32-token vocabulary with a planted preference task, ~15 min | real small LM (GPU ext.) |
| Coordinate check (A7) | Medium | MLP widths 64-2048, 3 steps each, ~5 min | Transformer (GPU ext.) |
| LR sweep × width (A7) | Medium | widths 128/256/512, 6 LRs, 1 epoch on 10k MNIST, ~25 min | more widths, Transformer (GPU ext.) |

## 4. Directory layout

```
training-dynamics/
  BRIEF.md  COURSE.md  STATE.md  README.md  pyproject.toml  uv.lock
  template/        template.typ, build.sh, handout.csl, refs.bib
  papers/          the 21 PDFs and the post
  data/            mnist.npz, cifar10_5k.npz (predownloaded)
  tdyn/            shared [AI] package: integrators, GD/SGD loops, hvp + power iteration (A0 stub
                   graduates here), tiny MLP/transformer, plotting, results.json helpers
  a0-prereqs/ … a7-parameterization/ capstone/
    handout.typ/.pdf  tests/{adapters.py,test_*.py,conftest.py}  <pkg>/  scripts/  solutions/  notes.md
```

## 5. Build status

See `STATE.md`.
