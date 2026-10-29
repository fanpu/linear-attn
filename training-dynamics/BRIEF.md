# Training Dynamics: Core Areas and Papers

Oct 29, 2026 · @Fan Pu Zeng

## Purpose

This brief lists the seven areas and roughly 20 papers that form the core of neural network training dynamics, as context for an agent building an exercise curriculum.

- **Learner.** An ML engineer and aspiring ML researcher with a strong CS undergrad level of math (calculus, linear algebra, probability, basic real analysis). Comfortable with PyTorch and production training. Not yet fluent in dynamical systems, SDEs, or kernel methods.
- **Anchor.** The running toy model from James Ang's post [Training Dynamics](https://jamesmldoesml.github.io/blogs/learning-dynamics/), a two-layer scalar linear network trained on the identity with squared loss and gradient flow. Every area below starts from a variant of that model, so exercises can grow it step by step.
- **Learner preferences for exercises.** All math in LaTeX. Define any non-ML jargon on first use. Prefer pen-and-paper derivations paired with a short numerical experiment that checks the derivation.

The anchor model and its loss:

```latex
f_{a,b}(x) = a\,b\,x, \qquad L(a,b) = \tfrac{1}{2}(ab-1)^2, \qquad \dot a = -\partial_a L,\; \dot b = -\partial_b L
```

## Prerequisites

Five tools come up in every area, so the curriculum should warm up on them first.

| Tool | What the learner must be able to do | Where it is used |
| --- | --- | --- |
| Gradient flow | Take the small step limit of GD to get an ODE, and know when the limit is a bad approximation | Every area |
| Basic ODEs and phase portraits | Find fixed points, linearize around them, classify stable vs. saddle, sketch trajectories | Areas 1, 2, 5 |
| Hessian and sharpness | Compute the Hessian of a small loss, its top eigenvalue, and its trace | Areas 3, 4 |
| SVD and eigendecomposition | Diagonalize a linear regression problem into independent modes | Areas 1, 2, 6 |
| SDE intuition | Read an Ito SDE as drift plus noise, and know the Ornstein-Uhlenbeck process (a noisy pull back toward a point) | Area 4 |

Stability of GD on a quadratic is the single most useful fact to drill. For a loss with curvature lambda, GD converges only if the step size satisfies the bound below, which is where the 2 over eta threshold in Area 3 comes from.

```latex
L(\theta)=\tfrac{\lambda}{2}\theta^2,\quad \theta_{k+1}=(1-\eta\lambda)\,\theta_k \;\Rightarrow\; \text{converges iff } \lambda < \frac{2}{\eta}
```

## Area 1. Symmetries and conservation laws

Symmetries of the parameterization create conserved quantities under gradient flow, and those pin the trajectory to a low-dimensional curve. This is the post's a squared minus b squared result, generalized.

**Concepts to master**

- Rescaling symmetry and the Noether-style argument (differentiate the symmetry at the identity to get a constraint on the gradient).
- Balancedness between layers, and why small init gives balanced layers.
- Deep linear networks decompose into independent singular modes, each behaving like the scalar toy model.
- Plateau-then-drop learning curves (sigmoidal mode learning).
- How weight decay, momentum, finite learning rates and minibatch noise break the conservation laws, and modified gradient flow as the tool to track that.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Saxe, McClelland, Ganguli, Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120) | ICLR 2014 | Linear input-output map but nonlinear weight dynamics. Exact solutions reproduce long plateaus followed by sudden error drops. |
| [Du, Hu, Lee, Algorithmic Regularization in Learning Deep Homogeneous Models](https://proceedings.neurips.cc/paper/2018/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract.html) | NeurIPS 2018 | Gradient flow keeps differences of squared layer norms fixed for linear, ReLU and Leaky ReLU nets. Small init therefore means balanced layers. |
| [Kunin et al., Neural Mechanics](https://iclr.cc/virtual/2021/poster/2715) | ICLR 2021 | Each differentiable symmetry constrains gradients and Hessians and yields a conservation law. Finite learning rates break them, and modified gradient flow gives exact integrals for the broken quantities, checked on VGG-16. |

**Key equations for exercises**

On a balanced init with target s, the product u = ab follows a logistic ODE, which is the scalar version of Saxe's mode dynamics.

```latex
a=b,\; u=ab,\; L=\tfrac12(s-u)^2 \;\Rightarrow\; \dot u=(a^2+b^2)(s-u)=2u\,(s-u)
```

For a two-layer linear net the conserved quantity becomes a matrix.

```latex
f(x)=W_2W_1x \;\Rightarrow\; \frac{d}{dt}\left(W_1W_1^\top - W_2^\top W_2\right)=0
```

Adding weight decay with strength gamma turns conservation into exponential decay of the imbalance.

```latex
\dot a=-\partial_aL-\gamma a,\;\dot b=-\partial_bL-\gamma b \;\Rightarrow\; \frac{d}{dt}(a^2-b^2)=-2\gamma\,(a^2-b^2)
```

**Exercise hooks**

- Solve the logistic ODE above and derive the time to learn a mode as a function of s and the init scale. Verify numerically.
- Prove the matrix conservation law, then check it numerically for random init.
- Measure how the imbalance drifts under finite step GD, momentum, and weight decay, and compare with the predicted formulas.

## Area 2. Implicit bias and the lazy vs. rich regimes

The init scale, and the balance between layers, decides whether a network behaves like a fixed kernel (lazy) or learns features (rich), and that decides which zero-loss solution it reaches. This answers the post's "which point on ab = 1" question.

**Concepts to master**

- Linearization of a network around its init, and the neural tangent kernel (NTK) as the matrix of gradient inner products between inputs.
- Function-space dynamics, meaning how predictions move under gradient flow, driven by the tangent kernel.
- Lazy regime (kernel stays fixed, min RKHS norm solution) vs. rich regime (kernel moves, different implicit bias such as sparsity or low rank).
- Overall init scale vs. relative scale between layers as two separate knobs.
- Grokking as a delayed lazy to rich transition.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Jacot, Gabriel, Hongler, Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) | NeurIPS 2018 | Defines the NTK. In the infinite-width limit the kernel stays fixed and training is kernel regression. Background. |
| [Chizat, Oyallon, Bach, On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956) | NeurIPS 2019 | Lazy training is caused by scaling, not width alone. Background for Woodworth. |
| [Woodworth et al., Kernel and Rich Regimes in Overparametrized Models](https://learningtheory.org/colt2020/virtual/papers/paper_196.html) | COLT 2020 | Init scale controls the transition between kernel and rich regimes and changes generalization. Fully solved for simple depth-D linear models (L2-like bias at large scale, L1-like bias at small scale). |
| [Kunin, Raventós, et al., Get rich quick](https://arxiv.org/abs/2406.06158) | NeurIPS 2024 spotlight | Exact solutions of a minimal model where layer-specific init variances and learning rates set the degree of feature learning through conserved quantities. Balanced init is needed for fast rich learning in linear nets, while upstream (earlier layer faster) unbalanced init speeds it up in nonlinear nets. Closest paper to the anchor model. |
| [Kumar, Bordelon, Gershman, Pehlevan, Grokking as the Transition from Lazy to Rich Training Dynamics](https://arxiv.org/abs/2310.06110) | ICLR 2024 | Grokking appears when the initial NTK's top eigenvectors are misaligned with the labels and the net starts lazy, so it fits a kernel solution first and learns features late. |

**Key equations for exercises**

Generic function-space dynamics under gradient flow on squared loss.

```latex
\frac{d f_t(x)}{dt} = -\frac1n\sum_{i=1}^n K_t(x,x_i)\,\big(f_t(x_i)-y_i\big),\qquad K_t(x,x')=\nabla_\theta f_t(x)^\top \nabla_\theta f_t(x')
```

For the anchor model the tangent kernel is a scalar times x times x prime, and the conserved quantity c sets how much that scalar can move. This links directly to the post's equation for the product p = ab.

```latex
K_t(x,x')=(a_t^2+b_t^2)\,x\,x',\qquad a_t^2+b_t^2=\sqrt{c^2+4p_t^2},\qquad c=a_0^2-b_0^2
```

**Exercise hooks**

- Derive the tangent kernel of the anchor model and show how far it moves during training as a function of c. Large imbalance means the kernel barely moves (lazy), balanced small init means it grows from near zero (rich).
- Reproduce Woodworth's diagonal linear network experiment on a sparse regression task, sweeping init scale, and plot test error against scale.
- Extend the anchor model to a width-m two-layer ReLU net and measure kernel movement for different overall and relative init scales, following Get rich quick.

## Area 3. Finite step sizes and the edge of stability

Real GD is not gradient flow. Small finite steps add a hidden penalty on gradient norm, and large steps push training to a regime where sharpness sits at 2 over eta and the loss oscillates. This picks up where the post's Section 3.1 stops.

**Concepts to master**

- Backward error analysis, meaning find the modified flow that the discrete iterates follow exactly, instead of measuring their error against the original flow.
- Implicit gradient regularization as the first correction term, and its bias toward flat minima.
- Linear stability of a minimum under GD, sharpness (top Hessian eigenvalue), and the 2 over eta threshold.
- Progressive sharpening and the edge of stability (EoS), where sharpness rises to 2 over eta and then hovers while the loss is non-monotone.
- Central flows, which describe the time-averaged path of an oscillating optimizer, including how RMSProp-style methods adapt.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Barrett, Dherin, Implicit Gradient Regularization](https://arxiv.org/abs/2009.11162) | ICLR 2021 | Discrete GD steps implicitly penalize large loss gradients, derived with backward error analysis. Biases toward flat minima. |
| [Smith, Dherin, Barrett, De, On the Origin of Implicit Regularization in SGD](https://arxiv.org/abs/2101.12176) | ICLR 2021 | Extends backward error analysis to SGD, accounting for minibatch ordering within an epoch. Explains why SGD can beat GD at test time. |
| [Cohen et al., Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://openreview.net/forum?id=jh-rTtvkGeM) | ICLR 2021 | The empirical discovery. Cited in the post. |
| [Cohen, Damian, et al., Understanding Optimization in Deep Learning with Central Flows](https://proceedings.iclr.cc/paper_files/paper/2025/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html) | ICLR 2025 | A differential equation for the smoothed trajectory that predicts long-run training of real nets. Explains progress despite rising loss, and how adaptive optimizers implicitly seek low-curvature regions to take larger steps. |

**Key equations for exercises**

The modified loss that GD with step eta approximately follows.

```latex
\tilde L(\theta)=L(\theta)+\frac{\eta}{4}\,\lVert\nabla L(\theta)\rVert^2
```

For the anchor model, every global minimum has one zero Hessian eigenvalue (the flat direction along ab = 1) and one equal to a squared plus b squared. So GD can only settle stably at minima with small enough a squared plus b squared, which are the near-balanced ones.

```latex
\nabla^2L\big|_{ab=1}=\begin{pmatrix} b^2 & 1\\ 1 & a^2\end{pmatrix},\quad \lambda_{\max}=a^2+b^2\ge 2,\quad \text{stable iff } a^2+b^2<\frac{2}{\eta}
```

**Exercise hooks**

- Compute the IGR term for the anchor model by hand. Then compare finite step GD with gradient flow on the original and on the modified loss.
- Use the stability condition to explain the post's divergence at eta = 1 and init (2, 2), and find the largest eta for which some minimum is stable.
- Pick an init whose gradient flow endpoint is unstable for a given eta. Simulate GD, track sharpness every step, and watch it settle at or below 2 over eta.
- Reproduce progressive sharpening and EoS on a small MLP on CIFAR-10 subset with full-batch GD, logging top Hessian eigenvalue via power iteration.

## Area 4. SGD noise and drift along the solution manifold

Once training reaches a manifold of zero-loss solutions, the gradient vanishes but noise does not, and the noise produces a slow deterministic drift along the manifold toward flatter points. Gradient flow alone stops wherever it lands. This is the mechanism behind the Blanc and Damian references in the post.

**Concepts to master**

- SDE approximation of SGD, and the Ornstein-Uhlenbeck process as the local picture near a minimum (noise kicks you off the manifold, curvature pulls you back).
- Why noise plus curvature gives a net drift along the manifold even though the noise has zero mean.
- Label noise SGD as a clean model of this, and why it is used. When every example can be fit exactly, ordinary minibatch noise is zero at the minimum, so something must keep injecting noise.
- The implicit regularizer (sum of squared model-gradient norms, or Hessian trace) and its link to flatness.
- Two timescales. A fast phase reaches the manifold, a much slower phase drifts along it.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Blanc, Gupta, Valiant, Valiant, Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process](https://proceedings.mlr.press/v125/blanc20a.html) | COLT 2020 | Near any zero-error point, label noise SGD behaves as if minimizing the sum over data points of the squared norm of the model's parameter gradient. Holds for any architecture, and drives toward simple functions. Local result. Cited in the post. |
| [Damian, Ma, Lee, Label Noise SGD Provably Prefers Flat Global Minimizers](https://papers.neurips.cc/paper_files/paper/2021/hash/e6af401c28c1790eaef7d55c92ab6ab6-Abstract.html) | NeurIPS 2021 | Label noise SGD converges to a stationary point of L plus lambda R, where R penalizes sharp minima and lambda depends on step size, noise strength and batch size. Global version of Blanc. Cited in the post. |
| [Li, Wang, Arora, What Happens after SGD Reaches Zero Loss?](https://arxiv.org/abs/2110.06914) | ICLR 2022 spotlight | General framework. An SDE on the manifold describes the limiting drift, set jointly by the loss and the noise covariance. Valid for order 1 over eta squared steps, with arbitrary noise covariance. |

**Key equations for exercises**

For the anchor model, Blanc's regularizer and the Hessian trace on the manifold both reduce to a squared plus b squared, so all three papers predict a drift toward the balanced points a = b = plus or minus 1.

```latex
R_{\text{Blanc}}(a,b)=\sum_i\lVert\nabla_{(a,b)}f(x_i)\rVert^2=(a^2+b^2)\sum_i x_i^2,\qquad \operatorname{tr}\nabla^2L\big|_{ab=1}=a^2+b^2
```

**Exercise hooks**

- Show that plain minibatch SGD on the anchor model has zero noise at any exact minimum, so it never moves from an unbalanced minimum like (2, 1/2).
- Implement label noise SGD (fresh Gaussian noise added to each target every step), start at (2, 1/2), and watch the drift toward (1, 1). Measure how the drift time scales with eta and the noise variance.
- Derive the drift on the 1-D manifold ab = 1 by hand using the OU picture (fast oscillation across the manifold, slow motion along it).

## Area 5. Weight decay and phase transitions

Weight decay turns the anchor model into a system with a phase transition. Past a critical strength the learned function collapses to zero, and with more layers zero becomes a trap even at small weight decay. This is the teaser at the end of the post.

**Concepts to master**

- Bifurcation, meaning a fixed point changes stability as a control parameter crosses a critical value (here, weight decay strength plays the role of temperature).
- Continuous vs. discontinuous transitions, and how depth changes which one you get.
- Critical slowing down, meaning convergence time diverges near the critical point because the relevant Hessian eigenvalue goes to zero.
- Weight decay forces balance between layers at the optimum.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Ziyin, Li, Meng, Exact Solutions of a Deep Linear Network](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9a940e858b17f01c402e164835140c4a-Abstract.html) | NeurIPS 2022 | Closed-form global minima of deep linear nets with weight decay and stochastic neurons. With more than one hidden layer, weight decay creates bad minima at zero, unlike the one-hidden-layer case, so standard inits can fail. |

**Key equations for exercises**

Using the weight decay convention that matches the post's critical value of one half.

```latex
L_\gamma(a,b)=\tfrac12(ab-1)^2+\gamma(a^2+b^2) \;\Rightarrow\; a^2=b^2,\quad ab=1-2\gamma \text{ for } \gamma<\tfrac12,\quad ab=0 \text{ otherwise}
```

The Hessian at the origin shows the transition. Its eigenvalues are 2 gamma plus or minus 1, so the origin is a saddle below one half and a local minimum above it.

```latex
\nabla^2L_\gamma(0,0)=\begin{pmatrix}2\gamma & -1\\ -1 & 2\gamma\end{pmatrix}
```

With three scalar layers the cross terms at the origin vanish, so the origin is a local minimum for every positive gamma.

```latex
f(x)=abc\,x \;\Rightarrow\; \nabla^2L_\gamma(0,0,0)=2\gamma I
```

**Exercise hooks**

- Derive all stationary points of the two-layer loss above and classify them. Plot ab against gamma and confirm the continuous transition at one half.
- Measure convergence time near gamma = one half under gradient flow and show it diverges (critical slowing down).
- Repeat for depth 3. Find the gamma at which a nonzero solution first exists and show that the learned product jumps rather than shrinking smoothly. Check which inits fall into the zero basin.
- Compare with the post's numerically observed oscillation and divergence regions when finite step size is added.

## Area 6. Function-space view and LLM finetuning dynamics

The same "change in parameters becomes change in predictions" picture from Area 2, applied per example and per step to softmax outputs, explains concrete LLM finetuning failures. The post uses this paper's framing as its definition of training dynamics.

**Concepts to master**

- Empirical NTK (eNTK), the tangent kernel of a finite network at its current weights, as a measure of how much an update on one example moves predictions on another.
- One-step influence of training example u on prediction o, and how it accumulates over training.
- The softmax Jacobian, which couples all classes (or tokens) together so that pushing one down moves mass to the others unevenly.
- Gradient ascent on unlikely responses (the negative term in DPO) and the squeezing effect.

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Ren, Sutherland, Learning Dynamics of LLM Finetuning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe1aa79e5eea7955f553c61a307273e-Abstract-Conference.html) | ICLR 2025 oral, outstanding paper | Decomposes each step's change in predictions into three terms and uses it to explain hallucinations that grow after SFT, repeated phrases, and the squeezing effect in off-policy DPO, where long training makes even preferred outputs less likely. Starts from an MNIST example. Cited in the post. |

**Key equations for exercises**

The one-step decomposition, schematically. A is the softmax Jacobian at the observed input, K is the eNTK between the observed and updated inputs, and G is the loss gradient with respect to the logits.

```latex
\Delta\log\pi_t(y\mid x_o) \approx -\eta\,\mathcal{A}_t(x_o)\,\mathcal{K}_t(x_o,x_u)\,\mathcal{G}_t(x_u,y_u)
```

The squeezing effect in its simplest form, with logits as the parameters (so K is the identity). Pushing down class k changes every other class j as below to first order, so classes with probability below the threshold lose mass and the most likely class gains the most.

```latex
z\leftarrow z-\eta\,\nabla_z\log p_k \;\Rightarrow\; \Delta\log p_j\approx\eta\Big(p_j+p_k-\sum_i p_i^2\Big)\quad (j\ne k)
```

**Exercise hooks**

- Show that the anchor model is the scalar, squared-loss special case of this decomposition (A is 1, K is the anchor tangent kernel, G is the residual).
- Derive the squeezing formula above by hand, then simulate repeated negative steps on a 10-class softmax and plot the entropy collapse.
- Train a small MLP on MNIST, compute the eNTK between a handful of examples, and predict which test predictions move most after a single SGD step on one training example. Check against the actual change.
- Run SFT then off-policy DPO on a tiny language model and log the log-probability of chosen, rejected, and the argmax response over time.

## Area 7. Parameterization as a theory of hyperparameters

How init scales and per-layer learning rates scale with width decides whether a wide network stays in the rich regime or collapses to a kernel. Getting this right is what lets hyperparameters tuned on a small model transfer to a large one. This is Area 2 turned into an engineering recipe for pretraining.

**Concepts to master**

- Standard, NTK, mean-field and maximal update (muP) parameterizations as different width-scaling rules for init and learning rate.
- The dichotomy that a parameterization either learns features in the infinite-width limit or follows kernel gradient descent, never both.
- Coordinate check, meaning plot the size of per-step activation updates against width to see whether features still move at large width.
- Hyperparameter transfer across width (muTransfer).

| Paper | Venue | What to take from it |
| --- | --- | --- |
| [Yang, Hu, Tensor Programs IV, Feature Learning in Infinite-Width Neural Networks](https://proceedings.mlr.press/v139/yang21c.html) | ICML 2021 | Standard and NTK parameterizations have no infinite-width limit that learns features. Proposes muP, and shows each parameterization in its family either feature learns or is a kernel method, not both. |
| [Yang et al., Tensor Programs V, Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) | NeurIPS 2021 | Under muP many optimal hyperparameters stay stable as width grows. Tune on a small proxy and transfer. Shown on Transformers and ResNets, with a PyTorch package. |

**Exercise hooks**

- Build a width-m two-layer net and compute by hand how much the hidden features move after one SGD step under standard vs. muP scaling, as m grows.
- Run the coordinate check on an MLP and a small Transformer for widths 64 to 2048 under both parameterizations.
- Sweep learning rate at several widths under standard parameterization and muP, and plot where the optimum sits.

## Framing, ordering, and notes for the curriculum agent

Open and close the curriculum with the two framing papers, and follow the dependency graph below in between.

| Paper | Venue | Role |
| --- | --- | --- |
| [Simon, Kunin, et al., There Will Be a Scientific Theory of Deep Learning](https://arxiv.org/abs/2604.21691) | arXiv 2026 | The map. Groups the field into solvable settings, tractable limits, simple empirical laws, theories of hyperparameters, and universal behaviors. Ends with open directions and advice for beginners. Read first, reread last. |
| [Biderman et al., Position, Don't Just Fix It in Post](https://proceedings.mlr.press/v306/biderman26a.html) | ICML 2026 | The motivation. Argues that understanding model behavior requires studying the training process that produces it, with an empirical, large-model emphasis. |

[embedded content: study order · 7 areas, Area 1 first]

Do Area 1 first. Areas 2, 3 and 5 can then run in parallel. Area 4 needs the Hessian and stability work from Area 3, and Areas 6 and 7 build on the kernel picture from Area 2.

**Notes for the curriculum agent**

- Give every area two tracks, a pen-and-paper derivation on the anchor model and a short numerical check of that derivation.
- Grow the anchor model in a fixed order so each exercise reuses the last one. Scalar two layers, then weight decay, then depth 3, then matrices, then a width-m ReLU layer, then a softmax output. This matches the extensions the post's tutorial proposes (ReLU, weight decay, more layers, label noise with weight decay).
- Most exercises run on a laptop. The EoS reproduction in Area 3 and the Transformer runs in Areas 6 and 7 want a single GPU.
- Theory here mostly assumes GD, SGD or gradient flow, while practice uses Adam. Add at least one RMSProp or Adam exercise in Area 3 using the central flows view.
- End with a capstone that asks the learner to predict, before running, which minimum the anchor model reaches under a chosen combination of init, step size, label noise and weight decay.

**Source post**

- [James Ang, Training Dynamics](https://jamesmldoesml.github.io/blogs/learning-dynamics/). Paper links in each table point to the pages checked for this brief.
