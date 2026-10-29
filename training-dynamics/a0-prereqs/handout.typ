#import "/template/template.typ": *

#show: doc.with(
  assignment: 0,
  title: "Warm-up: the five tools",
  kind: "Assignment handout",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "about 4 hours · CPU only · every check runs in under a minute",
)

= Overview

Every later assignment treats training as a _dynamical system_: a state (the parameters) moving under a fixed rule (the optimizer). This warm-up asks: can you turn a training rule into a differential equation and read off its fixed points, their stability, its time scales and its noise? It drills the five prerequisite tools of the course brief (gradient flow, phase portraits, Hessians, eigendecompositions, stochastic differential equations) on objects small enough to do by hand, chiefly the _anchor model_ of James Ang's post @ang2026post: a two-layer scalar linear network $f_(a,b)(x) = a b x$ trained on $y = x$ with
$ L(a, b) = 1/2 (a b - 1)^2, $ <eq-anchor>
after normalizing $1/n sum_i x_i^2 = 1$ (harmless; the post's §2). Nothing comes before A0. A1 (conservation laws) uses the flow, the phase portrait and the mode decomposition; A3 (edge of stability) starts from the $2\/eta$ threshold and sharpness; A4 (SGD noise) from the Ornstein–Uhlenbeck process.

==== What you will derive
+ GD is forward Euler on gradient flow, with one-step error $O(eta^2)$ (Problem (#raw("flow-limit")))
+ When GD on a quadratic converges, oscillates or diverges, and how fast (Problem (#raw("gd-stability")))
+ The fixed points of the anchor's gradient flow and their linear stability (Problem (#raw("phase-portrait")))
+ The closed-form gradient flow of linear regression, mode by mode (Problem (#raw("modes")))
+ The mean and variance of the Ornstein–Uhlenbeck process, in continuous and discrete time (Problem (#raw("ou")))

==== What you will implement
+ `gd` in `prereqs/gd.py` (Problem (#raw("gd-stability")))
+ `classify_fixed_point` in `prereqs/fixed_points.py` (Problem (#raw("phase-portrait")))
+ `hvp`, `power_iteration`, `hutchinson_trace` in `prereqs/hessian.py` (Problem (#raw("hessian")))
+ `mode_dynamics` in `prereqs/modes.py`, your closed form typed in (Problem (#raw("modes")))
+ `ou_simulate` in `prereqs/sde.py` (Problem (#raw("ou")))

==== What you will run
+ Six check scripts in `scripts/`, one per problem, each comparing a measurement with your derivation
+ A reading of the two framing papers, with three questions for `notes.md` (Problem (#raw("framing")))

#runin[Setup.] From the course root: `uv sync`, then `uv run pytest a0-prereqs` (all 48 tests fail with `NotImplementedError` until you fill in the stubs). Each check runs as `uv run python a0-prereqs/scripts/check_<name>.py` and writes PNGs and its section of `a0-prereqs/out/results.json`; `uv run python -m tdyn.results a0-prereqs/out/results.json a0-prereqs/solutions/results.json` shows where you differ from the reference. Worked solutions: `solutions/solutions.pdf`. The shared package `tdyn/` holds everything not tagged `[you]`, including staff versions of `gd` and the Hessian tools for later assignments; write your own here.

#runin[Notation.] Parameters $theta in RR^d$, loss $L$, step size $eta$, step index $k$, continuous time $t$. For the anchor, $theta = (a, b)$, residual $r = a b - 1$, imbalance $c = a^2 - b^2$. $nabla^2 L$ is the Hessian, $lambda_max$ its largest eigenvalue. A dot is a time derivative, $dot(theta) = d theta \/ d t$.

= Gradient flow: the small-step limit

Gradient descent (GD) updates
$ theta_(k+1) = theta_k - eta nabla L(theta_k). $ <eq-gd>
An _ordinary differential equation_ (ODE) $dot(theta) = F(theta)$ specifies a velocity at every point; its solution from $theta(0)$ is a curve $theta(t)$. _Gradient flow_ is the ODE
$ dot(theta) = - nabla L(theta). $ <eq-flow>
The simplest numerical ODE solver is _forward Euler_ with time step $h$: $theta_(k+1) = theta_k + h F(theta_k)$, i.e. follow the current velocity for time $h$. Its _local error_ is the error of one step started on the exact solution; its _global error_ is the accumulated error at a fixed final time $T$; a method has _order_ $p$ if the global error is $O(h^p)$. Ang's post (§1) derives @eq-flow as the $eta -> 0$ limit of @eq-gd @ang2026post; this problem makes the limit quantitative and finds where it fails, which matters because most theory in the course is about the flow.

#problem("flow-limit", [GD is forward Euler on gradient flow], kind: ("derive", "check", "predict"), time: "25 min", owner: "you")[
  #parts(
    [#tag("derive") Show that @eq-gd is forward Euler applied to @eq-flow, and say which time $t$ the iterate $theta_k$ approximates.
     #deliverable[the time $t_k$ as a function of $k$ and $eta$.]],
    [#tag("derive") Expand the exact flow $theta(eta)$ started at $theta_0$ to second order in $eta$, and compare it with one GD step $theta_1$.
     #deliverable[the leading term of $theta(eta) - theta_1$ as a formula in $nabla L(theta_0)$ and $nabla^2 L(theta_0)$, the power of $eta$ it carries, and (by counting steps to reach a fixed time $T$) the power $p$ in a global error of $O(eta^p)$.]
     _Hints, gentle to strong:_ (1) $theta(eta) = theta(0) + eta dot(theta)(0) + eta^2/2 dot.double(theta)(0) + O(eta^3)$. (2) For $dot.double(theta)$, differentiate $-nabla L(theta(t))$ in $t$ by the chain rule: the derivative of $nabla L$ along a curve is $nabla^2 L$ times the velocity. (3) Reaching time $T$ takes $T\/eta$ steps; assume the per-step errors add without growing.],
    [#tag("check") Run `scripts/check_flow_limit.py`. It runs GD on the anchor @eq-anchor from $(a, b) = (1.5, 0.5)$ at $eta in {0.1, 0.03, 0.01, 0.003}$, compares with an accurate solution of the flow at the same times, and fits log-log slopes of the one-step error and of the largest error on $t in [0, 10]$ against $eta$ (`flow_limit_error.png`). The two slopes should match your two powers to within $0.1$. A slope near $1$ for the one-step error would mean your expansion missed a first-order term.],
    [#tag("predict") The script also runs $eta = 0.9$ from the same start. Before opening `flow_limit_eta09.png`, write one line in `notes.md`: does GD at $eta = 0.9$ end at the same minimum as the flow? Then look at the figure and write two sentences on why the small-step picture fails here even though the loss is the same smooth function. (Problem (#raw("gd-stability")) and Problem (#raw("hessian")) give you the vocabulary; come back to this.)
     #deliverable[the prediction, written before running, and the two sentences.]],
  )
]

= Stability of gradient descent on a quadratic

The single most useful fact in this course is when GD on a quadratic converges. Near a minimum $theta^*$ any smooth loss looks like a quadratic, $L(theta) approx L(theta^*) + 1/2 (theta - theta^*)^top H (theta - theta^*)$ with $H = nabla^2 L(theta^*)$, so this is the local picture of GD near any minimum. Cohen et al. found that full-batch GD on neural networks drives $lambda_max (nabla^2 L)$ up to the threshold this problem derives and then hovers there, the _edge of stability_ @cohen2021eos[abstract]; that is A3.

#example("gd-by-hand", [GD on $L(theta) = lambda/2 theta^2$ with $eta lambda = 1.5$])[
  Take $lambda = 1$, $eta = 1.5$, $theta_0 = 1$. Then $nabla L(theta) = theta$, so $theta_(k+1) = theta_k - 1.5 theta_k = -0.5 theta_k$, and
  $theta_0 = 1$, $theta_1 = -0.5$, $theta_2 = 0.25$, $theta_3 = -0.125$, $theta_4 = 0.0625$.
  The iterate overshoots the minimum every step and lands on the other side, but each overshoot is half the last, so it converges while oscillating. (The test `test_gd_worked_example` checks exactly this sequence.)
]

A real symmetric matrix $H$ has an _eigendecomposition_ $H = Q Lambda Q^top$: $Q$ is orthogonal (its columns $q_i$ are orthonormal eigenvectors), $Lambda = "diag"(lambda_1, dots, lambda_d)$ holds the real eigenvalues. In the coordinates $z = Q^top theta$ the matrix becomes diagonal.

#problem("gd-stability", [Stability of GD on a quadratic], kind: ("derive", "implement", "check"), time: "35 min", owner: "you")[
  #parts(
    [#tag("derive") For $L(theta) = lambda/2 theta^2$ with $lambda > 0$, write $theta_k$ in closed form.
     #deliverable[the ranges of $eta lambda$ for which GD (i) converges monotonically, (ii) converges while flipping sign, (iii) neither converges nor diverges, (iv) diverges.]],
    [#tag("derive") For $L(theta) = 1/2 theta^top H theta$ with $H$ symmetric positive definite, use $z = Q^top theta$ to show that each coordinate $z_i$ follows (a) with $lambda = lambda_i$. Define the _contraction factor_ $rho(eta)$ as the factor by which $norm(theta_k)$ shrinks per step for large $k$ and a generic start.
     #deliverable[(i) the condition on $eta$ and $lambda_max$ for convergence; (ii) $rho(eta)$ in terms of the eigenvalues; (iii) for eigenvalues ${1, 10}$: the critical step size $eta_c$, the step size $eta^*$ that minimizes $rho$, and $rho(eta^*)$, to three decimals.]],
    [#tag("implement") Implement GD @eq-gd in `prereqs/gd.py`:
     #defline[`def gd(grad_fn, theta0, eta: float, steps: int) -> np.ndarray`]
     #arglist(
       ("grad_fn, theta0", [maps $theta$ of shape `(d,)` to $nabla L(theta)$; the start $theta_0$ (not modified).]),
       ("returns", [float64 array of shape `(steps+1, d)`, row $k$ is $theta_k$; no early stopping or clipping.]),
     )
     #testline("gd")],
    [#tag("check") Run `scripts/check_gd_stability.py`: GD on a 2-D quadratic with eigenvalues $1$ and $10$ in a rotated basis, $eta$ swept on a grid of spacing $0.0005$; it measures $rho$ per $eta$ and prints the measured $eta_c$, $eta^*$, $rho(eta^*)$ (`gd_stability.png`). They should equal your (b)(iii) to the grid spacing. If your $eta^*$ is off, check which eigenvalue limits the rate on each side of it.],
  )
]

= Fixed points, linearization and phase portraits

A _fixed point_ of $dot(theta) = F(theta)$ is a $theta^*$ with $F(theta^*) = 0$; started there, the system stays. Near it, write $theta = theta^* + delta$ and Taylor-expand: $dot(delta) approx J delta$ with the _Jacobian_ $J_(i j) = partial F_i \/ partial theta_j (theta^*)$. This linear ODE is solved by $delta(t) = sum_i c_i e^(mu_i t) v_i$ for eigenpairs $(mu_i, v_i)$ of $J$, so the eigenvalues decide what happens nearby (@tab-fp). A fixed point is _hyperbolic_ if no eigenvalue has zero real part; for hyperbolic points the linearization shows the true local picture (the Hartman–Grobman theorem). A _phase portrait_ draws the trajectories of a 2-D system in the plane. A _conserved quantity_ is a function $C(theta)$ constant along every trajectory, so trajectories lie on its level sets. For gradient flow $F = -nabla L$, so $J = -nabla^2 L$ is symmetric and its eigenvalues are real.

#figure(
  table(
    columns: (auto, auto, auto),
    [Eigenvalues $mu_1, mu_2$ of $J$], [Name], [Nearby trajectories],
    [real, both $< 0$], [stable node], [all approach $theta^*$],
    [real, both $> 0$], [unstable node], [all leave $theta^*$],
    [real, opposite signs], [saddle], [approach along one direction, leave along the other],
    [complex, $"Re" < 0$ / $"Re" > 0$], [stable / unstable spiral], [rotate inward / outward],
    [purely imaginary], [center], [closed orbits (for the linear system)],
    [at least one $= 0$], [degenerate], [linearization inconclusive in that direction],
  ),
  caption: [Classification of a fixed point of a 2-D system by the eigenvalues of its Jacobian.],
) <tab-fp>

#example("classify-by-hand", [Two Jacobians])[
  For a $2 times 2$ matrix the eigenvalues solve $mu^2 - tau mu + Delta = 0$ with trace $tau$ and determinant $Delta$. (i) $J = mat(1, 2; 3, 0)$: $tau = 1$, $Delta = -6 < 0$, so the eigenvalues have opposite signs ($3$ and $-2$): a saddle. (ii) $J = mat(-2, 1; -1, -2)$: $tau = -4$, $Delta = 5$, discriminant $tau^2 - 4 Delta = -4 < 0$, so $mu = -2 plus.minus i$: a stable spiral.
]

#problem("phase-portrait", [The anchor model's phase portrait], kind: ("derive", "implement", "check"), time: "45 min", owner: "you")[
  #parts(
    [#tag("derive") Write the gradient flow of @eq-anchor as two equations for $dot(a), dot(b)$ and find every fixed point.
     #deliverable[the set of fixed points, and the loss at each.]],
    [#tag("derive") Compute the Jacobian of the flow at a general $(a, b)$, then its eigenvalues and eigenvectors at the origin and at a point $(a, 1\/a)$ of the curve $a b = 1$. Classify each with @tab-fp and say in words what a zero eigenvalue means here.
     #deliverable[the eigenvalues at the origin; the eigenvalues at $(a, 1\/a)$ as functions of $a$; the direction of the zero-eigenvalue eigenvector relative to the curve $a b = 1$.]],
    [#tag("derive") Verify by differentiating that $c = a^2 - b^2$ is conserved by the flow (Ang derives this from the rescaling symmetry; that argument is A1's subject). Use it to find exactly which initializations end at the origin.
     #deliverable[that set of initializations, and the 1-D ODE the flow reduces to on it.]],
    [#tag("implement") Implement @tab-fp in `prereqs/fixed_points.py`:
     #defline[`def classify_fixed_point(J, tol: float = 1e-9) -> str`]
     #arglist(
       ("J, tol", [the real $2 times 2$ Jacobian; a real part, imaginary part or eigenvalue counts as zero if its size is at most `tol * max(1, max|J_ij|)`.]),
       ("returns", [one of `"stable node"`, `"unstable node"`, `"saddle"`, `"stable spiral"`, `"unstable spiral"`, `"center"`, `"degenerate"`.]),
     )
     #testline("classify_fixed_point")],
    [#tag("check") Run `scripts/check_phase_portrait.py`. It prints the Jacobian's eigenvalues measured by finite differences at the origin and at four points of $a b = 1$, with your labels; compare with (b) to six digits. Its figure `phase_portrait.png` reproduces the post's phase portrait: trajectories from a $13 times 13$ grid of initializations coloured by the branch of $a b = 1$ they reach, the dividing line in red, the conserved curves dashed. Expected: the branch is predicted by your (c) for $100%$ of the initializations, and $c$ drifts by less than $10^(-6)$.],
  )
]

= Hessian, sharpness and trace

The _sharpness_ of a loss at $theta$ is $lambda_max (nabla^2 L(theta))$, the largest Hessian eigenvalue, i.e. the curvature along the steepest direction; the _trace_ $"tr" nabla^2 L = sum_i lambda_i$ is the total curvature. A network with $d$ parameters has a $d times d$ Hessian, which nobody forms when $d$ is large. What is cheap is a _Hessian-vector product_ (HVP): since $nabla^2 L(theta) v = nabla_theta (nabla L(theta) dot v)$,
$ nabla^2 L(theta) v = nabla_theta [ nabla_theta L(theta)^top v ], $ <eq-hvp>
an HVP costs about two gradient evaluations by backpropagating twice (_double backprop_). Two algorithms turn HVPs into spectral information. _Power iteration_ repeats
$ v_(k+1) = (H v_k) / norm(H v_k), quad hat(lambda)_k = v_k^top H v_k, $ <eq-power>
which converges to the eigenvector of the largest-_magnitude_ eigenvalue when it is unique (here always positive), with error shrinking like $|lambda_2 \/ lambda_1|^k$. _Hutchinson's estimator_ uses random probes $z$ with $EE[z z^top] = I$:
$ "tr" H = EE[z^top H z] approx 1/S sum_(s=1)^S z_s^top H z_s, $ <eq-hutch>
here with _Rademacher_ probes, whose entries are independent and equal to $plus.minus 1$ with probability $1\/2$ each.

#problem("hessian", [Sharpness and trace from Hessian-vector products], kind: ("implement", "derive", "check"), time: "40 min", owner: "you")[
  #parts(
    [#tag("implement") Implement in `prereqs/hessian.py`, with the parameters as one flat float64 tensor `params` of shape `(d,)` and `loss_fn(params)` a scalar tensor:
     #defline[`def hvp(loss_fn, params, v) -> torch.Tensor`]
     #arglist(("returns", [$nabla^2 L("params") thin v$ by @eq-hvp, detached, shape `(d,)`, without forming the Hessian.]))
     #defline[`def power_iteration(hvp_fn, dim, iters, seed=0) -> tuple[float, torch.Tensor]`]
     #arglist(("returns", [$(hat(lambda), v)$ after `iters` steps of @eq-power from a random unit $v_0$ drawn with `torch.Generator().manual_seed(seed)`; $v$ a unit vector.]))
     #defline[`def hutchinson_trace(hvp_fn, dim, samples, seed=0) -> float`]
     #arglist(("returns", [the estimate @eq-hutch with `samples` Rademacher probes drawn from a seeded generator.]))
     #testline("hvp") Likewise `-k test_power_iteration` and `-k test_hutchinson_trace`; the references are the explicit anchor Hessian and `torch.autograd.functional.hessian` on a 193-parameter MLP. _Hint:_ `torch.autograd.grad(loss, p, create_graph=True)` returns a gradient you can differentiate again, here as the scalar `g @ v`; make `p` a fresh leaf with `params.detach().requires_grad_(True)`.],
    [#tag("derive") From your Problem (#raw("phase-portrait")) (b), write the sharpness and the trace of $nabla^2 L$ at a minimum $(a, 1\/a)$ of the anchor. Using Problem (#raw("gd-stability")), find the interval of $a > 0$ in which the minimum $(a, 1\/a)$ is linearly stable for GD with $eta = 0.5$, and the same for $eta = 1$.
     #deliverable[the two expressions; the two intervals (one of them is degenerate).]],
    [#tag("check") Run `scripts/check_sharpness.py`. With your three functions it estimates sharpness and trace at 41 points of $a b = 1$ and plots them against $a$ with the thresholds $2\/eta$ for $eta = 0.5, 0.9, 1$ (`sharpness.png`); expected: sharpness within $10^(-8)$ relative of your expression, trace within a few Hutchinson standard deviations. It also marks the two endpoints of Problem (#raw("flow-limit")) (d). This picture is where A3 starts.
     #deliverable[two sentences in `notes.md`: what the two stars say about the minimum GD at $eta = 0.9$ could and could not settle at.]],
  )
]

= Eigendecomposition and SVD: linear regression as independent modes

A linear model $hat(y) = W x$ with $W in RR^(d_"out" times d_"in")$ is trained on
$ L(W) = 1/2 EE norm(W x - y)^2, quad Sigma_(x x) = EE[x x^top], quad Sigma_(y x) = EE[y x^top], $ <eq-reg>
where the expectation is over the training set (an average over examples). $Sigma_(x x)$ is the input covariance (assume it is positive definite) and $Sigma_(y x)$ the input–output correlation. Every real matrix has a _singular value decomposition_ (SVD) $M = U S V^top$, $U, V$ orthogonal, $S$ non-negative diagonal; for symmetric positive semidefinite $M$ it is the eigendecomposition.

#problem("modes", [Gradient flow of linear regression, mode by mode], kind: ("derive", "implement", "check"), time: "30 min", owner: "you")[
  #parts(
    [#tag("derive") Show that gradient flow on @eq-reg is
     $ dot(W) = Sigma_(y x) - W Sigma_(x x). $ <eq-wflow>
     #deliverable[$nabla_W L$ and the minimizer $W^star$.]],
    [#tag("derive") Write $Sigma_(x x) = U Lambda U^top$ and rotate, $tilde(W) = W U$. Show that each column of $tilde(W) - W^star U$ (a _mode_) evolves on its own, and solve @eq-wflow.
     #deliverable[$W(t)$ in closed form in terms of $W_0$, $W^star$, $U$, $Lambda$ and $t$; the time for the error in mode $i$ to shrink by a factor $e$.]],
    [#tag("implement") Type your closed form into `prereqs/modes.py` (no numerical integration):
     #defline[`def mode_dynamics(W0, Sxx, Syx, t: float) -> np.ndarray`]
     #arglist(("returns", [$W(t)$, shape `(d_out, d_in)`, solving @eq-wflow with $W(0) = $ `W0`.]))
     #testline("mode_dynamics")],
    [#tag("check") Run `scripts/check_modes.py`: a regression with $d_"in" = 5$, $d_"out" = 3$ and eigenvalues of $Sigma_(x x)$ spread over two decades ($0.1$ to $10$). It plots the per-mode error $norm((W(t) - W^star) u_i)$ on a log scale (`modes.png`), fits each mode's decay rate, and compares your closed form with RK4 integration. Expected: straight lines whose fitted rates match your (b) to $10^(-3)$, and closed form within $10^(-6)$ of RK4. The slowest mode sets how long training takes.],
    [#tag("derive") Suppose the inputs are whitened ($Sigma_(x x) = I$) and $W_0 = 0$. Describe $W(t)$ through the SVD of $Sigma_(y x)$: what are its singular vectors and how do its singular values grow?
     #deliverable[one sentence.]],
  )
  #note[This is the one-layer case of Saxe, McClelland and Ganguli's analysis @saxe2014exact[§1, pp. 2–4]. With two layers, $hat(y) = W_2 W_1 x$, the modes become the singular modes of $Sigma_(y x)$ and each is learned on an S-shaped curve whose timing depends on its singular value; A1 derives it. Read §1 of the paper for the general case.]
]

= SDE intuition: drift plus noise, the Ornstein–Uhlenbeck process

SGD is GD plus noise; its continuous-time language is the _stochastic differential equation_ (SDE). _Brownian motion_ $W_t$ is the random path with $W_0 = 0$, continuous paths, and independent Gaussian increments $W_(t+h) - W_t tilde cal(N)(0, h)$: the variance of an increment equals its duration, so its typical size is $sqrt(h)$, not $h$. An _Itô SDE_
$ d X = mu(X) thin d t + sigma thin d W $
reads as "drift $mu$ plus noise of scale $sigma$", and operationally it is the $Delta t -> 0$ limit of the _Euler–Maruyama_ scheme
$ X_(k+1) = X_k + mu(X_k) Delta t + sigma sqrt(Delta t) thin xi_k, quad xi_k tilde cal(N)(0, 1) "independent". $ <eq-em>
The _Ornstein–Uhlenbeck_ (OU) process takes $mu(X) = -theta X$ with $theta > 0$: a pull toward $0$ of strength $theta$ (here $theta$ is a rate, not a parameter vector) against constant random kicks. It is the local picture of SGD near a minimum, which part (f) makes precise.

#problem("ou", [The Ornstein–Uhlenbeck process], kind: ("derive", "implement", "check", "predict"), time: "35 min", owner: "you")[
  #parts(
    [#tag("derive") For $d X = -theta X thin d t + sigma thin d W$, $X_0 = x_0$, take expectations in @eq-em, let $Delta t -> 0$, and solve.
     #deliverable[the mean $m(t) = EE[X_t]$.]],
    [#tag("derive") Do the same for the variance: from @eq-em get a recursion $v_(k+1) = (dots) v_k + (dots)$, keep terms of first order in $Delta t$, and solve the resulting ODE.
     #deliverable[$v(t) = "Var"[X_t]$ and the stationary variance $v_infinity = lim_(t -> infinity) v(t)$.]],
    [#tag("implement") Implement @eq-em for OU in `prereqs/sde.py`:
     #defline[`def ou_simulate(x0, theta, sigma, dt, steps, n_paths, rng) -> np.ndarray`]
     #arglist(("returns", [float64 array of shape `(steps+1, n_paths)`, row $k$ the paths at time $k Delta t$, all randomness from the numpy Generator `rng`.]))
     #testline("ou_simulate")],
    [#tag("check") Type your (a) and (b) into the `FORMULAS` block at the top of `scripts/check_ou.py` and run it: 5000 paths with $theta = 1$, $sigma = 0.5$, $x_0 = 2$, $Delta t = 0.01$. Expected: the sample mean and variance follow your curves and the histogram at $t = 8$ matches $cal(N)(0, v_infinity)$, within the sampling error $v_infinity sqrt(2\/5000) approx 0.0025$ of a variance estimate plus an $O(Delta t)$ bias (`ou.png`).],
    [#tag("predict") The script then runs Euler–Maruyama with large steps, $theta Delta t$ up to $1.9$. First write in `notes.md` what you expect its stationary variance to do as $theta Delta t$ grows toward $2$. Then solve the _exact_ recursion of (b) for its fixed point (no first-order truncation) and type it into `em_stationary_var_formula`.
     #deliverable[the prediction; the exact discrete-time stationary variance.]],
    [#tag("derive") Near a minimum $theta^*$ of a 1-D loss with curvature $lambda$, suppose SGD's gradient is $lambda (theta - theta^*) + xi$ with zero-mean noise of variance $s^2$. Show that SGD with step $eta$ is exactly @eq-em for an OU process, and identify its $theta$, $sigma$ and $Delta t$. In one `notes.md` paragraph: how far do SGD's iterates spread around the minimum, as a function of $eta$, $lambda$, $s$, and where does (e) meet Problem (#raw("gd-stability"))?
     #deliverable[the identification and the paragraph.]],
  )
  #note[On a manifold of minima the same balance (noise pushes off, curvature pulls back) also drives a slow drift _along_ it toward flatter points, shown for label-noise SGD by Blanc et al. @blanc2020ou[§1]; A4 derives it on the anchor.]
]

= Framing: what a theory of training dynamics is for

#problem("framing", [Read the two framing papers], kind: ("read",), time: "30 min", owner: "you")[
  Read Simon et al. @simon2026theory §1 (pp. 2–5), Table 1 and the opening of §2.1 (pp. 6–7): the five bodies of work (solvable settings, insightful limits, simple empirical laws, theories of hyperparameters, universal phenomena). Then read the abstract and §2.4 "Predict, Intervene, Design" (pp. 4–5) of Biderman et al. @biderman2026position, whose three capabilities form a hierarchy, each needing everything the one before it does. Both PDFs are in `papers/`. Answer briefly in `notes.md`:
  #parts(
    [Which of the five bodies of work does the anchor model belong to, and why? Name a second one the course reaches from it.],
    [Biderman et al. want to "predict outcomes from early training signals". What would that mean for the anchor model: which outcome, from which signal, at what time?],
    [One claim in either paper you do not believe yet, and what evidence would change your mind.],
  )
]

= What to write down

In `notes.md`, three to five sentences: when GD is well approximated by gradient flow and what breaks first when it is not; what the eigenvalues of a Jacobian, a Hessian and a covariance each told you today and how those three are related for gradient flow; why the noise in an SDE scales as $sqrt(Delta t)$; and one question you would bring to a tutorial.

#bibliography("/template/refs.bib", style: "/template/handout.csl", title: "References")
