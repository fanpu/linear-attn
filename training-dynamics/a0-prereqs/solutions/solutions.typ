#import "/template/template.typ": *

#show: doc.with(
  assignment: 0,
  title: "Warm-up: the five tools",
  kind: "Worked solutions",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "reference numbers: solutions/results.json · figures: solutions/figures/ · equation numbers (n) refer to the handout",
)

Every derivation here is complete; read only as far as you need. "Measured" numbers come from running each check script with the reference implementations (`--solutions`) on CPU; your numbers should agree within the tolerances stated in each problem. Diff yours with `uv run python -m tdyn.results a0-prereqs/out/results.json a0-prereqs/solutions/results.json`. Equation numbers such as (2) refer to the handout; this document numbers its own equations as (S1), (S2), ...

#set math.equation(numbering: n => "(S" + str(n) + ")")

= Problem (#raw("flow-limit")): GD is forward Euler on gradient flow

#runin[(a)] Forward Euler with velocity field $F(theta) = -nabla L(theta)$ and time step $h$ is $theta_(k+1) = theta_k + h F(theta_k) = theta_k - h nabla L(theta_k)$, which is GD (2) with $h = eta$. Each step advances time by $eta$, so $theta_k$ approximates the flow at
$ t_k = k eta. $

#runin[(b)] Along the flow, $dot(theta) = -nabla L(theta)$ and, by the chain rule, $dot.double(theta) = -nabla^2 L(theta) dot(theta) = nabla^2 L(theta) nabla L(theta)$. Taylor's theorem around $t = 0$ gives
$ theta(eta) = theta_0 - eta nabla L(theta_0) + eta^2/2 nabla^2 L(theta_0) nabla L(theta_0) + O(eta^3), $
and one GD step is $theta_1 = theta_0 - eta nabla L(theta_0)$, so
$ theta(eta) - theta_1 = eta^2/2 thin nabla^2 L(theta_0) nabla L(theta_0) + O(eta^3). $ <s-local>
The local error is $O(eta^2)$. Reaching a fixed time $T$ takes $K = T\/eta$ steps; if the local errors add up without being amplified, the global error is $K dot O(eta^2) = O(eta)$, so $p = 1$: GD is a first-order method. (More carefully, an error made at step $j$ is carried forward by the flow and can grow by at most $e^(M (T - t_j))$ with $M$ the Lipschitz constant of $nabla L$; for a fixed $T$ this is a constant, so the order stays $1$. This is the standard Grönwall argument.)

#runin[(c) Expected.] The slopes are $2$ (one-step) and $1$ (global). The leading term @s-local even predicts the one-step error itself: at $(1.5, 0.5)$ the residual is $r = -0.25$, $nabla L = (b r, a r) = (-0.125, -0.375)$, $nabla^2 L = mat(0.25, 0.5; 0.5, 2.25)$, so $norm(nabla^2 L nabla L) = norm((-0.21875, -0.90625)) = 0.9323$ and the prediction is $0.466 thin eta^2$.

#figure(
  table(
    columns: 4,
    [$eta$], [one-step error], [$0.466 eta^2$ from @s-local], [max error on $[0, 10]$],
    [$0.1$], [$4.37 times 10^(-3)$], [$4.66 times 10^(-3)$], [$8.13 times 10^(-3)$],
    [$0.03$], [$4.12 times 10^(-4)$], [$4.20 times 10^(-4)$], [$2.25 times 10^(-3)$],
    [$0.01$], [$4.63 times 10^(-5)$], [$4.66 times 10^(-5)$], [$7.35 times 10^(-4)$],
    [$0.003$], [$4.19 times 10^(-6)$], [$4.19 times 10^(-6)$], [$2.19 times 10^(-4)$],
  ),
  caption: [Measured errors of GD against the flow on the anchor from $(1.5, 0.5)$. Fitted log-log slopes: $1.983$ (one-step) and $1.030$ (max error).],
)

#figure(image("figures/flow_limit_error.png", width: 60%), caption: [`flow_limit_error.png`: the two errors against $eta$, with reference lines of slope 1 and 2.])

#runin[(d)] At $eta = 0.9$ GD does _not_ end at the flow's minimum. The flow goes monotonically to $(1.5538, 0.6436)$, the point of $a b = 1$ with $a^2 - b^2 = 2$ (the post's formula $a_infinity^2 = (c + sqrt(c^2 + 4))\/2$ with $c = 2$). GD overshoots: $a b$ crosses $1$ at every one of the 60 steps, the imbalance $a^2 - b^2$ falls from $2$ to $0.562$, and GD settles at $(1.1488, 0.8704)$, a different and more balanced minimum. Why: the small-step limit needs $eta$ to be small _relative to the curvature_, i.e. $eta lambda_max << 1$, not small in absolute terms. Near the flow's endpoint the sharpness is $a^2 + b^2 = sqrt(c^2 + 4) = 2 sqrt(2) = 2.83$, so $eta lambda_max = 2.55 > 2$: by Problem (#raw("gd-stability")) GD cannot converge to that point at all, and it oscillates across the valley until it reaches minima flat enough to be stable (Problem (#raw("hessian")) (c)). The post's §3.1 gives the factor by which each step changes the imbalance, $1 - eta^2 r_k^2$, which is far from $1$ while the oscillation is large.

#figure(image("figures/flow_limit_eta09.png", width: 92%), caption: [`flow_limit_eta09.png`: GD at $eta = 0.9$ against the flow from $(1.5, 0.5)$.])

= Problem (#raw("gd-stability")): Stability of GD on a quadratic

#runin[(a)] $nabla L(theta) = lambda theta$, so $theta_(k+1) = (1 - eta lambda) theta_k$ and
$ theta_k = (1 - eta lambda)^k theta_0. $
With $q = 1 - eta lambda$ (and $eta lambda > 0$): (i) monotone convergence for $0 < eta lambda < 1$ ($0 < q < 1$; at $eta lambda = 1$ exactly, $q = 0$ and GD lands on the minimum in one step); (ii) convergence with sign flips for $1 < eta lambda < 2$ ($-1 < q < 0$); (iii) at $eta lambda = 2$, $q = -1$ and $theta_k = (-1)^k theta_0$ bounces forever with constant amplitude; (iv) divergence for $eta lambda > 2$ ($q < -1$), with growing sign flips.

#runin[(b)] With $H = Q Lambda Q^top$ and $z = Q^top theta$: $z_(k+1) = Q^top (theta_k - eta Q Lambda Q^top theta_k) = (I - eta Lambda) z_k$, so each coordinate obeys (a) with its own $lambda_i$: $z_(i,k) = (1 - eta lambda_i)^k z_(i,0)$. Since $Q$ is orthogonal, $norm(theta_k) = norm(z_k)$. (i) GD converges from every start iff $|1 - eta lambda_i| < 1$ for all $i$, i.e. (all $lambda_i > 0$)
$ eta < 2 / lambda_max. $
(ii) For a generic start (every $z_(i,0) != 0$) the slowest-shrinking coordinate dominates, so
$ rho(eta) = max_i |1 - eta lambda_i| = max(|1 - eta lambda_min|, |1 - eta lambda_max|). $
The first term decreases in $eta$ and the second increases once $eta > 1\/lambda_max$, so $rho$ is minimized where they cross: $1 - eta lambda_min = eta lambda_max - 1$, giving $eta^* = 2\/(lambda_min + lambda_max)$ and $rho(eta^*) = (lambda_max - lambda_min)\/(lambda_max + lambda_min) = (kappa - 1)\/(kappa + 1)$ with $kappa = lambda_max \/ lambda_min$ the _condition number_. (iii) For eigenvalues ${1, 10}$:
$ eta_c = 0.200, quad eta^* = 2/11 = 0.182, quad rho(eta^*) = 9/11 = 0.818. $
Below $eta^*$ the slow mode ($lambda = 1$) limits the rate, above it the stiff mode ($lambda = 10$) does; for $eta > 1\/lambda_max = 0.1$ the stiff mode already oscillates while GD still converges.

#runin[(c)] Reference: `solutions/gd_solution.py`, a loop that copies `theta0`, applies (2) and stores every iterate.

#runin[(d) Expected.] Measured on the grid: $eta_c = 0.2000$, $eta^* = 0.1820$, $rho(eta^*) = 0.8183$; and $hat(rho) = 0.99, 0.95, 0.90, 0.85, 0.90$ at $eta = 0.01, 0.05, 0.1, 0.15, 0.19$, exactly $max(|1 - eta|, |1 - 10 eta|)$.

#figure(image("figures/gd_stability.png", width: 92%), caption: [`gd_stability.png`. Left: measured contraction factor against the two mode factors; the measured curve is their maximum. Right: the two eigen-coordinates for $eta = 0.04$ (both monotone), $0.17$ (stiff mode oscillates and shrinks by $0.7$ per step) and $0.21$ (stiff mode grows by $1.1$ per step).])

= Problem (#raw("phase-portrait")): The anchor model's phase portrait

#runin[(a)] With $r = a b - 1$: $partial_a L = b r$, $partial_b L = a r$, so
$ dot(a) = -b(a b - 1), quad dot(b) = -a(a b - 1). $
A fixed point needs $b r = 0$ and $a r = 0$. If $r = 0$ both hold: every point of the hyperbola $a b = 1$ is a fixed point, with loss $0$. If $r != 0$ then $a = b = 0$: the origin, with loss $1\/2$. These are all.

#runin[(b)] Writing $F = (-a b^2 + b, -a^2 b + a)$,
$ J(a, b) = mat(-b^2, 1 - 2 a b; 1 - 2 a b, -a^2) = -nabla^2 L(a, b). $
_Origin:_ $J = mat(0, 1; 1, 0)$, eigenvalues $plus.minus 1$ with eigenvectors $(1, 1)\/sqrt(2)$ (eigenvalue $+1$: trajectories leave along $a = b$) and $(1, -1)\/sqrt(2)$ (eigenvalue $-1$: they arrive along $a = -b$). A saddle.
_On $a b = 1$:_ at $(a, 1\/a)$, $J = -mat(1\/a^2, 1; 1, a^2)$, with $det J = 1 - 1 = 0$ and $"tr" J = -(a^2 + 1\/a^2)$, so the eigenvalues are
$ mu_1 = 0, quad mu_2 = -(a^2 + b^2) = -(a^2 + 1/a^2). $
The eigenvector of $mu_1 = 0$ is $(1, -1\/a^2)$, which is $d\/d a thin (a, 1\/a)$: the tangent to the curve $a b = 1$. The eigenvector of $mu_2$ is $(b, a) = nabla(a b)$, the normal. So a perturbation off the curve decays at rate $a^2 + b^2$, and a perturbation along it neither grows nor decays, because it lands on another fixed point. By @tab-fp-ref the point is degenerate; the minima form a _line of neutrally stable fixed points_, and which one training reaches is decided by something other than the loss (here the conserved $c$; this is the post's implicit-bias question).

#figure(kind: table, supplement: [Table], numbering: _ => [S1],
  table(columns: 3, [Point], [Eigenvalues of $J$], [Label],
    [$(0, 0)$], [$-1, +1$], [saddle],
    [$(a, 1\/a)$], [$0, -(a^2 + 1\/a^2)$], [degenerate],
  ), caption: [Linearization of the anchor's gradient flow.]) <tab-fp-ref>

#runin[(c)] $d\/d t (a^2 - b^2) = 2 a dot(a) - 2 b dot(b) = -2 a b r + 2 a b r = 0$. The origin has $c = 0$, so only initializations with $a_0^2 = b_0^2$, i.e. $a_0 = plus.minus b_0$, can reach it. On $a = b = s$ the flow is $dot(s) = s(1 - s^2)$, which sends every $s != 0$ to $plus.minus 1$. On $a = -b$, write $a = s$, $b = -s$: $dot(a) = -b(a b - 1) = s(-s^2 - 1)$, so
$ dot(s) = -s (1 + s^2), $
which decays to $0$ from every start. So exactly the initializations on the line $a + b = 0$ end at the origin (the saddle's stable direction from (b)). A one-line proof of the colouring in the check: $d\/d t (a + b) = -(a + b)(a b - 1)$, so $(a + b)(t) = (a_0 + b_0) exp(-integral_0^t r)$ never changes sign. Starts with $a_0 + b_0 > 0$ end on the branch $a, b > 0$, those with $a_0 + b_0 < 0$ on $a, b < 0$.

#runin[(d)] Reference: `solutions/fixed_points_solution.py`. It computes the eigenvalues with `np.linalg.eigvals`, returns `"degenerate"` if any has modulus below the tolerance, then separates complex pairs (spiral or center) from real pairs (node or saddle). The trace–determinant test of Example (#raw("classify-by-hand")) is an equivalent route.

#runin[(e) Expected.] Finite-difference eigenvalues: $(-1, 1)$ at the origin; $(-4.25, 0)$ at $a = 0.5$ and $a = 2$; $(-2, 0)$ at $a = plus.minus 1$, matching $-(a^2 + 1\/a^2)$ to $10^(-10)$. Labels: saddle, then degenerate four times. Of 156 initializations, $100%$ reach the branch given by the sign of $a_0 + b_0$; final loss below $10^(-27)$; largest drift of $c$ $5.6 times 10^(-8)$ (the RK4 error at step $0.005$); a start on $a = -b$ ends $1.8 times 10^(-9)$ from the origin.

#figure(image("figures/phase_portrait.png", width: 92%), caption: [`phase_portrait.png`, the post's picture. Left: streamlines of $-nabla L$, the conserved curves $a^2 - b^2 = c$ (dashed; the streamlines lie on them), $a b = 1$ (black), the separatrix $a = -b$ (red). Right: trajectories from a $13 times 13$ grid, blue if they reach $a, b > 0$, orange if $a, b < 0$, on the predicted regions $a + b > 0$ and $a + b < 0$.])

= Problem (#raw("hessian")): Sharpness and trace from Hessian-vector products

#runin[(a)] Reference: `solutions/hessian_solution.py`. `hvp` makes `params` a fresh leaf, takes `g = grad(loss, p, create_graph=True)`, then `grad(g @ v, p)`. `power_iteration` draws $v_0$ from `torch.Generator().manual_seed(seed)`, normalizes, applies (5) `iters` times and returns the Rayleigh quotient of the last vector. `hutchinson_trace` averages $z^top H z$ over Rademacher $z$ (`randint(0, 2) * 2 - 1`). Two remarks. With Rademacher probes $z^top H z = sum_i H_(i i) + sum_(i != j) z_i z_j H_(i j)$, so the variance is $2 sum_(i != j) H_(i j)^2$ and a diagonal $H$ is estimated exactly from one probe (that is what `test_hutchinson_trace_diagonal_is_exact` checks); Gaussian probes add a $2 sum_i H_(i i)^2$ term. Power iteration's rate is $|lambda_2\/lambda_1|$ per step: $0.74$ for the test MLP ($2.457\/3.319$), so 100 steps reach machine precision.

#runin[(b)] From Problem (#raw("phase-portrait")) (b), $nabla^2 L = -J$ has eigenvalues $0$ and $a^2 + b^2$ on the manifold, so
$ lambda_max (nabla^2 L) = a^2 + 1/a^2, quad "tr" nabla^2 L = a^2 + 1/a^2. $
Only the nonzero eigenvalue matters for linear stability: the zero direction is neutral to first order, since neighbouring points of the manifold are also minima. Stability needs $a^2 + 1\/a^2 < 2\/eta$. With $u = a^2$ this is $u^2 - (2\/eta) u + 1 < 0$, i.e. $u in (1\/eta - sqrt(1\/eta^2 - 1), 1\/eta + sqrt(1\/eta^2 - 1))$.
For $eta = 0.5$: $a^2 in (2 - sqrt(3), 2 + sqrt(3))$, so
$ a in (0.5176, 1.9319). $
For $eta = 1$: $a^2 + 1\/a^2 >= 2$ with equality only at $a = 1$, so no minimum is strictly stable; $(1, 1)$ is marginal ($eta lambda_max = 2$, case (iii) of Problem (#raw("gd-stability"))). The largest step size with a stable minimum is therefore $eta < 1$, approached by the balanced minimum, which A3 develops.

#runin[(c) Expected.] Sharpness along $a b = 1$ matches $a^2 + 1\/a^2$ to $4 times 10^(-16)$ relative; the trace estimates are within $0.010$ (400 probes: $z^top H z = a^2 + b^2 plus.minus 2$, so the standard error is $0.1$). Measured stable set for $eta = 0.5$ on the grid: $a in [0.527, 1.899]$, inside $(0.518, 1.932)$ up to the $4.7%$ grid spacing. MLP (193 parameters): power iteration $3.319157$ against the exact $3.319157$; Hutchinson with 200 probes $10.12$ against the exact trace $9.74$ (standard error $0.55$).
The two stars: the flow's endpoint $(1.554, 0.644)$ has sharpness $2 sqrt(2) = 2.828 > 2\/0.9 = 2.222$, so it is linearly _unstable_ for GD at $eta = 0.9$, and GD could never settle there. The GD endpoint $(1.149, 0.870)$ has sharpness $2.077 < 2.222$, inside the stable set $a in (0.792, 1.263)$ for $eta = 0.9$. GD at a large step size can only stop at minima flatter than $2\/eta$: a finite step size biases GD toward flat, balanced minima. This is the picture A3 starts from.

#figure(image("figures/sharpness.png", width: 70%), caption: [`sharpness.png`: sharpness (power iteration) and trace (Hutchinson) along $a b = 1$, the thresholds $2\/eta$, the stable band for $eta = 0.5$ (shaded) and the two endpoints of Problem (#raw("flow-limit")).])

= Problem (#raw("modes")): Gradient flow of linear regression, mode by mode

#runin[(a)] $L(W) = 1/2 EE[(W x - y)^top (W x - y)]$. Perturbing $W -> W + Delta$ changes $L$ by $EE[(W x - y)^top Delta x] = "tr"(Delta^top EE[(W x - y) x^top])$ to first order, so
$ nabla_W L = EE[(W x - y) x^top] = W Sigma_(x x) - Sigma_(y x), $
and gradient flow is $dot(W) = Sigma_(y x) - W Sigma_(x x)$, equation (8). Setting the gradient to zero: $W^star = Sigma_(y x) Sigma_(x x)^(-1)$, the least-squares solution.

#runin[(b)] Let $E = W - W^star$. Since $W^star Sigma_(x x) = Sigma_(y x)$, $dot(E) = -E Sigma_(x x)$. Rotate: $tilde(E) = E U$ with $Sigma_(x x) = U Lambda U^top$; then $dot(tilde(E)) = -E U Lambda U^top U = -tilde(E) Lambda$. Column $i$ of $tilde(E)$, the error of mode $i$, obeys $dot(tilde(e))_i = -lambda_i tilde(e)_i$ by itself, so $tilde(e)_i (t) = e^(-lambda_i t) tilde(e)_i (0)$. Rotating back,
$ W(t) = W^star + (W_0 - W^star) U e^(-Lambda t) U^top = W^star + (W_0 - W^star) e^(-Sigma_(x x) t). $
Mode $i$ shrinks by a factor $e$ in time $1\/lambda_i$, so the smallest eigenvalue of the input covariance sets the training time, and an ill-conditioned $Sigma_(x x)$ means slow learning (the continuous-time face of Problem (#raw("gd-stability"))'s condition number).

#runin[(c)] Reference: `solutions/modes_solution.py`: `eigh` of $Sigma_(x x)$, $W^star$ by a linear solve, then the formula above.

#runin[(d) Expected.] Fitted decay rates $0.1000, 0.3162, 1.0000, 3.1623, 10.000$, equal to the eigenvalues to a relative $10^(-7)$; closed form within $2 times 10^(-8)$ of RK4 (step $0.005$).

#figure(image("figures/modes.png", width: 62%), caption: [`modes.png`: per-mode error $norm((W(t) - W^star) u_i)$; each is a straight line of slope $-lambda_i$ on the log scale.])

#runin[(e)] With $Sigma_(x x) = I$ and $W_0 = 0$: $W^star = Sigma_(y x)$ and $W(t) = (1 - e^(-t)) Sigma_(y x)$. If $Sigma_(y x) = U S V^top$, then $W(t) = U [(1 - e^(-t)) S] V^top$: the singular vectors are those of $Sigma_(y x)$ from the first instant, and every singular value grows as $s_alpha (1 - e^(-t))$, all at the same rate, with no plateau. Saxe et al. show that with two layers each singular value instead follows an S-shaped curve whose timing scales like $1\/s_alpha$, so strong modes are learned first (A1).

= Problem (#raw("ou")): The Ornstein–Uhlenbeck process

#runin[(a)] Taking expectations in (10) with $mu(X) = -theta X$, and using $EE[xi_k] = 0$: $m_(k+1) = (1 - theta Delta t) m_k$, so $(m_(k+1) - m_k)\/Delta t = -theta m_k$, which tends to $dot(m) = -theta m$. Hence
$ m(t) = x_0 e^(-theta t). $
(Directly: $m_k = x_0 (1 - theta Delta t)^k -> x_0 e^(-theta t)$ with $k Delta t = t$.)

#runin[(b)] Subtract the mean recursion: $X_(k+1) - m_(k+1) = (1 - theta Delta t)(X_k - m_k) + sigma sqrt(Delta t) xi_k$, and $xi_k$ is independent of $X_k$, so variances add:
$ v_(k+1) = (1 - theta Delta t)^2 v_k + sigma^2 Delta t. $ <s-vrec>
Expanding, $(v_(k+1) - v_k)\/Delta t = -2 theta v_k + sigma^2 + theta^2 Delta t thin v_k$; as $Delta t -> 0$, $dot(v) = -2 theta v + sigma^2$ with $v(0) = 0$, so
$ v(t) = sigma^2 / (2 theta) (1 - e^(-2 theta t)), quad v_infinity = sigma^2 / (2 theta). $
The noise term is $sigma^2 Delta t$, first order in $Delta t$, only because the increment is scaled by $sqrt(Delta t)$; with $Delta t$ instead, the noise would vanish in the limit.

#runin[(c)] Reference: `solutions/sde_solution.py`, a loop over steps, vectorized over paths, with `rng.standard_normal(n_paths)` per step.

#runin[(d) Expected.] At $t = 1$: sample mean $0.7396$ (formula $0.7358$; standard error $0.0046$), variance $0.1096$ (formula $0.1081$). At $t = 8$: variance $0.1290$ against $v_infinity = 0.125$, within $1.6$ standard errors ($0.0025$) plus the Euler–Maruyama bias $+0.5%$ from (e).

#runin[(e)] Prediction to compare with: the stationary variance grows with $Delta t$ and blows up as $theta Delta t -> 2$. The fixed point of the exact recursion @s-vrec is
$ v_infinity^"EM" = (sigma^2 Delta t) / (1 - (1 - theta Delta t)^2) = sigma^2 / (theta (2 - theta Delta t)), $
which equals $sigma^2\/(2 theta)$ as $Delta t -> 0$ and diverges at $theta Delta t = 2$. For $theta Delta t > 2$ there is no stationary distribution at all: the mean recursion has $|1 - theta Delta t| > 1$. Measured with $theta = 1$, $sigma = 0.5$: $0.123, 0.132, 0.146, 0.178, 0.249, 0.416, 0.833, 2.498$ at $theta Delta t = 0.01, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9$, against the formula's $0.126, 0.132, 0.147, 0.179, 0.250, 0.417, 0.833, 2.500$.

#figure(image("figures/ou.png", width: 100%), caption: [`ou.png`. Left: sample mean and variance against the formulas. Middle: the paths at $t = 8$ against $cal(N)(0, v_infinity)$. Right: Euler–Maruyama's stationary variance against $theta Delta t$, with the continuous-time value and the exact discrete-time formula.])

#runin[(f)] Let $X_k = theta_k - theta^*$. SGD is $X_(k+1) = X_k - eta (lambda X_k + xi_k) = X_k - lambda X_k eta - eta xi_k$. Compare with (10) for $mu(X) = -theta_"OU" X$: the drift matches with $theta_"OU" = lambda$ and $Delta t = eta$, and the noise matches when $sigma sqrt(Delta t) thin cal(N)(0, 1)$ has the law of $-eta xi_k$, i.e. $sigma sqrt(eta) = eta s$, so $sigma = s sqrt(eta)$ (if $xi_k$ is Gaussian; otherwise the match is in mean and variance). Plugging into (e):
$ "Var"[theta_k - theta^*] -> (eta s^2) / (lambda (2 - eta lambda)) approx (eta s^2) / (2 lambda) quad (eta lambda << 1). $
The spread grows linearly with the step size and the gradient-noise variance and shrinks with the curvature: sharp directions hold the iterate close, flat directions let it wander. (The expected excess loss $lambda\/2 dot "Var" approx eta s^2\/4$ does not depend on $lambda$ at all.) The divergence at $eta lambda = 2$ is the GD stability threshold of Problem (#raw("gd-stability")): noise does not move the threshold, it only makes the approach to it visible as a growing spread. A4 runs this on the anchor, where the minimum is a manifold and the curvature $a^2 + b^2$ changes along it, which turns the fluctuation into a drift.

= Problem (#raw("framing")): reading notes

There is no single right answer; here is what a good answer covers, with page pointers.
(a) The anchor is a _solvable setting_: Simon et al. list deep linear networks $f(x; theta) = W_L dots W_1 x$ with $L >= 2$ as their first example of an analytically solvable setting (§2.1, eq. (1), p. 6), and the anchor is the scalar $L = 2$ case. From it the course reaches _insightful limits_ (small initialization and lazy vs rich learning, Table 1 row 2.2; A5), _simple empirical laws_ (the edge of stability, row 2.3; A3) and _theories of hyperparameters_ (step size as sharpness regularization, µP, row 2.4; A3 and A7).
(b) For the anchor under gradient flow, the conserved $c = a_0^2 - b_0^2$ is an "early signal" that predicts the outcome exactly: measured at any time, it gives the endpoint $a_infinity^2 = (c + sqrt(c^2 + 4))\/2$, the sharpness there, $sqrt(c^2 + 4)$, and hence (Problem (#raw("hessian"))) whether GD at step $eta$ can settle there. Under finite steps or noise $c$ drifts, and predicting from early training means predicting that drift (A1, A3, A4). This is the first of Biderman et al.’s capabilities, _predicting training_ (§2.4); _intervening_ would be, for example, lowering $eta$ when the measured sharpness approaches $2\/eta$.
(c) Personal. Candidates worth arguing about: whether coarse statistics will suffice for safety-relevant behaviour (Simon et al. §1.2), and whether "predict, intervene, design" is really a hierarchy rather than three partly independent skills (Biderman et al. §2.4).
