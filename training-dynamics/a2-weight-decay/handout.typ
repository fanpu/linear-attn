#import "/template/template.typ": *

#show: doc.with(
  assignment: 2,
  title: "Weight decay and phase transitions",
  kind: "Assignment handout",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "about 4.5 hours · CPU only · all six checks together run in about 25 seconds",
)

// The fixed test sentence, for a problem whose tests live in one file with several adapters.
#let filetest(file, adapters) = [
  To test your implementation, run #raw("uv run pytest a2-weight-decay -k " + file) (the adapters
  #adapters.map(a => raw("run_" + a)).join(", ") in #raw("tests/adapters.py") are pre-wired to these
  signatures; edit them only if yours differ).
]

= Overview

Add weight decay to the anchor and it acquires a _phase transition_: below a critical strength the network learns a shrunken version of $y = x$, above it the learned function is exactly zero. This assignment finds every stationary point of the regularized anchor, shows that the transition at two layers is _continuous_ and that convergence slows down without bound as one approaches it, then adds a third layer and watches the transition turn _discontinuous_: the learned product jumps to zero, and zero becomes a trap that small initializations fall into even at small weight decay. Last, finite step sizes carve oscillation and divergence regions out of the $(gamma, eta)$ plane, reproducing the diagram at the end of Ang's post. A1 supplied the conserved imbalance and the logistic learning curve, both reused here; A3 (edge of stability) builds on the step-size analysis of the last problem.

==== What you will derive
+ That weight decay balances the layers at every stationary point, for scalars, matrices and depth 3 (Problem (#raw("balance")))
+ All stationary points of the two-layer loss, their Hessians and types, and the bifurcation at $gamma = 1/2$ (Problem (#raw("stationary")))
+ The relaxation time near $gamma = 1/2$ with its exact constants, and the algebraic decay at $gamma = 1/2$ (Problem (#raw("slowing")))
+ The depth-3 critical value $gamma^*$, the jump in the learned product, and when zero is the global minimum (Problem (#raw("depth3")))
+ The basin of the zero solution on the balanced ray and its size for small weight decay (Problem (#raw("basins")))
+ The step-size stability boundary in the $(gamma, eta)$ plane, and the oscillation band on the balanced line (Problem (#raw("finite-step")))

==== What you will implement
+ `stationary.py`, `slowing.py` in `wdecay/`: stationary points with types, learned product, relaxation time, time to converge (Problems (#raw("stationary")), (#raw("slowing")))
+ `depth3.py`, `basins.py`: the depth-3 stationary points, $gamma^*$, $gamma_c$, the endpoint of a balanced start (Problems (#raw("depth3")), (#raw("basins")))
+ `finite_step.py`: the largest stable step and the predicted phase of GD (Problem (#raw("finite-step")))

==== What you will run
+ Six check scripts, one per problem, each comparing a simulation with your derivation
+ A short reading of Ziyin, Li and Meng on deep linear networks with weight decay (Problem (#raw("ziyin")))

#runin[Setup.] From the course root: `uv run pytest a2-weight-decay` (all 48 tests fail with `NotImplementedError` until you fill in the stubs). Checks run as `uv run python a2-weight-decay/scripts/check_<name>.py` and write PNGs and their section of `a2-weight-decay/out/results.json`; compare with `uv run python -m tdyn.results a2-weight-decay/out/results.json a2-weight-decay/solutions/results.json`. Five checks have a `FORMULAS` block at the top where you type a derived formula. Worked solutions: `solutions/solutions.pdf`. #runin[Time box.] If a Derive part takes more than 15 minutes with all hints, read that part in `solutions/solutions.pdf`, close it, and redo the part from a blank page before moving on. The solutions are part of the method, not a failure. Rusty on the math? Parts point into the primer `m0-math-primer/handout.pdf` at the point of use, as in #primer("ODE.4", "logistic"); its first page is a one-page self-diagnostic.

#runin[Notation.] As in A0 and A1: $theta = (a, b)$, residual $r = a b - s$ with target $s > 0$ ($s = 1$ unless stated), product $p = a b$, imbalance $c = a^2 - b^2$, step size $eta$, weight decay $gamma$. At depth 3, $theta = (a, b, c)$ and $p = a b c$ (there $c$ is the third weight, not the imbalance). $u$ is the common value of the weights on a _balanced_ point $a = b (= c)$. $H = nabla^2 L_gamma$, with eigenvalues $lambda_"min" <= dots <= lambda_"max"$.

= Weight decay and bifurcations

_Weight decay_ adds $gamma norm(theta)^2$ to the loss. On the anchor,
$ L_gamma (a, b) = 1/2 (a b - 1)^2 + gamma (a^2 + b^2), $ <eq-loss>
with gradient flow
$ dot(a) = -b r - 2 gamma a, quad dot(b) = -a r - 2 gamma b, quad r = a b - 1. $ <eq-flow>
In GD the extra term multiplies the weights by $1 - 2 eta gamma$ every step, hence the name. The factor $2$ in $2 gamma$ is this course's convention; it puts the critical value at $gamma = 1/2$, as in Ang's post @ang2026post[§4].

A _stationary point_ (or critical point) is a $theta^*$ with $nabla L_gamma (theta^*) = 0$; it is a fixed point of the flow. A0 classified fixed points by the Jacobian of the flow, which here is $-H$: $theta^*$ is a strict local _minimum_ (stable) if every eigenvalue of $H(theta^*)$ is positive, a _saddle_ if there are both signs, a _maximum_ if all are negative, and _degenerate_ (the linearization says nothing) if one is zero. A _bifurcation_ is a qualitative change in the set of fixed points or in their stability as a _control parameter_ crosses a critical value; here the control parameter is $gamma$, which the post compares to temperature in a material. An _order parameter_ is a quantity that is zero on one side of the transition and nonzero on the other; ours is the learned product $p_infinity = lim_(t -> infinity) a b$. The transition is _continuous_ if $p_infinity$ goes to zero continuously at the critical value and _discontinuous_ if it jumps.

Ziyin, Li and Meng find the global minima of deep linear networks with weight decay in closed form @ziyin2022exact. Their Theorem 1 says a network with one hidden layer learns nothing (global minimum at zero) iff $norm(EE[x y])^2 <= gamma_u gamma_w$; in our convention ($2 L_gamma$ has $gamma_u = gamma_w = 2 gamma$ and $EE[x y] = s$) that is $gamma >= s\/2$. Read their §3–4 for general widths and inputs. Note their "depth" counts _hidden_ layers: our $f = a b c x$ is their $D = 2$.

#example("ridge", [One layer has no transition])[
  Take a single weight, $f = w x$, $L_gamma (w) = 1/2 (w - 1)^2 + gamma w^2$ (ridge regression). $L_gamma' (w) = (1 + 2 gamma) w - 1$, so the only stationary point is $w^* = 1\/(1 + 2 gamma)$, a minimum with Hessian $1 + 2 gamma > 0$. The learned function shrinks smoothly, never reaches zero for finite $gamma$ ($w^* = 1\/2$ at $gamma = 1\/2$, $1\/3$ at $gamma = 1$), and gradient flow converges at rate $1 + 2 gamma >= 1$ for every $gamma$: no transition and no slowing down. Ziyin et al.’s Figure 1 (right) makes the same point for a linear regressor against a ResNet. The transitions below come from the _product_ of layers.
]

#problem("balance", [Weight decay balances the layers], kind: ("derive", "check"), time: "25 min", owner: "you")[
  #parts(
    [#tag("derive") For $L_gamma = ell(a b) + gamma (a^2 + b^2)$ with any differentiable $ell$, compute $a thin partial_a L_gamma - b thin partial_b L_gamma$ (A1 Problem (#raw("noether")) did $gamma = 0$). Conclude what every stationary point satisfies, and quote $dot(c)$ along the flow from A1 Problem (#raw("broken")) (a) #primer("LA.6", "symmetries"), #primer("ODE.3", "linear decay").
     #deliverable[the identity $a thin partial_a L_gamma - b thin partial_b L_gamma = dots$ as an expression in $gamma$, $a$, $b$; the constraint on stationary points as an equation in $a$, $b$; $c(t)$.]
     #hints([The same computation as A1 Problem (#raw("noether")) (a), with the extra $2 gamma (a, b)$ in the gradient (primer §LA.6); the flow part is then a linear ODE (primer §ODE.3).], [The $ell'$ terms cancel exactly; only the weight-decay terms are left.], [First line of the solution: "With $nabla ell(a b) = ell'(a b)(b, a)$, the loss part cancels as in A1 and only weight decay is left."])],
    [#tag("derive") For a two-layer linear network $W_2 W_1$ with any loss $L(W_2 W_1)$ plus $gamma (norm(W_1)_F^2 + norm(W_2)_F^2)$, show that $D = W_1 W_1^top - W_2^top W_2$ obeys a closed linear ODE and solve it #primer("LA.5", "matrix calculus"). What does every stationary point satisfy, and what does the $(j, j)$ entry of that say about hidden neuron $j$? (This is Ziyin et al.’s Lemma 1, §3.1, from the flow instead of from optimality.)
     #deliverable[$D(t)$ (type it into `imbalance_at` in `scripts/check_balance.py`); the per-neuron statement, as an equation between the norm of row $j$ of $W_1$ and the norm of column $j$ of $W_2$.]
     #hints([Product rule on $D$ with the matrix gradients plus their weight-decay terms (primer §LA.5); you should get a linear matrix ODE $dot(D) = k D$ (primer §ODE.3).], [A1 Problem (#raw("matrix-law")) (b) showed that the $gamma$-free part of the gradient leaves $D$ fixed.], [First line of the solution: "With $G = nabla_W L$ at $W = W_2 W_1$, the gradients are $nabla_(W_1) = W_2^top G + 2 gamma W_1$ and $nabla_(W_2) = G W_1^top + 2 gamma W_2$."])],
    [#tag("derive") Repeat (a) for $f = a b c x$ with $gamma (a^2 + b^2 + c^2)$: which differences of squares decay, and what holds at a stationary point? #primer("LA.6", "rescaling two weights")
     #deliverable[the decay law for $a^2 - b^2$ and $b^2 - c^2$, as formulas in $t$; the stationary constraint as equations in $a$, $b$, $c$.]
     #hints([Rescale two weights and leave the third alone: the same identity holds for each pair (primer §LA.6).], [$partial_a L_gamma = b c thin ell' + 2 gamma a$; multiply by $a$ and compare with $b thin partial_b L_gamma$.], [First line of the solution: "For $f = a b c x$, $partial_a L_gamma = b c thin ell' + 2 gamma a$ and so on."])],
    [#tag("check") Run `scripts/check_balance.py`. Expected: the matrix network's $D(t)$ and the two depth-3 differences on your formula to $10^(-9)$ relative; flow endpoints from 200 random starts with $abs(abs(a_infinity) - abs(b_infinity)) = O(1)$ at $gamma = 0$ (anywhere on $a b = 1$, A1) and at round-off at $gamma = 0.1$; the per-neuron gap of the matrix network at round-off (`balance.png`).],
  )
]

= Two layers: a continuous transition

#problem("stationary", [Stationary points and the bifurcation at one half], kind: ("derive", "implement", "check"), time: "50 min", owner: "you")[
  #parts(
    [#tag("derive") With Problem (#raw("balance")), find _all_ stationary points of @eq-loss for $gamma > 0$. Then compute the Hessian $H(a, b)$ of @eq-loss, and its eigenvalues and eigenvectors at the origin and at each nonzero stationary point (simplify with what you know there). Classify every point for $gamma < 1/2$, $gamma = 1/2$ and $gamma > 1/2$, and say what happens to the nonzero points as $gamma arrow.t 1/2$ #primer("LA.1", "2×2 eigenvalues"), #primer("LA.2", "classifying stationary points"), #primer("FP.5", "bifurcations").
     #deliverable[the stationary points as functions of $gamma$; the learned product $p_infinity (gamma)$ from a small balanced start; both eigenvalue pairs (type them into the `FORMULAS` block of `scripts/check_bifurcation.py`); the classification.]
     #hints([Only $a = b$ and $a = -b$ survive (a); the Hessians are of the pattern $mat(p, q; q, p)$ (primer §LA.1).], [On $a = -b$, $r = -a^2 - 1 < 0$; look at the sign of $partial_a L_gamma$.], [At a nonzero point, $a^2 = b^2 = 1 - 2 gamma$ makes the diagonal of $H$ the same constant.])],
    [#tag("implement") In `wdecay/stationary.py`:
     #defline[`def hessian_wd(theta, gamma)`  ·  `def stationary_points(gamma)`  ·  `def learned_product(gamma)`]
     #arglist(
       ("hessian_wd", [$H(theta)$ of @eq-loss for points of shape `(..., 2)`, in closed form.]),
       ("stationary_points", [a list of `(point, kind)`, `kind` in `"minimum"`, `"saddle"`, `"maximum"`, `"degenerate"`.]),
       ("learned_product", [$p_infinity (gamma)$ from (a), broadcasting over arrays.]),
     )
     #filetest("test_stationary", ("hessian_wd", "stationary_points", "learned_product"))],
    [#tag("check") Run `scripts/check_bifurcation.py`. It integrates @eq-flow from $(0.1, 0.1)$ to $t = 3000$ for 79 values of $gamma$. Expected: $a b$ on your $p_infinity (gamma)$ to $10^(-7)$, a kink (not a jump) at $gamma = 1/2$ with $p_infinity$ vanishing linearly (log-log slope $1$), against the ridge curve of the example; autograd Hessian eigenvalues equal to yours to round-off (`bifurcation.png`).],
  )
]

== Critical slowing down

Near a stable fixed point $theta^*$, the flow is linear to first order (A0): $theta(t) - theta^* approx sum_i c_i e^(-lambda_i t) bold(v)_i$ with $(lambda_i, bold(v)_i)$ the eigenpairs of $H(theta^*)$. At late times the slowest mode wins, so the distance to $theta^*$ shrinks by a factor $e$ every
$ tau = 1 / lambda_"min" (H(theta^*)), $ <eq-tau>
the _relaxation time_. _Critical slowing down_ is the divergence of $tau$ at a bifurcation, where some $lambda_i -> 0$. Ang's post notes it in the margin of its phase diagram @ang2026post[§4]. Define also the _time to converge_ from a start $theta_0$,
$ t_epsilon = min{t : norm(theta(t) - theta^*) <= epsilon}. $ <eq-teps>

#problem("slowing", [Critical slowing down], kind: ("derive", "implement", "check"), time: "45 min", owner: "you")[
  #parts(
    [#tag("derive") Show that on the line $a = b = u$ the product $p = u^2$ obeys A1’s logistic equation $dot(p) = 2 p (s_"eff" - p)$ with an effective target $s_"eff"$. Find $s_"eff" (gamma)$; it is negative for $gamma > 1/2$ #primer("ODE.4", "logistic").
     #deliverable[$s_"eff" (gamma)$ as a formula in $gamma$.]
     #hints([Restrict the flow to $a = b = u$ and compute $dot(p) = 2 u dot(u)$ (primer §ODE.4 for the logistic form).], [Compare $-u(u^2 - 1 + 2 gamma)$ with the right side of the logistic equation.], [First line of the solution: "On $a = b = u$, @eq-flow gives $dot(u) = -u (u^2 - 1 + 2 gamma)$."])],
    [#tag("derive") From Problem (#raw("stationary")), give $tau(gamma)$ by @eq-tau for every $gamma > 0$, $gamma != 1/2$. Near the critical point $tau approx C_minus \/ (1/2 - gamma)$ below and $C_plus \/ (gamma - 1/2)$ above: find $C_minus$, $C_plus$ and their ratio. Where else does $tau$ diverge, and which direction is slow there? #primer("FP.4", "relaxation time"), #primer("ALG.5", "scaling")
     #deliverable[$tau(gamma)$, piecewise in $gamma$; $C_minus$, $C_plus$ and their ratio as numbers; the other divergence (a value of $gamma$) and its slow direction (a vector).]
     #hints([$tau = 1 \/ lambda_"min"$ of $H$ at the point the flow converges to (primer §FP.4); use your eigenvalue pairs from Problem (#raw("stationary")).], [Below $1\/2$ the minimum's two eigenvalues cross at some $gamma$: find it. Above $1\/2$ use the origin. Then read off $C_(minus), C_(plus)$ as limits of $tau |gamma - 1/2|$ (primer §ALG.5).], [First line of the solution: "The flow converges to the minimum for $gamma < 1/2$ and to the origin for $gamma > 1/2$, so with (stationary) (a): ..."])],
    [#tag("derive") At $gamma = 1/2$ exactly, solve for $p(t)$ from a balanced start $p_0$. How does $norm(theta(t))$ decay? #primer("ODE.2", "separable equations")
     #deliverable[$p(t)$ (type it into `critical_product` in `scripts/check_slowing.py`); the power of $t$ in $norm(theta(t)) prop t^k$ at late times.]
     #hints([Put $gamma = 1/2$ into (a): the logistic equation with target zero, which is separable (primer §ODE.2).], [$integral d p \/ p^2 = -1\/p$; on the balanced line $norm(theta)^2 = a^2 + b^2 = 2 p$.], [First line of the solution: "At $gamma = 1/2$, $dot(p) = -2 p^2$, so $1\/p = 1\/p_0 + 2 t$."])],
    [#tag("derive") Using A1’s solution of the logistic equation with target $s_"eff"$, find $t_epsilon$ of @eq-teps in closed form for a balanced start $a_0 = b_0 = u_0 > 0$. For fixed $epsilon$ and $u_0$, is the log-log slope of $t_epsilon$ against $|gamma - 1/2|$ exactly $-1$? Why or why not? #primer("ODE.4", "time to within ε"), #primer("ODE.7", "time to converge")
     #deliverable[$t_epsilon (gamma, u_0, epsilon)$, valid on both sides and at $gamma = 1/2$; yes or no on the slope, and the factor in $t_epsilon approx tau dot (dots)$ that makes the difference.]
     #hints([$norm(theta - theta^*) = sqrt(2) |u - u^*|$ on the line (time between two levels of a logistic: primer §ODE.4).], [Solve for the $p_epsilon$ at which the distance equals $epsilon$, then invert A1’s $u(t)$; the formula $t = 1/(2 s) ln (p (s - p_0))/(p_0 (s - p))$ holds for negative $s$ too.], [$t_epsilon approx tau ln(K\/epsilon)$ where $K$ depends on $gamma$ (primer §ODE.7, §ALG.4).])],
    [#tag("implement") In `wdecay/slowing.py`:
     #defline[`def relaxation_time(gamma)`  ·  `def time_to_within(gamma, u0, eps)`]
     #arglist(
       ("relaxation_time", [$tau(gamma)$ from (b); `np.inf` at $gamma = 1/2$; broadcasts.]),
       ("time_to_within", [$t_epsilon$ from (d), scalars; `0.0` if the start is already within $epsilon$.]),
     )
     #filetest("test_slowing", ("relaxation_time", "time_to_within"))],
    [#tag("check") Run `scripts/check_slowing.py`. For 9 values of $|gamma - 1/2|$ from $10^(-3)$ to $10^(-1)$ on each side it measures $tau$ as the time to go from distance $10^(-4)$ to $10^(-8)$, over $ln 10^4$. Expected: your $tau$ within $0.1%$, log-log slopes $-1.00$ on both sides, $tau |gamma - 1/2|$ equal to your $C_minus$, $C_plus$; $t_epsilon$ on your formula to $10^(-6)$ relative, with slopes a little shallower than $-1$; at $gamma = 1/2$ a straight line on log-log axes with your exponent (`slowing.png`).],
  )
]

= Depth 3: a discontinuous transition

Add a third scalar layer, $f = a b c x$:
$ L_gamma (a, b, c) = 1/2 (a b c - s)^2 + gamma (a^2 + b^2 + c^2). $ <eq-loss3>
Ziyin et al. prove that with two or more hidden layers the origin is always a local minimum (their Proposition 2, §4.1), that a nonzero solution exists only below a critical weight decay that depends on the architecture and the data (Proposition 3), and that common initializations can start inside the zero solution's basin (§5, Figure 3). The anchor gives all of this in closed form.

#problem("depth3", [The jump at depth 3], kind: ("derive", "implement", "check", "predict"), time: "55 min", owner: "you")[
  #parts(
    [#tag("derive") Compute $nabla^2 L_gamma$ of @eq-loss3 at the origin. Why do the cross terms that made the two-layer origin a saddle vanish here? Expand $1/2 (a b c - s)^2$ and count degrees #primer("TAY.1", "the Hessian sees the quadratic part").
     #deliverable[the Hessian at the origin as a $3 times 3$ matrix; its type (minimum, saddle, maximum or degenerate) for every $gamma > 0$.]
     #hints([The Hessian at a point sees only the quadratic part of the Taylor expansion there (primer §TAY.1, §LA.2).], [List the degree of each term of $1/2 (a b c - s)^2$ in the weights.], [First line of the solution: "$1/2 (a b c - s)^2 = 1/2 s^2 - s thin a b c + 1/2 a^2 b^2 c^2$."])],
    [#tag("derive") Using Problem (#raw("balance")) (c), show that every nonzero stationary point has $|a| = |b| = |c| = u > 0$ and $a b c > 0$, and that $u$ solves a quartic which you can write as $gamma = g(u)$. Sketch $g$ on $u > 0$; find $gamma^*(s) = max_u g(u)$, the maximizer $u^*$, and the product at the maximum. Evaluate $gamma^*(1)$ to four digits #primer("ALG.2", "critical parameters").
     #deliverable[the quartic; $gamma^*(s)$ in closed form; $p$ just below $gamma^*$ (type it into `product_at_jump` in `scripts/check_depth3.py`).]
     #hints([Multiply the three components of $nabla L_gamma = 0$ by $a$, $b$, $c$; then write the condition as $gamma = g(u)$ and maximize (primer §ALG.2).], [$g'(u) = (s - 4 u^3)\/2$.])],
    [#tag("derive") At a balanced point $(u, u, u)$, $H$ has the form $alpha I + beta (bold(1) bold(1)^top - I)$. Find its eigenvalues (use the quartic to eliminate $gamma$) and say which of the two roots is a minimum and which a saddle #primer("LA.1", "constant diagonal, constant off-diagonal").
     #deliverable[the eigenvalue along $(1, 1, 1)$ and the one across it, as formulas in $u$ and $s$; which root (smaller or larger) is the minimum and which the saddle.]
     #hints([Eigenvalues of $alpha I + beta (bold(1) bold(1)^top - I)$: one along $bold(1)$, one (twice) across it (primer §LA.1).], [Compute $partial_a^2 L_gamma$ and $partial_a partial_b L_gamma$ at $(u, u, u)$, then replace $2 gamma$ using the quartic.], [First line of the solution: "At $(u, u, u)$: $partial_a^2 L_gamma = (b c)^2 + 2 gamma = u^4 + 2 gamma$ and $partial_a partial_b L_gamma = c r + a b c^2 = u (u^3 - s) + u^4$."])],
    [#tag("derive") For which $gamma$ does the nonzero minimum have the same loss as the origin, $s^2\/2$? Between that $gamma_c$ and $gamma^*$, which is the global minimum? #primer("ALG.2", "eliminating the parameter")
     #deliverable[$gamma_c (s)$ in closed form, and $gamma_c (1)$ to four digits; which point (the origin or the nonzero minimum) is the global minimum for $gamma_c < gamma < gamma^*$.]
     #hints([Write $L_gamma$ at a stationary point, set it equal to $s^2\/2$, and eliminate $gamma$ with the quartic (primer §ALG.2).], [With $q = u^3$ and the quartic, the loss at a stationary point is a quadratic in $q$.], [First line of the solution: "With $q = u^3$ and the quartic, $3 gamma u^2 = 3/2 q (s - q)$."])],
    [#tag("predict") Before running the check, write in `notes.md`: starting from $(1, 1, 1)$, how does the learned $a b c$ change as $gamma$ grows past $gamma^*$? If $gamma$ is then swept back down to $0$, warm-starting each run from the previous end point plus a small random kick, at which $gamma$ does the depth-3 net start learning again? The depth-2 net?],
    [#tag("implement") In `wdecay/depth3.py`:
     #defline[`def balanced_roots(gamma, s=1.0)`  ·  `def critical_gamma(s=1.0)`  ·  `def coexistence_gamma(s=1.0)`  ·  `def balanced_stationary(gamma, s=1.0)`]
     #arglist(
       ("balanced_roots", [the positive roots $u$ of your quartic, ascending (two or none).]),
       ("critical_gamma", [$gamma^*(s)$ from (b).]),
       ("coexistence_gamma", [$gamma_c (s)$ from (d).]),
       ("balanced_stationary", [`[(u, kind), ...]` with `kind` from (c).]),
     )
     #filetest("test_depth3", ("balanced_roots", "critical_gamma", "coexistence_gamma", "balanced_stationary"))],
    [#tag("check") Run `scripts/check_depth3.py`. Expected: the autograd Hessian at the origin equal to your (a); from $(1, 1.01, 0.99)$ the learned $a b c$ on your larger root cubed to $10^(-9)$ and jumping to $0$ at your $gamma^*$, against the depth-2 product for contrast; bisection on $gamma$ reproducing your $gamma^*(s)$ to $10^(-5)$ relative for $s = 0.25, 0.5, 1, 2$ and a log-log slope in $s$ that you can predict from (b); a hysteresis loop at depth 3 and none at depth 2 (`depth3.png`). Compare the exponent of $s$ with Ziyin et al.’s Proposition 3 at $D = 2$.],
  )
]

== Which starts fall into zero

A _basin of attraction_ of a stable point is the set of starts whose flow converges to it. The boundary between two basins is made of the _stable manifolds_ of saddles: the sets of starts that flow into the saddle itself.

#problem("basins", [Basins of the zero solution], kind: ("derive", "implement", "check"), time: "35 min", owner: "you")[
  #parts(
    [#tag("derive") On the balanced ray $a = b = c = u$ (it is invariant: why?), write the 1-D flow $dot(u) = F(u)$ for @eq-loss3, draw its phase line (sign of $F$ between its zeros, including $u < 0$), and state where every start $u_0 in RR$ ends, for $gamma < gamma^*$ and $gamma > gamma^*$. Then find the radius $u_s$ of the zero basin on the ray to leading order in small $gamma$ #primer("FP.3", "phase lines"), #primer("TAY.4", "small parameters").
     #deliverable[the endpoint rule ($u_infinity$ for every $u_0 in RR$, on each side of $gamma^*$); $u_s approx$ a simple function of $gamma$ and $s$ (type it into `small_gamma_radius` in `scripts/check_basins.py`).]
     #hints([Permuting $(a, b, c)$ is a symmetry, so the ray is invariant (primer §ODE.1); then draw the phase line of a 1-D flow (primer §FP.3).], [The zeros of $F$ are $0$ and the positive roots of your quartic; the sign of $F$ between them decides every endpoint. For small $gamma$, balance the two terms that stay when $u$ is small (primer §TAY.4).], [First line of the solution: "$L_gamma$ is symmetric under permutations of $(a, b, c)$, so at a point with $a = b = c$ the three gradient components are equal and the flow stays on the ray."])],
    [#tag("derive") The plane $a = b$ is also invariant. Which saddle's stable manifold separates the two basins in it, and why does the boundary in the half-plane $a = b < 0$ look like a mirror image? (Find a second symmetry of @eq-loss3.) #primer("FP.6", "stable manifolds")
     #deliverable[the saddle whose stable manifold is the boundary, with its coordinates in terms of the roots of your quartic; the second symmetry as an explicit map $(a, b, c) |-> (dots)$.]
     #hints([Basin boundaries are stable manifolds of saddles (primer §FP.6); which saddle of Problem (#raw("depth3")) lies in the plane $a = b$?], [Look for a sign flip of two weights that leaves $a b c$, and so the loss, unchanged.], [First line of the solution: "Swapping $a$ and $b$ is a symmetry, so the plane $a = b$ is invariant."])],
    [#tag("implement") In `wdecay/basins.py`:
     #defline[`def balanced_endpoint(u0, gamma, s=1.0)`  ·  `def zero_basin_radius(gamma, s=1.0)`]
     #arglist(
       ("balanced_endpoint", [$u_infinity$ for the start $(u_0, u_0, u_0)$, from (a); broadcasts over `u0`.]),
       ("zero_basin_radius", [the exact $u_s$ (`np.inf` above $gamma^*$).]),
     )
     #filetest("test_basins", ("balanced_endpoint", "zero_basin_radius"))],
    [#tag("check") Run `scripts/check_basins.py`. Expected: 201 balanced starts ending where you predict (to $10^(-10)$, away from $u_s$) with the measured edge bracketing your $u_s$ at $gamma = 0.1$; the basin map of the plane $a = b$ cut by the saddle's stable manifold; for i.i.d. $cal(N)(0, sigma^2)$ starts at $gamma = 0.03$, the fraction that learns rising from $0$ to $1$ as $sigma$ crosses a few times $u_s$, later for the weaker signal $s = 0.4$; your small-$gamma$ radius within $1%$ of the exact one for $gamma <= 0.1$ (`basins.png`). In `notes.md`: read off $sigma_(1\/2)$, the scale at which half the starts learn, for both $s$, and compare with $u_s$.],
  )
]

= Finite steps: oscillation and divergence

GD with step $eta$ on @eq-loss,
$ theta_(k+1) = theta_k - eta nabla L_gamma (theta_k), $ <eq-gd>
has the same fixed points as the flow. A0 showed that GD on a quadratic with curvature $lambda$ converges iff $0 < eta lambda < 2$; near a fixed point, GD is that quadratic in each eigendirection of $H(theta^*)$. Past the threshold the iterate first flips sign along the offending eigenvector every step; whether it then settles into a bounded oscillation or diverges is decided by the nonlinear terms. Ang's post shows the resulting diagram, with analytic stability boundaries and numerically observed oscillation and divergence regions @ang2026post[§4, the phase-diagram figure; the offline copy has only its caption].

#problem("finite-step", [The phase diagram of GD with weight decay], kind: ("derive", "implement", "check", "predict"), time: "50 min", owner: "you")[
  #parts(
    [#tag("derive") Linearize @eq-gd at the fixed point the flow converges to. Using Problem (#raw("stationary")), find $eta_"max" (gamma)$ such that it is linearly stable iff $eta < eta_"max"$, for $gamma < 1/2$ and $gamma > 1/2$. It has three pieces; on each, name the eigenvector that loses stability. Where is $eta_"max"$ largest, and what are its limits as $gamma -> 1/2$ from either side and as $gamma -> 0$? #primer("GD.3", "linearizing GD")
     #deliverable[$eta_"max" (gamma)$, piecewise in $gamma$; the unstable direction (a vector) on each piece; the maximum (value and location) and the three limits.]
     #hints([GD near a fixed point is the linear map $I - eta H$: stable iff every eigenvalue of $H$ lies in $(0, 2\/eta)$ (primer §GD.3).], [Use your eigenvalue pairs from Problem (#raw("stationary")); on each range of $gamma$ the largest eigenvalue at the attracting point sets the bound.], [First line of the solution: "Near a fixed point, $theta_(k+1) - theta^* approx (I - eta H)(theta_k - theta^*)$, which contracts iff $|1 - eta lambda| < 1$, i.e. $0 < lambda < 2\/eta$, for every eigenvalue."])],
    [#tag("predict") Before running the check, write in `notes.md`: for which $gamma$ do you expect a band of bounded oscillation above $eta_"max"$, and for which immediate divergence? (Think about which direction goes unstable, and what A1 said happens to $c$ when $eta^2 r^2 > 2$.)],
    [#tag("derive") On the invariant line $a = b = u$ with $s_"eff" = 1 - 2 gamma > 0$, @eq-gd is the 1-D map $u |-> u - eta u (u^2 - s_"eff")$. Find the $eta$ at which its fixed point $sqrt(s_"eff")$ loses stability #primer("GD.4", "1-D maps and trapping intervals"). Then substitute $v = u sqrt(eta \/ mu)$, $mu = 1 + eta s_"eff"$, to get $v |-> mu (v - v^3)$, and find the largest $mu$ for which the interval $|v| <= sqrt(1 + 1\/mu)$ is mapped into itself; beyond it almost every orbit escapes.
     #deliverable[the oscillation onset and the divergence onset as functions of $gamma$ (type both into `scripts/check_finite_step.py`).]
     #hints([$|mu (v - v^3)| > |v|$ iff $v^2 > 1 + 1\/mu$ (trapping intervals: primer §GD.4).], [On $|v| <= 1$ the map's largest value is at $v = 1\/sqrt(3)$.], [The condition is a cubic in $mu$ with an integer root (primer §ALG.2).])],
    [#tag("implement") In `wdecay/finite_step.py`:
     #defline[`def max_stable_step(gamma)`  ·  `def predicted_phase(eta, gamma)`]
     #arglist(
       ("max_stable_step", [$eta_"max" (gamma)$ from (a); broadcasts.]),
       ("predicted_phase", [`"nonzero"`, `"zero"` or `"unstable"` by linear stability.]),
     )
     #filetest("test_finite_step", ("max_stable_step", "predicted_phase"))],
    [#tag("check") Run `scripts/check_finite_step.py` (10,000 GD steps on each of $10^4$ cells, vectorized, about a second). Expected: from $(0.6, 0.4)$, your `predicted_phase` agrees with the measured converge/not-converge label on every cell more than $3%$ from your boundary; oscillation in only part of the $gamma$ range (check your prediction), with a ragged edge to the divergence region; from $(0.45, 0.45)$, both onsets within $1%$ (the $eta$ grid step) of your (c) (`finite_step.png`). Compare the left panel with the post's figure caption: which of its boundaries are analytic, and which only numerical, in your derivation?],
  )
]

= Reading

#problem("ziyin", [Exact solutions of a deep linear network], kind: ("read",), time: "20 min", owner: "you")[
  Read Ziyin et al. @ziyin2022exact §1 (the list of implications), §3.1 (Lemma 1 and Theorem 1), §4.1 (Propositions 2 and 3) and §5 ("Insufficiency of the existing initialization schemes", "Prediction variance of stochastic nets"). Answer briefly in `notes.md`:
  #parts(
    [Their hidden neurons are stochastic: $f = a epsilon b x$ with $EE[epsilon] = 1$, $EE[epsilon^2] = 1 + sigma^2$ (dropout is an example). Compute $EE_epsilon [1/2 (a epsilon b - 1)^2]$ and show the noise acts as an extra penalty on the product. Does it move the two-layer critical $gamma$? What does it do to $p_infinity$? Compare with their remark in §5 ("Learnability of a neural network") that the two-layer condition does not depend on stochasticity.],
    [Explain, with your Problem (#raw("depth3")) (a), why zero is a bad local minimum for two or more hidden layers but a saddle or the global minimum for one. Which term of the loss is responsible?],
    [Their §5 argues that initializations of radius about $1\/sqrt("width")$ can start in the zero basin when the signal $norm(EE[x y])$ is weak. Which of your measurements is the scalar version of this, and how does your basin radius scale with $s$?],
  )
]

= What to write down

In `notes.md`, three to five sentences: why weight decay balances layers; what makes the two-layer transition continuous and the three-layer one discontinuous; why relaxation slows down near a continuous transition (and what plays that role at $gamma^*$); one question for a tutorial.

#bibliography("/template/refs.bib", style: "/template/handout.csl", title: "References")
