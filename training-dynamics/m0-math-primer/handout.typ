#import "/template/template.typ": *

#show: doc.with(
  title: "M0: Math primer",
  kind: "Reference handout · read on demand",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "no tests, no stubs · start with the one-page self-diagnostic · answers to everything at the end",
)

#set heading(numbering: none)
#set math.equation(numbering: none)

// A pointer to a problem in an assignment: #pr("A1", "logistic", "(a)") -> A1 (logistic)(a)
#let pr(a, slug, parts) = [#a~(#raw(slug))#parts]
// "Where you use it" line closing each subsection.
#let where(body) = block(width: 100%, above: 0.9em, below: 1.4em, inset: (left: 8pt, y: 3pt),
  stroke: (left: 1.5pt + blue))[#text(weight: "bold")[Where you use it.] #body]
// Exercises at the end of a section, numbered ODE-E1, ODE-E2, ...
#let exercises(id, ..items) = block(above: 1.6em, below: 1.2em, width: 100%)[
  #text(weight: "bold")[Exercises (answers at the end)]
  #enum(numbering: n => id + "-E" + str(n), body-indent: 0.8em, spacing: 0.9em, ..items)
]

= Self-diagnostic

One line per question, about a minute each, pen and paper. Check against the next page and read only the sections whose question you missed or had to think hard about. The assignments point into this primer at the point of use, "(primer §ODE.4, logistic)", so you can also skip it entirely and come here on demand.

#v(0.4em)
#table(
  columns: (auto, 1fr, auto), align: (center, left, center), inset: (x: 6pt, y: 6pt),
  [\#], [Question], [Section],
  [1], [Solve $dot(x) = -3 x$, $x(0) = 2$. At what time is $x = 1$?], [ODE.3],
  [2], [Solve $dot(u) = 2 u (1 - u)$ with $u(0) = u_0 in (0, 1)$. When does $u$ reach $1\/2$?], [ODE.4],
  [3], [Solve $dot(v) = -2 theta v + sigma^2$, $v(0) = 0$. Whose variance is this?], [ODE.5, PR.4],
  [4], [Solve $dot(x) = 2 y$, $dot(y) = 2 x$ from $(x_0, y_0)$.], [ODE.6],
  [5], [Classify the fixed point $0$ of $dot(x) = -x + 2 y$, $dot(y) = 2 x - y$.], [FP.2],
  [6], [Fixed points of $dot(x) = mu x - x^3$ and their stability, for $mu < 0$ and $mu > 0$; the relaxation time of each stable one.], [FP.4, FP.5],
  [7], [GD on $L(theta) = 5 theta^2$: for which $eta$ does it converge, and for which does it flip sign while converging?], [GD.2],
  [8], [One GD step on $L = 1/2 (a b - 1)^2$: write $c' = a'^2 - b'^2$ exactly in terms of $c = a^2 - b^2$, $r = a b - 1$, $eta$.], [TAY.3],
  [9], [Eigenvalues and eigenvectors of $mat(2 gamma, -1; -1, 2 gamma)$; for which $gamma >= 0$ is it indefinite?], [LA.1, LA.2],
  [10], [Gradient of $L(W) = 1/2 norm(W X - Y)_F^2$ with respect to the matrix $W$.], [LA.5],
  [11], [If $L(alpha a, b\/alpha) = L(a, b)$ for every $alpha > 0$, what is $a thin partial_a L - b thin partial_b L$?], [LA.6],
  [12], [$z tilde cal(N)(0, sigma^2)$: $EE[z^4]$? The variance of the mean of $B$ i.i.d. draws of variance $v$?], [PR.1, PR.2],
  [13], [Solve $a^2 - b^2 = 3$, $a b = 2$ with $a, b > 0$.], [ALG.1],
  [14], [For which $gamma > 0$ does $u^3 - u + 2 gamma = 0$ have a positive root?], [ALG.2],
)

#pagebreak()

== Self-diagnostic: answers

#table(
  columns: (auto, 1fr, auto), align: (center, left, center), inset: (x: 6pt, y: 6pt),
  [\#], [Answer], [If wrong, read],
  [1], [$x(t) = 2 e^(-3 t)$; $x = 1$ at $t = ln 2 \/ 3 approx 0.231$ (the half-life).], [ODE.3],
  [2], [$u(t) = 1 \/ (1 + (1\/u_0 - 1) e^(-2 t))$; $t_(1\/2) = 1/2 ln((1 - u_0)\/u_0)$.], [ODE.4],
  [3], [$v(t) = sigma^2 \/ (2 theta) thin (1 - e^(-2 theta t))$: the variance of the Ornstein–Uhlenbeck process $d X = -theta X d t + sigma d W$ started at a point.], [ODE.5, PR.4],
  [4], [$x + y = (x_0 + y_0) e^(2 t)$, $x - y = (x_0 - y_0) e^(-2 t)$; add and subtract to get $x(t)$, $y(t)$.], [ODE.6],
  [5], [$J = mat(-1, 2; 2, -1)$ has eigenvalues $1$ on $(1, 1)$ and $-3$ on $(1, -1)$: a saddle.], [FP.2],
  [6], [$mu < 0$: only $x^* = 0$, stable, $tau = 1\/|mu|$. $mu > 0$: $0$ unstable, $plus.minus sqrt(mu)$ stable with linearization $-2 mu$, so $tau = 1\/(2 mu)$. A pitchfork at $mu = 0$.], [FP.4, FP.5],
  [7], [Curvature $lambda = 10$: converges iff $0 < eta < 0.2$; flips sign for $0.1 < eta < 0.2$.], [GD.2],
  [8], [$c' = c (1 - eta^2 r^2)$: the first-order terms cancel exactly.], [TAY.3],
  [9], [$2 gamma - 1$ on $(1, 1)$, $2 gamma + 1$ on $(1, -1)$; indefinite for $0 <= gamma < 1\/2$.], [LA.1, LA.2],
  [10], [$(W X - Y) X^top$.], [LA.5],
  [11], [$0$: differentiate the symmetry at $alpha = 1$.], [LA.6],
  [12], [$3 sigma^4$; $v \/ B$.], [PR.1, PR.2],
  [13], [$A = a^2$ solves $A^2 - 3 A - 4 = 0$, so $A = 4$: $(a, b) = (2, 1)$.], [ALG.1],
  [14], [$gamma = u (1 - u^2)\/2$ has maximum $1\/(3 sqrt(3)) approx 0.192$ at $u = 1\/sqrt(3)$: a positive root iff $gamma <= 0.192$ (two of them below).], [ALG.2],
)

#v(1em)
#runin[How to use this primer.] It is a reminder for someone who once passed these courses, not a textbook. Each subsection gives terse definitions, one worked example in the course's notation (on the anchor model $L(a, b) = 1/2 (a b - 1)^2$ where that does not give away an assignment's deliverable), and a "Where you use it" line naming the assignment and problem. Each section ends with three or four short exercises; every answer is in the last section. The last section before the answers looks ahead at math that A3–A7 develop themselves. If you want a textbook: Strogatz, _Nonlinear Dynamics and Chaos_, chapters 2–3 and 5–6, covers §ODE and §FP at this level; nothing here requires it.

#runin[Notation.] As in the course: parameters $theta$, loss $L$, step size $eta$, gradient flow $dot(theta) = -nabla L(theta)$. Anchor: $theta = (a, b)$, residual $r = a b - 1$ (or $a b - s$ for a target $s$), product $p = a b$, imbalance $c = a^2 - b^2$, weight decay $gamma$ with $L_gamma = L + gamma (a^2 + b^2)$. $H = nabla^2 L$, sharpness $lambda_max$. A dot is $d \/ d t$.

#outline(title: [Contents], depth: 2, indent: 1em)

#pagebreak()

= ODE: ordinary differential equations, the kit

== ODE.1 What a solution is

An ODE $dot(x) = F(x)$ with $x in RR^d$ assigns a velocity to every point. A _solution_ from $x_0$ is a differentiable curve $x(t)$ with $x(0) = x_0$ and $dot(x)(t) = F(x(t))$ for every $t$ in some interval. If $F$ is continuously differentiable, there is exactly one solution through each point (the existence and uniqueness theorem, Picard–Lindelöf). Three consequences you will use without saying so:

- *Trajectories never cross or merge*, and a fixed point ($F(x^*) = 0$) is never reached in finite time from anywhere else; approach is asymptotic.
- *Invariance by uniqueness.* If a quantity $q(t)$ obeys a linear homogeneous ODE $dot(q) = k(t) q$ and $q(0) = 0$, then $q equiv 0$ is a solution, hence the only one. The same trick shows that a guessed form (an _ansatz_) persists: if the velocity at every point of the form is again of that form, the solution stays in it.
- *Invariance by symmetry.* If a map $S$ commutes with the flow, $F(S x) = S F(x)$, the set of points with $S x = x$ is invariant. Swapping two weights of a symmetric loss keeps the plane where they are equal invariant.

Solutions need not exist forever: $dot(x) = x^2$ from $x_0 > 0$ is $x = 1\/(1\/x_0 - t)$, which blows up at $t = 1\/x_0$. Gradient flow on a loss bounded below does not blow up in practice, but a GD _map_ can (§GD.4).

#example("ode-invariant", [The balanced line is invariant])[
  For $L = 1/2 (s - a b)^2$, $dot(a) = b (s - a b)$ and $dot(b) = a (s - a b)$. So $q = a + b$ obeys $dot(q) = (a + b)(s - a b) = k(t) thin q$ with $k(t) = s - p(t)$. Hence $q(t) = q_0 exp(integral_0^t k)$ (§ODE.3): $q$ never changes sign, and if $a_0 = -b_0$ then $a = -b$ forever. Your turn in A1: the same argument for $a - b$.
]

#where[#pr("A0", "phase-portrait", "(c)")\; #pr("A1", "logistic", "(a)"), #pr("A1", "saxe", "(a)") (the ansatz); #pr("A2", "basins", "(a)–(b)").]

== ODE.2 Separable equations

A scalar autonomous ODE $dot(x) = f(x)$ is _separable_: where $f(x) != 0$, $d t = d x \/ f(x)$, so
$ t = integral_(x_0)^(x(t)) (d x') / f(x'). $
Do the integral, then solve for $x(t)$ if you can. Even if you cannot invert, the integral _is_ the time to go from $x_0$ to any $x$: this is how every "time to learn" in the course is computed.

#example("ode-sep", [Algebraic decay])[
  $dot(p) = -2 p^2$, $p(0) = p_0 > 0$. Then $integral_(p_0)^p d p' \/ (-2 p'^2) = 1/2 (1\/p - 1\/p_0) = t$, so
  $ p(t) = p_0 / (1 + 2 p_0 t). $
  For large $t$, $p approx 1\/(2 t)$ whatever $p_0$: the approach to $0$ is a power law, not an exponential. This happens when the linear term of $f$ at the fixed point vanishes (§FP.5).
]

#where[#pr("A1", "logistic", "(a)–(b)")\; #pr("A2", "slowing", "(c)").]

== ODE.3 Linear first order, constant or varying coefficient

$dot(x) = -lambda x$ has $x(t) = x_0 e^(-lambda t)$. For $lambda > 0$: decay with _time constant_ $tau = 1\/lambda$ (shrinks by a factor $e$ per $tau$) and _half-life_ $ln 2 \/ lambda$; for $lambda < 0$, growth with doubling time $ln 2 \/ |lambda|$. With a constant input, $dot(x) = -lambda x + b$ has the fixed point $x^* = b\/lambda$ and
$ x(t) = x^* + (x_0 - x^*) e^(-lambda t): $
subtract the fixed point and you are back to pure decay. With a time-varying rate, $dot(x) = -k(t) x$ gives $x(t) = x_0 exp(-integral_0^t k(t') d t')$, so $ln(x(t)\/x_0) = -integral_0^t k$; $x$ never changes sign. A matrix obeying $dot(D) = -k D$ does this entry by entry: $D(t) = D(0) e^(-k t)$.

#example("ode-lin", [Ridge regression flow])[
  One weight, $L_gamma (w) = 1/2 (w - 1)^2 + gamma w^2$ (A2’s example). Gradient flow: $dot(w) = 1 - (1 + 2 gamma) w$, so $w^* = 1\/(1 + 2 gamma)$, $lambda = 1 + 2 gamma$, and $w(t) = w^* + (w_0 - w^*) e^(-(1 + 2 gamma) t)$. At $gamma = 1\/2$: $w^* = 1\/2$, time constant $1\/2$, half-life $ln 2 \/ 2 = 0.347$.
]

#where[#pr("A0", "ou", "(a)")\; #pr("A1", "broken", "(a)") and (c) (the time-varying rate $k = eta r^2$); #pr("A2", "balance", "(a)–(c)").]

== ODE.4 The logistic equation, solved fully

$ dot(u) = 2 u (s - u), quad s > 0, quad 0 < u_0 < s. $
Growth proportional to $u$, braked as $u$ approaches its ceiling $s$. It is the learning curve of a balanced mode (A1). *Solve.* Partial fractions, $1/(u (s - u)) = 1/s (1/u + 1/(s - u))$, so
$ t = integral_(u_0)^u (d u') / (2 u' (s - u')) = 1/(2 s) [ln u' - ln(s - u')]_(u_0)^u = 1/(2 s) ln (u (s - u_0)) / (u_0 (s - u)). $
Solve for $u$: $u \/ (s - u) = u_0 \/ (s - u_0) thin e^(2 s t)$, i.e.
$ u(t) = s / (1 + (s\/u_0 - 1) e^(-2 s t)). $
Checks: $u(0) = u_0$; $u -> s$; while $u << s$, $u approx u_0 e^(2 s t)$ (exponential growth at rate $2 s$); near $s$, $s - u approx "const" dot e^(-2 s t)$ (linearize: $f'(s) = 2 s - 4 s = -2 s$). This form never overflows for large $t$.

*Time between two levels.* From the first display, the time from $u_1$ to $u_2$ is $1/(2 s) ln (u_2 (s - u_1)) / (u_1 (s - u_2))$. In particular the *time to get within $epsilon$ of $s$* from $u_0$ is
$ t_epsilon = 1/(2 s) ln ((s - epsilon)(s - u_0)) / (u_0 epsilon) approx 1/(2 s) [ln s/u_0 + ln s/epsilon] quad (u_0, epsilon << s): $
a _plateau_ term that grows like $ln(1\/u_0)$ and a _final approach_ term at the linear rate $2 s$.

*Negative or zero target.* For $s < 0$ the same formulas hold (both $s - u_0$ and $s - u$ are negative, the logarithm stays real) and $u -> 0$ at rate $2|s|$; for $s = 0$ it is §ODE.2’s algebraic decay.

#example("ode-logistic", [Numbers])[
  $s = 1$, $u_0 = 10^(-3)$, $epsilon = 10^(-3)$: $t_epsilon = 1/2 ln((0.999)^2 \/ 10^(-6)) = 6.9068$, of which $1/2 ln 10^3 = 3.45$ is spent getting off the ground. Integrating $dot(u) = 2 u (1 - u)$ numerically to $t = 6.9068$ gives $u = 0.999000$.
]

#where[#pr("A1", "logistic", "(a)–(b)"), #pr("A1", "saxe", "(b)")\; #pr("A2", "slowing", "(a)"), (d).]

== ODE.5 Variation of constants

For $dot(x) = -lambda x + g(t)$ with a time-dependent input $g$: multiply by $e^(lambda t)$, so the left side becomes $d\/d t (e^(lambda t) x) = e^(lambda t) g(t)$, and integrate:
$ x(t) = e^(-lambda t) x_0 + integral_0^t e^(-lambda (t - t')) g(t') thin d t'. $
Read it as: the input at time $t'$ is remembered with weight $e^(-lambda (t - t'))$, a memory of length $1\/lambda$.

#example("ode-voc", [A decaying input])[
  $g(t) = A e^(-mu t)$ with $mu != lambda$: $integral_0^t e^(-lambda (t - t')) A e^(-mu t') d t' = A e^(-lambda t) (e^((lambda - mu) t) - 1)\/(lambda - mu)$, so
  $ x(t) = x_0 e^(-lambda t) + A (e^(-mu t) - e^(-lambda t)) / (lambda - mu). $
  For a constant input $g = b$ the formula returns §ODE.3. §PR.4 uses it for the variance of a noisy process.
]

#where[#pr("A0", "ou", "(b)") (the variance ODE); A4 (fluctuations around a minimum).]

== ODE.6 Linear systems $dot(x) = A x$

If $A$ has eigenpairs $(mu_i, v_i)$ that span $RR^d$, write $x_0 = sum_i c_i v_i$; each component evolves on its own:
$ x(t) = sum_i c_i e^(mu_i t) v_i = V e^(Lambda t) V^(-1) x_0 = e^(A t) x_0. $
For symmetric $A$ (every gradient flow near a point), the $v_i$ are orthonormal and $c_i = v_i^top x_0$. A matrix unknown obeying $dot(E) = -E Sigma$ evolves row by row, $E(t) = E(0) e^(-Sigma t)$ with $e^(-Sigma t) = U e^(-Lambda t) U^top$ for $Sigma = U Lambda U^top$. Eigenvalue $0$ means that component never moves: a conserved combination, or a line of fixed points.

#example("ode-sys", [Two coupled decays])[
  $dot(x) = -2 x + y$, $dot(y) = x - 2 y$: $A = mat(-2, 1; 1, -2)$ has $mu = -1$ on $(1, 1)$ and $mu = -3$ on $(1, -1)$ (§LA.1). So $x + y = (x_0 + y_0) e^(-t)$ and $x - y = (x_0 - y_0) e^(-3 t)$: add and subtract,
  $ vec(x, y)(t) = (x_0 + y_0)/2 e^(-t) vec(1, 1) + (x_0 - y_0)/2 e^(-3 t) vec(1, -1). $
  After a time of order $1$ the fast component is gone and the state slides in along $(1, 1)$. With $A = mat(0, s; s, 0)$ (the anchor linearized at the origin) the same change of variables gives $x + y$ growing like $e^(s t)$ and $x - y$ decaying like $e^(-s t)$.
]

#where[#pr("A0", "modes", "(b)"), (e); #pr("A1", "saxe", "(d)").]

== ODE.7 What "time to converge" means

When the approach is exponential, distance $d(t) approx K e^(-t\/tau)$, so $ln d(t) = ln K - t\/tau$ is a straight line in $t$ (a semi-log plot), with slope $-1\/tau$. Consequences: the time to reach $epsilon$ is $t_epsilon approx tau ln(K\/epsilon)$, only logarithmic in the target accuracy and the starting distance; $tau$ can be measured as $(t_2 - t_1) \/ ln(d_1 \/ d_2)$ between two late distances, which removes $K$; and with several modes, the slowest one sets $tau$ at late times. Escaping a saddle is the same law run backwards: a start at distance $delta_0$ along a direction with growth rate $mu > 0$ needs time $ln(Delta\/delta_0)\/mu$ to reach size $Delta$. An algebraic approach ($d prop t^(-1\/2)$, §ODE.2) is curved on a semi-log plot and straight on a log-log plot (§ALG.4).

#example("ode-time", [Reading off the time])[
  In the example of §ODE.6 from $(1, 0)$, $norm(x(t)) approx e^(-t) \/ sqrt(2)$ once the $e^(-3 t)$ part is gone. To reach $10^(-6)$: $t = ln(0.7071 \/ 10^(-6)) = 13.47$, confirmed by integrating. Halving the target adds $tau ln 2 = 0.69$.
]

#where[#pr("A0", "modes", "(d)")\; #pr("A1", "endpoint", "(c)") (escaping the saddle); #pr("A2", "slowing", "(b)"), (d), (f).]

#exercises("ODE",
  [Solve $dot(x) = -x^3$, $x(0) = 1$. When is $x = 0.1$?],
  [For $dot(u) = 2 u (s - u)$ with $s = 2$, $u_0 = 0.02$: how long until $u = 1$?],
  [Solve $dot(x) = -2 x + 4$, $x(0) = 0$. Time constant? When is $x$ within $0.01$ of its limit?],
  [Solve $dot(x) = -x + y$, $dot(y) = x - y$. What is conserved, and where does the state end?],
)

= FP: fixed points and linearization

== FP.1 Fixed points and the Jacobian

A _fixed point_ of $dot(x) = F(x)$ is an $x^*$ with $F(x^*) = 0$. Near it write $x = x^* + delta$; Taylor (§TAY.1) gives
$ dot(delta) = J delta + O(norm(delta)^2), quad J_(i j) = (partial F_i) / (partial x_j) (x^*), $
a linear system solved by §ODE.6. If no eigenvalue of $J$ has zero real part (the point is _hyperbolic_), the linearization shows the true local picture (Hartman–Grobman). For gradient flow, $F = -nabla L$, so $J = -nabla^2 L(x^*)$ is symmetric with real eigenvalues: a fixed point of gradient flow is a stationary point of $L$, and it is stable iff the Hessian is positive definite there (§LA.2).

#where[#pr("A0", "phase-portrait", "(b)")\; #pr("A2", "stationary", "(a)").]

== FP.2 Classifying a 2-D fixed point

For a $2 times 2$ Jacobian with trace $T$ and determinant $Delta$, the eigenvalues solve $mu^2 - T mu + Delta = 0$ (§LA.1).
- $Delta < 0$: real, opposite signs: *saddle* (in along one eigenvector, out along the other).
- $Delta > 0$, $T < 0$: *stable* (node if $T^2 >= 4 Delta$, spiral if not); $Delta > 0$, $T > 0$: *unstable*.
- $Delta = 0$: one eigenvalue is $0$: *degenerate*, the linearization is silent in that direction.
Gradient flows have symmetric $J$, so no spirals: only stable nodes, saddles, unstable nodes and degenerate points.

#example("fp-classify", [Two Jacobians])[
  $J = mat(-3, 1; 1, -3)$: $T = -6$, $Delta = 8$, eigenvalues $-2$ and $-4$: stable node, slow direction $(1, 1)$. $J = mat(1, 2; 2, 1)$: $Delta = -3 < 0$: saddle, eigenvalues $3$ on $(1, 1)$ and $-1$ on $(1, -1)$.
]

#where[#pr("A0", "phase-portrait", "(b)"), (d); #pr("A2", "stationary", "(a)"), #pr("A2", "depth3", "(c)").]

== FP.3 One-dimensional flows: the phase line

For $dot(x) = F(x)$ with $x in RR$, draw the real line, mark the zeros of $F$, and put an arrow on each interval between zeros by the sign of $F$. Every trajectory is monotone and goes to the next zero in the direction of its arrow (or to $plus.minus infinity$). A zero is stable iff $F'(x^*) < 0$ (arrows point in). The _basin_ of a stable zero is the interval between the unstable zeros on either side of it.

#example("fp-line", [Three fixed points])[
  $F(u) = -u (u - 1)(u - 3)$. Zeros $0, 1, 3$; $F'(0) = -3$, $F'(1) = 2$, $F'(3) = -6$: stable, unstable, stable. Signs: $F > 0$ on $u < 0$, $F < 0$ on $(0, 1)$, $F > 0$ on $(1, 3)$, $F < 0$ above $3$. Endpoint rule: $u_0 < 1$ ends at $0$, $u_0 > 1$ ends at $3$, $u_0 = 1$ stays. The unstable zero is the basin boundary.
]

#where[#pr("A2", "basins", "(a)")\; #pr("A0", "phase-portrait", "(c)") (the flow on an invariant line).]

== FP.4 Relaxation time

At a stable fixed point with Jacobian eigenvalues $mu_i < 0$ (real), $delta(t) = sum_i c_i e^(mu_i t) v_i$; at late times the slowest term wins, so the distance shrinks by a factor $e$ every
$ tau = 1 / (min_i |mu_i|), quad "for gradient flow" quad tau = 1 / lambda_"min" (nabla^2 L(x^*)). $
This is the _relaxation time_; §ODE.7 says how to measure it. A start with exactly no component along the slowest eigenvector relaxes faster, but a generic start does not.

#example("fp-tau", [The slow direction])[
  $nabla^2 L(x^*) = mat(1, 0.6; 0.6, 1)$ has eigenvalues $1.6$ on $(1, 1)$ and $0.4$ on $(1, -1)$ (§LA.1), so $tau = 1\/0.4 = 2.5$, set by the $(1, -1)$ direction. Getting from distance $1$ to $10^(-6)$ takes about $2.5 ln 10^6 = 34.5$.
]

#where[#pr("A2", "slowing", "(b)").]

== FP.5 Bifurcations and critical slowing down

As a _control parameter_ $mu$ changes, fixed points move and their eigenvalues change. A _bifurcation_ is a value of $mu$ where the set of fixed points or their stability changes qualitatively; for real eigenvalues this happens when one crosses zero. Two normal forms cover the course (Strogatz §3.1, §3.4; Figure 1):

- *Pitchfork*, $dot(x) = mu x - x^3$. For $mu < 0$ the only fixed point is $0$, stable with eigenvalue $mu$. For $mu > 0$, $0$ is unstable and $plus.minus sqrt(mu)$ are stable with eigenvalue $mu - 3 mu = -2 mu$. The stable state moves _continuously_ away from $0$: a continuous transition.
- *Saddle-node*, $dot(x) = mu - x^2$. For $mu > 0$, $+sqrt(mu)$ is stable (eigenvalue $-2 sqrt(mu)$) and $-sqrt(mu)$ unstable; at $mu = 0$ they merge, and for $mu < 0$ there is no fixed point nearby at all. A system sitting on the stable branch has to _jump_ somewhere else when $mu$ crosses $0$: a discontinuous transition, with _hysteresis_ (undoing the change of $mu$ does not undo the jump).

*Critical slowing down.* Near the bifurcation the relevant eigenvalue goes to zero, so $tau$ diverges: in the pitchfork $tau = 1\/|mu|$ on the zero side and $1\/(2 mu)$ on the other, both $prop 1\/|mu|$ but with different constants; in the saddle-node $tau = 1\/(2 sqrt(mu)) prop mu^(-1\/2)$. Exactly at the bifurcation the linear term is gone and the approach is algebraic: $dot(x) = -x^3$ gives $x approx (2 t)^(-1\/2)$ (§ODE.2).

#figure(image("figures/fixed_points.png", width: 100%), caption: [Left and middle: the two bifurcations, fixed points against the control parameter. Right: the anchor's gradient flow (§FP.6). Made by `m0-math-primer/make_figures.py`.])

#where[#pr("A2", "stationary", "(a)"), #pr("A2", "slowing", "(b)–(c)"), #pr("A2", "depth3", "(b)–(c)"), (e).]

== FP.6 Phase portraits, conserved quantities, and the anchor

A _phase portrait_ draws the trajectories of a 2-D system. To sketch one: find the fixed points and classify them (§FP.2); draw the eigenvector directions at each; look for invariant lines (§ODE.1) and conserved quantities. A _conserved quantity_ $C(x)$, with $d\/d t thin C(x(t)) = nabla C dot F = 0$, confines each trajectory to a level set of $C$. The _stable manifold_ of a saddle (the starts that flow into it) is a curve in 2-D and separates basins; that is how basin boundaries arise. If a whole curve $x^*(sigma)$ consists of fixed points, differentiating $F(x^*(sigma)) = 0$ in $sigma$ gives $J thin d x^* \/ d sigma = 0$: the tangent to the curve is a zero-eigenvalue eigenvector.

#example("fp-anchor", [The anchor with target $s$, at the origin])[
  $L = 1/2 (a b - s)^2$, $F = (-b (a b - s), -a (a b - s))$. Fixed points: the origin and every point of the hyperbola $a b = s$. Jacobian: $partial_a (-b (a b - s)) = -b^2$ and $partial_b (-b (a b - s)) = -(a b - s) - a b = s - 2 a b$, so
  $ J(a, b) = mat(-b^2, s - 2 a b; s - 2 a b, -a^2). $
  At the origin $J = mat(0, s; s, 0)$: $T = 0$, $Delta = -s^2$, eigenvalues $plus.minus s$, unstable along $(1, 1)$, stable along $(1, -1)$: a saddle, whose stable manifold is the line $a = -b$ (invariant by §ODE.1). On $a b = s$, $Delta = a^2 b^2 - s^2 = 0$, so one eigenvalue is $0$, along the tangent of the hyperbola by the argument above, and the other equals the trace. The dashed curves in Figure 1 are level sets of a conserved quantity; A0 asks you to find it and A1 why it exists.
]

#where[#pr("A0", "phase-portrait", "(a)–(c)")\; #pr("A1", "endpoint", "(a)")\; #pr("A2", "basins", "(b)").]

#exercises("FP",
  [Fixed points of $dot(x) = x - x^3$, their types, and the relaxation time of each stable one.],
  [Classify the origin for gradient flow on $L(x, y) = x^2 + 3 x y + y^2$; along which direction does the flow leave?],
  [For $dot(x) = mu - x^2$ at $mu = 0.01$: the stable point and its relaxation time. How does $tau$ scale with $mu$?],
  [For $L(x, y) = 1/2 (x + y - 1)^2$, show every point of $x + y = 1$ is fixed, and find the Jacobian's eigenvalues and eigenvectors there.],
)

= GD: gradient flow versus gradient descent

== GD.1 GD is forward Euler

_Forward Euler_ with time step $h$ solves $dot(x) = F(x)$ by $x_(k+1) = x_k + h F(x_k)$: follow the current velocity for a time $h$. With $F = -nabla L$ and $h = eta$ it is GD, and $theta_k$ approximates the flow at time $t_k = k eta$. *Local error* (one step started on the exact solution): the exact solution is $x(h) = x_0 + h F(x_0) + h^2/2 dot.double(x)(0) + O(h^3)$ (§TAY.1), Euler keeps the first two terms, so the error is $O(h^2)$. *Global error* at a fixed time $T$: $T\/h$ steps of $O(h^2)$ each give $O(h)$, provided errors are not amplified too much on the way (Grönwall's inequality bounds the amplification by $e^(M T)$, a constant for fixed $T$). Euler is therefore a _first-order_ method, and "GD tracks gradient flow" means "to $O(eta)$ over a fixed time".

#example("gd-euler", [Euler on $dot(x) = -x$])[
  $x_0 = 1$, $h = 0.1$. One step: $0.9$ against $e^(-0.1) = 0.904837$, error $0.00484 approx h^2\/2 = 0.005$. Ten steps to $T = 1$: $0.9^10 = 0.348678$ against $e^(-1) = 0.367879$, error $0.0192$, close to the first-order prediction $(h\/2) T e^(-T) = 0.0184$. Halve $h$ and the one-step error drops by $4$, the error at $T$ by $2$.
]

#where[#pr("A0", "flow-limit", "(a)–(c)")\; #pr("A1", "matrix-law", "(e)") (counting steps to a fixed time).]

== GD.2 Stability of a linear map

GD on a quadratic $L(theta) = lambda/2 theta^2$ is the linear map $theta_(k+1) = (1 - eta lambda) theta_k$, so $theta_k = (1 - eta lambda)^k theta_0$. Everything depends on $q = 1 - eta lambda$:
#table(columns: 3, align: (center, center, left),
  [$eta lambda$], [$q$], [behaviour],
  [$(0, 1)$], [$(0, 1)$], [monotone convergence],
  [$1$], [$0$], [exact minimum in one step],
  [$(1, 2)$], [$(-1, 0)$], [converges, flipping sign every step (oscillation)],
  [$2$], [$-1$], [bounces between $plus.minus theta_0$ forever],
  [$> 2$], [$< -1$], [diverges with sign flips],
  [$< 0$ (negative curvature)], [$> 1$], [diverges monotonically, for every $eta > 0$],
)
So GD converges iff $0 < eta lambda < 2$, i.e. $lambda < 2\/eta$. For $L = 1/2 theta^top H theta$ with $H = Q Lambda Q^top$, rotate to $z = Q^top theta$ (§LA.1): each $z_i$ is multiplied by $1 - eta lambda_i$ independently, so GD converges iff $eta < 2 \/ lambda_max$, and for large $k$ the norm shrinks by the _contraction factor_ $max_i |1 - eta lambda_i|$ per step, which is decided by the two extreme eigenvalues.

#example("gd-lin", [$eta lambda = 1.6$])[
  $lambda = 4$, $eta = 0.4$: $q = -0.6$, so $theta_k = 1, -0.6, 0.36, -0.216, 0.130, dots$: overshoots every step and converges. At $eta = 0.55$, $q = -1.2$ and the same picture grows.
]

#where[#pr("A0", "gd-stability", "(a)–(b)"), #pr("A0", "hessian", "(b)")\; A3 (the $2\/eta$ threshold).]

== GD.3 The quadratic as the universal local model

Near a fixed point $theta^*$ of GD ($nabla L(theta^*) = 0$), Taylor-expand the gradient: $nabla L(theta) = H (theta - theta^*) + O(norm(theta - theta^*)^2)$ with $H = nabla^2 L(theta^*)$. So the error obeys
$ theta_(k+1) - theta^* = (I - eta H)(theta_k - theta^*) + O(norm(theta_k - theta^*)^2), $
§GD.2 in each eigendirection of $H$. The fixed point is linearly stable for GD iff every eigenvalue of $H$ lies in $(0, 2\/eta)$. A saddle ($H$ has a negative eigenvalue) is unstable for every $eta$. A zero eigenvalue along a curve of minima gives the factor $1$: neutral, it neither grows nor shrinks (neighbouring points are also minima), so it does not count against stability. For a general map $theta |-> G(theta)$ the rule is: stable iff every eigenvalue of the Jacobian $D G(theta^*)$ has modulus below $1$.

#example("gd-local", [Which direction goes first])[
  At a minimum with $H = mat(1, 0.6; 0.6, 1)$ (eigenvalues $1.6$ on $(1, 1)$, $0.4$ on $(1, -1)$), GD is stable iff $eta < 2\/1.6 = 1.25$. At $eta = 1.3$ the $(1, 1)$ component is multiplied by $1 - 2.08 = -1.08$ per step: it flips and grows, while the $(1, -1)$ component still shrinks by $0.48$.
]

#where[#pr("A0", "hessian", "(b)")\; #pr("A2", "finite-step", "(a)")\; A3.]

== GD.4 One-dimensional nonlinear maps

For a map $x |-> f(x)$, a fixed point $x^* = f(x^*)$ is stable iff $|f'(x^*)| < 1$; if $f'(x^*) < 0$ the approach alternates sides. When $f'(x^*)$ passes $-1$ the fixed point loses stability to a _period-2 orbit_ (period doubling), and further on the orbit can become chaotic. Whether orbits stay bounded is a separate, nonlinear question. The tool is a _trapping interval_: if a closed interval $I$ satisfies $f(I) subset.eq I$, every orbit started in $I$ stays in it. To check, compare the extreme values of $f$ on $I$ (at the endpoints and at critical points $f' = 0$) with the endpoints of $I$.

#example("gd-map", [The logistic map])[
  $f(x) = mu x (1 - x)$ on $[0, 1]$. Fixed point $x^* = 1 - 1\/mu$ with $f'(x^*) = 2 - mu$: stable for $1 < mu < 3$ (alternating for $mu > 2$), period doubling at $mu = 3$. The largest value of $f$ on $[0, 1]$ is $f(1\/2) = mu\/4$, so $[0, 1]$ is trapping iff $mu <= 4$; above, the top of the map leaves the interval and almost every orbit escapes to $-infinity$. At $mu = 2.5$: $x^* = 0.6$, $f' = -0.5$, alternating convergence.
]

#where[#pr("A2", "finite-step", "(c)").]

#exercises("GD",
  [GD on $L = 2 theta^2$ with $eta = 0.3$: write $theta_k$ and describe it. What is the largest stable $eta$?],
  [Forward Euler with $h = 0.5$ on $dot(x) = -x$ from $x_0 = 1$: four steps, against the exact value at $t = 2$.],
  [At a minimum on a line of minima, $H$ has eigenvalues $0, 1, 4$. For which $eta$ is GD linearly stable there?],
  [Fixed points of $x |-> x - eta x (x^2 - 4)$ and the range of $eta > 0$ for which $x^* = 2$ is stable.],
)

= TAY: Taylor expansions and order notation

== TAY.1 Taylor in one and several variables

$ f(x + h) = f(x) + f'(x) h + 1/2 f''(x) h^2 + O(h^3), $
$ L(theta + delta) = L(theta) + nabla L(theta)^top delta + 1/2 delta^top nabla^2 L(theta) delta + O(norm(delta)^3), quad nabla L(theta + delta) = nabla L(theta) + nabla^2 L(theta) delta + O(norm(delta)^2). $
Along a curve $theta(t)$, the chain rule gives $d\/d t thin g(theta(t)) = nabla g(theta)^top dot(theta)$ and, for the gradient itself, $d\/d t thin nabla L(theta(t)) = nabla^2 L(theta(t)) thin dot(theta)(t)$: "the derivative of $nabla L$ along a curve is the Hessian times the velocity". The Hessian at a point sees only the quadratic part of the expansion there: terms of degree three or more contribute nothing to second derivatives at the expansion point.

#example("tay-anchor", [The anchor near $(1, 1)$])[
  With $a = 1 + delta_a$, $b = 1 + delta_b$: $a b - 1 = delta_a + delta_b + delta_a delta_b$, so $L = 1/2 (delta_a + delta_b)^2 + O(delta^3)$. Read off $nabla L(1, 1) = 0$ and $nabla^2 L(1, 1) = mat(1, 1; 1, 1)$, with eigenvalue $0$ along $(1, -1)$: to second order the loss does not change along the curve of minima.
]

#where[#pr("A0", "flow-limit", "(b)")\; #pr("A1", "matrix-law", "(a)") (keep the first-order term); #pr("A2", "depth3", "(a)").]

== TAY.2 Order notation, and what "to first order" means

$g(eta) = O(eta^p)$ as $eta -> 0$ means $|g(eta)| <= C eta^p$ for small $eta$. "To first order in $eta$" means keep every term up to $eta^1$, from every factor, and write the rest as $+ O(eta^2)$. "Leading order" means the first term that does not vanish. Expansions to know (small $x$):
$ e^x = 1 + x + x^2/2 + O(x^3), quad ln(1 + x) = x - x^2/2 + O(x^3), quad (1 + x)^alpha = 1 + alpha x + O(x^2), $
$ sqrt(4 + c^2) = 2 + c^2/4 + O(c^4), quad 1/(1 - x) = 1 + x + O(x^2), quad "arcsinh"(y) = ln(2 y) + O(y^(-2)) "for large" y. $
Two patterns recur. *Products of exponentials*: $(1 - 2 eta gamma)^2 = 1 - 4 eta gamma + O(eta^2) = e^(-4 eta gamma) + O(eta^2)$, so a per-step factor is a flow over time $eta$. *Accumulation*: $K = T\/eta$ steps that each make an $O(eta^2)$ change add up to $O(eta)$; a per-step change of $O(eta)$ adds up to $O(1)$. Bookkeeping rule: decide the order you want before expanding, keep the $O(dot)$ term written in every line, and only cancel terms of the same order.

#where[#pr("A0", "ou", "(b)") (first order in $Delta t$); #pr("A1", "logistic", "(b)"), #pr("A1", "broken", "(b)–(e)").]

== TAY.3 Worked: a conserved quantity under one GD step

Gradient flow on the anchor conserves $c = a^2 - b^2$. What does one GD step do? With $r = a b - 1$, the step is $a' = a - eta b r$, $b' = b - eta a r$. Expand exactly (no truncation):
$ a'^2 - b'^2 = (a^2 - 2 eta a b r + eta^2 b^2 r^2) - (b^2 - 2 eta a b r + eta^2 a^2 r^2) = c - eta^2 r^2 (a^2 - b^2), $
$ c' = c (1 - eta^2 r^2). $
The $O(eta)$ terms cancel _exactly_: the gradient $r (b, a)$ is orthogonal to $(a, -b)$, the direction in which $c$ changes (§LA.6). What is left is $O(eta^2)$ per step, so over a fixed time ($T\/eta$ steps) $c$ drifts by $O(eta)$ (§TAY.2), and $ln(c'\/c) = ln(1 - eta^2 r^2) = -eta^2 r^2 + O(eta^4)$. Ang's post derives the same factor (§3.1). The general lesson: when the first-order term of a change vanishes for a structural reason, the second-order term is the whole story, and you must expand to second order exactly.

#where[#pr("A1", "matrix-law", "(c)"), #pr("A1", "broken", "(a)")\; #pr("A0", "flow-limit", "(d)")\; A3.]

== TAY.4 Solving with a small parameter

To find a quantity that depends on a small parameter $epsilon$, guess a series $x = x_1 epsilon + x_2 epsilon^2 + dots$, substitute, and match powers; or, more quickly, ask which terms _balance_ (are of the same size) and drop the rest. Typical uses: the small root of a polynomial, the leading behaviour of a closed form for a small initialization, a difference of nearly equal square roots.

#example("tay-small", [Three small-parameter answers])[
  (i) The small root of $x^3 - 3 x + epsilon = 0$: for small $x$ the cube is negligible, so $3 x approx epsilon$, $x approx epsilon\/3$; the next term solves $(epsilon\/3)^3 - 3 delta = 0$, $delta = epsilon^3\/81$. At $epsilon = 0.1$: $0.0333457$ exactly, $0.0333457$ from the two terms. (ii) $1/(2 s) ln((s - u_0)\/u_0) = 1/(2 s) [ln(s\/u_0) + ln(1 - u_0\/s)] = 1/(2 s) ln(s\/u_0) + O(u_0\/s^2)$: the leading term of a logistic time. (iii) $sqrt(1 + c) - sqrt(1 - c) = (1 + c\/2) - (1 - c\/2) + O(c^2) = c + O(c^2)$ (in fact $O(c^3)$ by symmetry).
]

#where[#pr("A1", "noether", "(c)"), #pr("A1", "logistic", "(b)"), #pr("A1", "saxe", "(b)")\; #pr("A2", "basins", "(a)").]

#exercises("TAY",
  [Expand $sqrt(4 + c^2)$ and $1\/sqrt(4 + c^2)$ to order $c^2$.],
  [To first order in $eta$, what is $(1 - 2 eta gamma)^2 - eta^2 r^2$?],
  [One GD step on $L = 1/2 (a b - 1)^2$: write $p' = a' b'$ exactly in terms of $p$, $r$, $eta$ and $a^2 + b^2$, then $r' = p' - 1$ as $r$ times a factor.],
  [The small root of $x^3 - 4 x + epsilon = 0$, to leading order and with the next correction.],
)

= LA: linear algebra reminders

== LA.1 Symmetric matrices and their eigenvalues

A real symmetric $H$ has real eigenvalues and an orthonormal basis of eigenvectors, $H = Q Lambda Q^top$ (the spectral theorem). In the coordinates $z = Q^top theta$, $H$ is diagonal; lengths are unchanged since $Q$ is orthogonal. The _Rayleigh quotient_ $v^top H v \/ v^top v$ lies between $lambda_"min"$ and $lambda_max$ and hits them at the eigenvectors. For $2 times 2$ matrices, the eigenvalues solve $mu^2 - (op("tr") H) mu + det H = 0$: their sum is the trace, their product the determinant. Patterns worth recognizing on sight:
- $mat(p, q; q, p)$: eigenvalue $p + q$ on $(1, 1)$, $p - q$ on $(1, -1)$.
- $alpha I + beta (bold(1) bold(1)^top - I)$ in $n$ dimensions ($alpha$ on the diagonal, $beta$ off it): $alpha + (n - 1) beta$ on $bold(1) = (1, dots, 1)$, and $alpha - beta$ on every vector orthogonal to $bold(1)$ (multiplicity $n - 1$).
- $H + k I$ has eigenvalues $lambda_i + k$, same eigenvectors.

#example("la-eig", [Two quick ones])[
  $mat(3, 1; 1, 3)$: $4$ on $(1, 1)$, $2$ on $(1, -1)$. $mat(5, 1, 1; 1, 5, 1; 1, 1, 5)$ ($alpha = 5$, $beta = 1$, $n = 3$): $7$ on $(1, 1, 1)$ and $4$ twice, on the plane $x + y + z = 0$.
]

#where[#pr("A0", "gd-stability", "(b)"), #pr("A0", "phase-portrait", "(b)")\; #pr("A2", "stationary", "(a)"), #pr("A2", "depth3", "(c)").]

== LA.2 Quadratic forms, definiteness, and the Hessian at a stationary point

The quadratic form of $H$ is $q(delta) = delta^top H delta = sum_i lambda_i z_i^2$. $H$ is _positive definite_ (PD) if all $lambda_i > 0$ ($q > 0$ for $delta != 0$), _positive semidefinite_ (PSD) if all $lambda_i >= 0$, _indefinite_ if it has both signs. At a stationary point ($nabla L = 0$), $L(theta^* + delta) = L(theta^*) + 1/2 delta^top H delta + O(delta^3)$, so: PD means a strict local minimum, indefinite a saddle, negative definite a maximum, and a zero eigenvalue means _degenerate_: the quadratic term is silent in that direction and higher-order terms decide ($x^4$ has a minimum at $0$, $x^3$ does not, both have zero second derivative). Quick $2 times 2$ test: $det H < 0$ means saddle; $det H > 0$ with $H_(11) > 0$ means PD.

#example("la-hess", [The anchor with weight decay at the origin])[
  The brief gives $nabla^2 L_gamma (0, 0) = mat(2 gamma, -1; -1, 2 gamma)$. By the pattern of §LA.1: $2 gamma - 1$ on $(1, 1)$ and $2 gamma + 1$ on $(1, -1)$. So the origin is a saddle for $gamma < 1\/2$ (escape along $a = b$), a strict minimum for $gamma > 1\/2$, and degenerate at $gamma = 1\/2$. A curve of exact minima, such as $a b = 1$ at $gamma = 0$, always has a zero eigenvalue along it (§FP.6).
]

#where[#pr("A2", "stationary", "(a)"), #pr("A2", "depth3", "(a)")\; #pr("A0", "hessian", "(b)").]

== LA.3 Trace, top eigenvalue, power iteration

$op("tr") H = sum_i H_(i i) = sum_i lambda_i$: total curvature. The _sharpness_ is $lambda_max$. For PSD $H$, $lambda_max <= op("tr") H$, with equality when $H$ has rank one ($H = v v^top$ has $lambda_max = op("tr") H = norm(v)^2$). Neither needs $H$ formed: a _Hessian-vector product_ $H v = nabla_theta (nabla L(theta)^top v)$ costs two backward passes, and then

#code[```python
v = torch.randn(d); v /= v.norm()
for _ in range(K): w = hvp(v); v = w / w.norm()
lam = v @ hvp(v)                      # Rayleigh quotient -> largest-|eigenvalue|
```]
Power iteration's error shrinks like $|lambda_2 \/ lambda_1|^k$ per step (the Rayleigh quotient's like the square of that). The trace comes from Hutchinson's estimator $op("tr") H = EE[z^top H z]$ for random $z$ with $EE[z z^top] = I$ (§PR.1).

#example("la-power", [Three steps by hand])[
  $H = mat(3, 1; 1, 3)$, $v_0 = (1, 0)$. $H v_0 = (3, 1)$, $v_1 = (3, 1)\/sqrt(10)$, Rayleigh $3.6$; $H v_1 prop (10, 6)$, $v_2 = (5, 3)\/sqrt(34)$, Rayleigh $132\/34 = 3.882$; next $3.969$. The gap to $4$ shrinks by $(2\/4)^2 = 1\/4$ per step.
]

#where[#pr("A0", "hessian", "(a)–(c)")\; A3 (sharpness), A4 (trace).]

== LA.4 SVD and the mode picture

Every real matrix factors as $M = U S V^top = sum_alpha s_alpha bold(u)_alpha bold(v)_alpha^top$ with orthonormal $bold(u)_alpha$ (outputs), $bold(v)_alpha$ (inputs) and $s_alpha >= 0$: $M bold(v)_alpha = s_alpha bold(u)_alpha$, $M^top bold(u)_alpha = s_alpha bold(v)_alpha$, $norm(M)_F^2 = sum s_alpha^2$. The $s_alpha^2$ are the eigenvalues of $M^top M$. Changing coordinates on both sides, $U^top M V = S$, makes $M$ diagonal. The _mode picture_: for linear regression with whitened inputs, $L(W) = 1/2 norm(W - Sigma_(y x))_F^2 + "const"$, and in the rotated coordinates $overline(W) = U^top W V$ the loss splits into one scalar problem per entry, $1/2 sum_alpha (overline(W)_(alpha alpha) - s_alpha)^2 + 1/2 sum_(alpha != beta) overline(W)_(alpha beta)^2$: independent modes, each with its own target $s_alpha$. Saxe et al. show that a two-layer network $W_2 W_1$ keeps this structure from an aligned start, and each mode becomes an anchor (A1).

#example("la-svd", [A permutation with gains])[
  $M = mat(0, 2; 1, 0)$: $M^top M = "diag"(1, 4)$, so $s_1 = 2$ with $bold(v)_1 = (0, 1)$, $bold(u)_1 = M bold(v)_1 \/ 2 = (1, 0)$, and $s_2 = 1$ with $bold(v)_2 = (1, 0)$, $bold(u)_2 = (0, 1)$. The map is "output 1 copies input 2 with gain 2" plus "output 2 copies input 1 with gain 1": two modes, the first stronger.
]

#where[#pr("A0", "modes", "(b)"), (e); #pr("A1", "saxe", "(a)–(d)")\; A6.]

== LA.5 Matrix calculus by perturbation

Use the Frobenius inner product $⟨A, B⟩ = op("tr")(A^top B) = sum_(i j) A_(i j) B_(i j)$, $norm(A)_F^2 = ⟨A, A⟩$. The gradient $nabla_W L$ is the matrix with
$ L(W + E) = L(W) + ⟨nabla_W L, E⟩ + O(norm(E)^2). $
*Recipe*: perturb, expand, keep the first-order term, and move $E$ to the right with $op("tr")(A B C) = op("tr")(C A B)$ and $op("tr")(A^top) = op("tr")(A)$ until it reads $op("tr")(K^top E)$; then $nabla_W L = K$. Rules that follow: $nabla_W ⟨A, W⟩ = A$; $nabla_W 1/2 norm(W)_F^2 = W$; $nabla_W 1/2 norm(A W B - Y)_F^2 = A^top (A W B - Y) B^top$. *Products*: if $L = ell(W_2 W_1)$ and $G = nabla_W ell$ at $W = W_2 W_1$, then perturbing $W_1$ by $E$ changes $W$ by $W_2 E$, and $⟨G, W_2 E⟩ = op("tr")(G^top W_2 E) = ⟨W_2^top G, E⟩$, so $nabla_(W_1) L = W_2^top G$; likewise $nabla_(W_2) L = G W_1^top$. Two more facts: $d\/d t (W W^top) = dot(W) W^top + W dot(W)^top$ (product rule, keep the order), and if $op("tr")(M K) = 0$ for every matrix $M$ then $K = 0$ (take $M$ with a single $1$).

#example("la-matcalc", [$nabla$ of $1/2 norm(Y - W_2 W_1 X)_F^2$])[
  Let $R = W_2 W_1 X - Y$. Perturb $W_2 -> W_2 + E$: $R -> R + E W_1 X$, and the change is $⟨R, E W_1 X⟩ = op("tr")(R^top E W_1 X) = op("tr")(W_1 X R^top E) = ⟨R (W_1 X)^top, E⟩$. Perturb $W_1 -> W_1 + E$: $⟨R, W_2 E X⟩ = op("tr")(X R^top W_2 E) = ⟨W_2^top R X^top, E⟩$. So
  $ nabla_(W_2) L = R X^top W_1^top, quad nabla_(W_1) L = W_2^top R X^top. $
  Shapes check: $W_2$ is $d_"out" times m$ and so is $R X^top W_1^top$. With a data average, $1/n X X^top = Sigma_(x x)$ and $1/n Y X^top = Sigma_(y x)$ appear.
]

#where[#pr("A0", "modes", "(a)")\; #pr("A1", "matrix-law", "(a)–(c)"), #pr("A1", "saxe", "(a)")\; #pr("A2", "balance", "(b)").]

== LA.6 Symmetries and differentiating identities

If $L(psi_alpha (theta)) = L(theta)$ for every $alpha$, with $psi_1 = "id"$, the function $g(alpha) = L(psi_alpha (theta))$ is constant, so by the chain rule
$ 0 = g'(1) = ⟨nabla L(theta), X(theta)⟩, quad X(theta) = partial/(partial alpha) psi_alpha (theta) |_(alpha = 1). $
The gradient is orthogonal to the generator $X$ everywhere. Under gradient flow any $Q$ with $nabla Q prop X$ is then conserved, $dot(Q) = -⟨nabla Q, nabla L⟩ = 0$. An identity that holds for _every_ $theta$ can be differentiated again in $theta$: the gradient of $⟨nabla L(theta), X(theta)⟩$ is
$ nabla^2 L(theta) X(theta) + (partial X \/ partial theta)^top nabla L(theta) = 0. $
The same trick computes $⟨nabla f, theta⟩$ for a _homogeneous_ $f$: if $f(alpha theta) = alpha^k f(theta)$, differentiating at $alpha = 1$ gives $⟨nabla f(theta), theta⟩ = k f(theta)$ (Euler's theorem); $f = a b$ has $k = 2$, $f = a b c$ has $k = 3$.

#example("la-sym", [Translation symmetry])[
  $L(theta_1, theta_2) = ell(theta_1 - theta_2)$ is unchanged by $psi_alpha (theta) = theta + (alpha - 1)(1, 1)$, generator $X = (1, 1)$. Then $⟨nabla L, X⟩ = ell' - ell' = 0$, so $theta_1 + theta_2$ (whose gradient is $X$) is conserved by gradient flow. Differentiating again, with $partial X \/ partial theta = 0$: $nabla^2 L (1, 1)^top = 0$ everywhere, a zero eigenvalue along the symmetry at every point, not only at minima.
]

#where[#pr("A1", "noether", "(a)–(b)"), #pr("A1", "matrix-law", "(b)"), #pr("A1", "broken", "(b)")\; #pr("A2", "balance", "(a)"), (c).]

#exercises("LA",
  [Eigenvalues and eigenvectors of $mat(2, -1, -1; -1, 2, -1; -1, -1, 2)$.],
  [Gradient of $L(W) = 1/2 op("tr")(W Sigma W^top) - op("tr")(W C^top)$ for symmetric $Sigma$.],
  [Power iteration on $H = "diag"(3, 1)$ from $v_0 = (1, 1)\/sqrt(2)$: $v_2$ and its Rayleigh quotient.],
  [For $L(a, b, c) = ell(a b c)$, write a one-parameter symmetry that rescales $a$ and $b$, its generator, and the quadratic quantity gradient flow conserves.],
)

= PR: probability reminders

== PR.1 Moments of a Gaussian (and of a coin)

For $z tilde cal(N)(0, sigma^2)$: $EE[z] = 0$, $EE[z^2] = sigma^2$, all odd moments vanish, $EE[z^4] = 3 sigma^4$ (in general $EE[z^(2 n)] = (2 n - 1)!! thin sigma^(2 n)$), so $"Var"(z^2) = 3 sigma^4 - sigma^4 = 2 sigma^4$. For independent variables, expectations of products factor: $EE[z^2 w^2] = EE[z^2] EE[w^2]$. A _Rademacher_ variable is $plus.minus 1$ with probability $1\/2$ each: $EE[z] = 0$, $z^2 = 1$ always. Hutchinson's trace estimator uses vectors of them: $EE[z^top H z] = sum_(i j) H_(i j) EE[z_i z_j] = op("tr") H$, with variance $2 sum_(i != j) H_(i j)^2$ (zero for a diagonal $H$).

#example("pr-gauss", [A fourth moment])[
  $a, b$ i.i.d. $cal(N)(0, sigma^2)$: $EE[(a^2 + b^2)^2] = EE[a^4] + 2 EE[a^2] EE[b^2] + EE[b^4] = 3 sigma^4 + 2 sigma^4 + 3 sigma^4 = 8 sigma^4$, so $"Var"(a^2 + b^2) = 8 sigma^4 - (2 sigma^2)^2 = 4 sigma^4$. (Monte Carlo at $sigma = 0.7$: $1.924$ against $8 sigma^4 = 1.921$.)
]

#where[#pr("A1", "noether", "(c)")\; #pr("A0", "hessian", "(a)"), (c) (Hutchinson).]

== PR.2 Variances, sample means, minibatches

$"Var"(X) = EE[X^2] - EE[X]^2$, so $EE[X^2] = "Var"(X) + EE[X]^2$; $"Var"(k X + b) = k^2 "Var"(X)$; for independent $X, Y$, $"Var"(X + Y) = "Var"(X) + "Var"(Y)$ and $"Var"(X Y) = EE[X^2] EE[Y^2] - EE[X]^2 EE[Y]^2$. The mean $overline(X)_B$ of $B$ i.i.d. draws with mean $m$ and variance $v$ has
$ EE[overline(X)_B] = m, quad "Var"(overline(X)_B) = v / B, quad EE[overline(X)_B^2] = m^2 + v / B. $
This is why minibatch gradient noise scales as $1\/B$ in variance ($1\/sqrt(B)$ in size): the batch gradient is a sample mean of per-example gradients, unbiased with covariance $Sigma_g \/ B$ (for draws with replacement). By the central limit theorem, for large $B$ it is approximately Gaussian, $overline(X)_B approx cal(N)(m, v\/B)$, whatever the distribution of one draw; its _standard error_ is $sqrt(v\/B)$. Useful for checks: a variance estimated from $n$ Gaussian samples has standard error about $v sqrt(2\/n)$.

#example("pr-batch", [Batch size and noise])[
  Per-example gradients (one coordinate) with mean $2$ and variance $9$. A batch of $B = 9$: mean $2$, standard deviation $sqrt(9\/9) = 1$, and $EE[g_B^2] = 4 + 1 = 5$ instead of $4$. In an update that is quadratic in the gradient (a step's $eta^2 g^2$ term), that extra $v\/B$ is the whole effect of the noise to leading order.
]

#where[#pr("A1", "broken", "(e)")\; #pr("A0", "ou", "(b)"), (d); A4.]

== PR.3 Brownian motion and Itô SDEs

_Brownian motion_ $W_t$: $W_0 = 0$, continuous paths, independent increments with $W_(t + h) - W_t tilde cal(N)(0, h)$. The variance of an increment equals its duration, so a typical increment is $sqrt(h)$, not $h$ (paths are continuous but nowhere differentiable). An _Itô SDE_
$ d X = mu(X) thin d t + sigma(X) thin d W $
reads "drift $mu$ plus noise of scale $sigma$" and means the $Delta t -> 0$ limit of the _Euler–Maruyama_ scheme $X_(k+1) = X_k + mu(X_k) Delta t + sigma(X_k) sqrt(Delta t) thin xi_k$, $xi_k tilde cal(N)(0, 1)$ independent. The $sqrt(Delta t)$ is forced: after $t\/Delta t$ steps the noise variance is $sigma^2 Delta t dot t \/ Delta t = sigma^2 t$, independent of $Delta t$; with $Delta t$ in its place the noise would vanish in the limit. Physicists write _Langevin dynamics_ as "gradient drift plus $sqrt(2 T)$ noise", $d theta = -nabla L(theta) d t + sqrt(2 T) thin d W$, with $T$ a temperature (§PR.5). *SGD as Euler–Maruyama*: $theta_(k+1) = theta_k - eta nabla L(theta_k) - eta xi_k$ with gradient noise $xi_k$ of variance $s^2$ is the scheme with $Delta t = eta$ and $sigma sqrt(eta) = eta s$, i.e. noise scale $sigma = s sqrt(eta)$: the noise in SGD is "small" only as $sqrt(eta)$.

#example("pr-bm", [Drift against diffusion])[
  $d X = m thin d t + sigma thin d W$ is solved by $X_t = X_0 + m t + sigma W_t$: mean $X_0 + m t$, variance $sigma^2 t$. The drift grows like $t$, the spread like $sqrt(t)$, so drift dominates after $t approx sigma^2 \/ m^2$. With $m = 0.01$ and $sigma = 0.1$, that is $t approx 100$.
]

#where[#pr("A0", "ou", "(a)"), (f); A4 (drift along the manifold of minima).]

== PR.4 The Ornstein–Uhlenbeck process: a worked derivation

$ d X = -theta X thin d t + sigma thin d W, quad X_0 = x_0. $
A pull toward $0$ at rate $theta$ (a rate here, not a parameter vector) against constant kicks. *Mean.* Take expectations in the Euler–Maruyama step $X_(k+1) = (1 - theta Delta t) X_k + sigma sqrt(Delta t) xi_k$: the noise has mean zero, so $m_(k+1) = (1 - theta Delta t) m_k$, i.e. $dot(m) = -theta m$ and $m(t) = x_0 e^(-theta t)$ (§ODE.3). *Variance.* $X_k$ and $xi_k$ are independent, so variances add (§PR.2):
$ v_(k+1) = (1 - theta Delta t)^2 v_k + sigma^2 Delta t. $
Divide $v_(k+1) - v_k = (-2 theta Delta t + theta^2 Delta t^2) v_k + sigma^2 Delta t$ by $Delta t$ and let $Delta t -> 0$: $dot(v) = -2 theta v + sigma^2$, a linear ODE with fixed point $sigma^2 \/ (2 theta)$ (§ODE.3, or variation of constants, §ODE.5). From $v(0) = 0$:
$ v(t) = sigma^2 / (2 theta) (1 - e^(-2 theta t)), quad v_infinity = sigma^2 / (2 theta). $
The stationary law is $cal(N)(0, sigma^2\/(2 theta))$: the balance between kicks ($sigma^2$ per unit time) and the pull (variance removed at rate $2 theta$). The variance relaxes twice as fast as the mean. Numbers: $theta = 1$, $sigma = 0.5$ gives $v_infinity = 0.125$; $20000$ Euler–Maruyama paths with $Delta t = 10^(-3)$ give $0.1256$ at $t = 8$ (sampling error $0.0013$). The _exact_ fixed point of the discrete recursion differs at $O(Delta t)$; finding it is A0’s job.

#where[#pr("A0", "ou", "(a)–(b)"), (f); A4 (the transverse fluctuation around $a b = 1$).]

== PR.5 Stationary distributions $prop exp(-L\/T)$

Langevin dynamics $d theta = -nabla L(theta) d t + sqrt(2 T) d W$ has the stationary density
$ p(theta) prop exp(-L(theta) \/ T) $
(the Gibbs distribution) whenever it is normalizable. Check on a quadratic: $L = lambda x^2 \/ 2$ gives OU with $theta = lambda$, $sigma^2 = 2 T$, so $v_infinity = 2 T \/ (2 lambda) = T \/ lambda$, and $exp(-lambda x^2 \/ (2 T))$ is exactly $cal(N)(0, T\/lambda)$. Consequences: each quadratic direction holds average loss $EE[lambda x^2\/2] = T\/2$ (equipartition), and flat directions spread more ($T\/lambda$). Two cautions for SGD: its noise is neither isotropic nor constant (the covariance depends on $theta$, and for the anchor ordinary minibatch noise vanishes at every exact minimum), so its stationary law is generally _not_ Gibbs; and on a manifold of minima what matters is how the noise interacts with the changing curvature, which A4 works out.

#where[A4; #pr("A0", "ou", "(f)").]

#exercises("PR",
  [$z, w$ i.i.d. $cal(N)(0, sigma^2)$: $"Var"(z^2)$ and $EE[z^2 w^2]$.],
  [Per-example gradient noise has variance $4$. Standard deviation of the batch mean at $B = 16$? Which $B$ gives $0.1$?],
  [OU with $theta = 2$, $sigma = 1$ from $X_0 = 0$: $v_infinity$, and the time for $v(t)$ to reach $90%$ of it.],
  [Langevin on $L(x, y) = 1/2 (4 x^2 + y^2)$ at $T = 0.01$: the stationary variances of $x$ and $y$.],
)

= ALG: algebra you keep doing

== ALG.1 Two equations: a difference of squares and a product

Given $a^2 - b^2 = c$ and $a b = s$: set $A = a^2$, $B = b^2$, so $A - B = c$ and $A B = s^2$. Then $A^2 - c A - s^2 = 0$, and since $A >= 0$,
$ a^2 = (c + sqrt(c^2 + 4 s^2)) / 2, quad b^2 = (-c + sqrt(c^2 + 4 s^2)) / 2, quad a^2 + b^2 = sqrt(c^2 + 4 s^2). $
The last identity is $(a^2 + b^2)^2 = (a^2 - b^2)^2 + 4 a^2 b^2$ and is worth remembering on its own. Signs are a separate decision: $a b = s > 0$ forces $a, b$ to share a sign, and which one is decided by something else (for a flow, by what cannot change sign along it, §ODE.1).

#example("alg-endpoint", [Numbers])[
  $c = 2$, $s = 1$: $a^2 = (2 + sqrt(8))\/2 = 1 + sqrt(2) = 2.414$, so $(a, b) = (1.5538, 0.6436)$ on the positive branch; check $a b = 1$, $a^2 - b^2 = 2$, $a^2 + b^2 = sqrt(8)$.
]

#where[#pr("A1", "endpoint", "(a)"), #pr("A1", "noether", "(c)")\; A5 (the anchor kernel $a^2 + b^2 = sqrt(c^2 + 4 p^2)$).]

== ALG.2 Polynomial conditions and critical parameters

Stationarity conditions often reduce to one polynomial equation in a single unknown $u$ with a parameter $gamma$ in it. When $gamma$ enters linearly, solve for it: $gamma = g(u)$. The roots for a given $gamma$ are the intersections of the horizontal line at height $gamma$ with the graph of $g$, so the number of roots changes exactly when $gamma$ passes an extreme value of $g$: the _critical parameter_ is $gamma^* = max_u g(u)$, found from $g'(u^*) = 0$, and the two roots merge at $u^*$ (a saddle-node, §FP.5). Two more tools: _Descartes' rule of signs_ (the number of positive roots is the number of sign changes in the coefficients, or less by an even number) and the _rational root theorem_ (a rational root $p\/q$ in lowest terms has $p$ dividing the constant term and $q$ the leading coefficient; find one, divide it out, factor the quadratic).

#example("alg-crit", [A cubic condition])[
  $u^3 - u + 2 gamma = 0$ for $u > 0$ is $gamma = g(u) = u (1 - u^2) \/ 2$. $g(0) = 0$, $g$ rises, then falls to $0$ at $u = 1$; $g'(u) = (1 - 3 u^2)\/2 = 0$ at $u^* = 1\/sqrt(3)$, so $gamma^* = g(u^*) = 1\/(3 sqrt(3)) = 0.1925$. Two positive roots for $0 < gamma < gamma^*$, one double root at $gamma^*$, none above; Descartes agrees (signs $+, -, +$: two changes, so $2$ or $0$). Factoring: $2 x^3 - 3 x^2 - 3 x + 2$ has candidate roots $plus.minus 1, plus.minus 2, plus.minus 1\/2$; $x = 2$ works, and dividing leaves $2 x^2 + x - 1 = (2 x - 1)(x + 1)$.
]

#where[#pr("A2", "depth3", "(b)"), (d); #pr("A2", "basins", "(a)")\; #pr("A2", "finite-step", "(c)").]

== ALG.3 Integrals you meet

*Partial fractions*: $1/(x (s - x)) = 1/s (1/x + 1/(s - x))$, so $integral d x \/ (x (s - x)) = 1/s ln(x \/ (s - x))$. *Standard forms*: $integral d x \/ sqrt(c^2 + x^2) = "arcsinh"(x\/|c|) = ln(x + sqrt(x^2 + c^2)) + "const"$, and $integral x thin d x \/ sqrt(c^2 + x^2) = sqrt(c^2 + x^2)$; with $x = 2 p$ these give $integral d p \/ sqrt(c^2 + 4 p^2) = 1/2 "arcsinh"(2 p \/ |c|)$. *Integrating along a trajectory*: if $p(t)$ moves monotonically from $p_0$ to $p_infinity$, then
$ integral_0^infinity h(p(t)) thin d t = integral_(p_0)^(p_infinity) h(p) / (dot(p)(p)) thin d p, $
which turns a time integral you cannot do (you do not know $p(t)$) into an integral over $p$ you can.

#example("alg-int", [Accumulated squared distance])[
  Along $dot(x) = -lambda x$ from $x_0 > 0$: $integral_0^infinity x^2 d t = integral_(x_0)^0 x^2 \/ (-lambda x) thin d x = integral_0^(x_0) x \/ lambda thin d x = x_0^2 \/ (2 lambda)$, the same as integrating $x_0^2 e^(-2 lambda t)$ directly, but the second route needs no solution.
]

#where[#pr("A1", "logistic", "(a)")\; #pr("A1", "broken", "(c)–(d)").]

== ALG.4 Log-log and semi-log plots

A power law $y = C x^k$ is a straight line on log-log axes, $log y = log C + k log x$, with slope $k$; an exponential $y = K e^(-x\/tau)$ is straight on semi-log axes ($log y$ against $x$), slope $-1\/tau$. To read an exponent: fit `np.polyfit(np.log(x), np.log(y), 1)` over the range where the asymptotic law should hold, and also look at the _local slope_ $d ln y \/ d ln x$ between neighbouring points, which should settle to $k$. Logarithmic factors ($y = x^(-1) ln(K\/x)$) bend the local slope by $O(1\/ln)$ and never quite settle: they do not change the exponent, but at finite range you will read $-0.95$, not $-1$. The two slopes the course measures most: $2$ for a one-step error of $O(eta^2)$, $1$ for an accumulated error of $O(eta)$.

#example("alg-slope", [Reading an exponent])[
  $y = 250, 25, 2.5$ at $x = 10^(-3), 10^(-2), 10^(-1)$: each decade in $x$ divides $y$ by $10$, slope $-1$, and $y x = 0.25$, so $y = 0.25 thin x^(-1)$.
]

#where[every Check; #pr("A0", "flow-limit", "(c)")\; #pr("A1", "logistic", "(d)")\; #pr("A2", "slowing", "(f)").]

== ALG.5 Scaling arguments: what "$prop 1\/|gamma - 1/2|$" means

"$tau prop 1\/|gamma - 1/2|$ as $gamma -> 1/2$" means $tau |gamma - 1/2| -> C$, a constant, in that limit; it says nothing about $tau$ far from $1/2$, and $C$ may differ on the two sides. To check one numerically: compute or measure $tau$ at $|gamma - 1/2| = 10^(-1), 10^(-2), 10^(-3)$ on each side; the local log-log slopes should approach $-1$ and $tau |gamma - 1/2|$ should approach $C$. *Time scales from rescaling.* In $dot(u) = 2 u (s - u)$, set $u = s v$ and $t' = s t$: then $d v \/ d t' = 2 v (1 - v)$, with no $s$ left. So every time in the problem is $(1\/s) F(u_0 \/ s)$ for one function $F$: the $1\/s$ is known before solving anything. *Dominant balance.* In an equation with a small parameter, decide which two terms are of the same size and drop the rest (§TAY.4); then check that the dropped terms really are smaller.

#example("alg-scale", [A critical exponent from data])[
  $tau = 52, 502, 5002$ at $delta = 10^(-2), 10^(-3), 10^(-4)$. Local slopes $-0.985$, $-0.998$: tending to $-1$. $tau delta = 0.52, 0.502, 0.5002 -> C = 0.5$. So $tau approx 0.5\/delta + 2$: the constant $2$ is a correction that the limit ignores.
]

#where[#pr("A2", "slowing", "(b)"), (d), (f), #pr("A2", "depth3", "(g)") (the slope $4\/3$ in $s$); #pr("A1", "logistic", "(b)"), #pr("A1", "saxe", "(b)").]

#exercises("ALG",
  [Solve $a^2 - b^2 = -1.5$, $a b = 1$ on the positive branch.],
  [For which $gamma > 0$ does $u^2 (1 - u) = gamma$ have positive roots, and how many?],
  [Measured $tau = 3.2, 10, 31.6$ at $delta = 10^(-1), 10^(-2), 10^(-3)$. Exponent and constant?],
  [Compute $integral_0^infinity x(t)^2 d t$ along $dot(x) = -x (1 + x^2)$ from $x_0 > 0$, without solving the ODE.],
)

= AHEAD: looking ahead

No exercises here: each later assignment builds these from scratch. They are here so the words are not new when you meet them.

== AHEAD.1 Backward error analysis (A3)

GD tracks gradient flow only to $O(eta)$ (§GD.1). _Backward error analysis_ asks the reverse question: which modified ODE $dot(theta) = -nabla tilde(L)(theta)$ do the GD iterates follow to _higher_ order? Expand the modified flow over one time step $eta$ with §TAY.1, match it term by term with the GD step, and solve for the correction. The answer to first order is $tilde(L) = L + eta/4 norm(nabla L)^2$ (Barrett and Dherin): GD implicitly penalizes large gradients. A1 already uses the resulting modified flow; A3 derives it.

== AHEAD.2 Linearization in parameters: the kernel view (A5, A6)

Linearize the network's output in its parameters around $theta_0$: $f(x; theta) approx f(x; theta_0) + nabla_theta f(x; theta_0)^top (theta - theta_0)$. Under gradient flow on squared loss the predictions then obey a _linear_ ODE in function space, $dot(f)(x) = -1/n sum_i K(x, x_i)(f(x_i) - y_i)$ with the _tangent kernel_ $K(x, x') = nabla_theta f(x)^top nabla_theta f(x')$. If $K$ stays fixed, §ODE.6 solves it: the eigenvectors of the kernel matrix are learned at rates given by its eigenvalues. Whether $K$ stays fixed (lazy) or moves (rich) is A5’s question; for the anchor $K = (a^2 + b^2) x x'$ and $a^2 + b^2 = sqrt(c^2 + 4 p^2)$ (§ALG.1). A6 uses the same object per example and per step.

== AHEAD.3 Fokker–Planck: the ODE for a density (A4)

An SDE moves individual paths; the _Fokker–Planck equation_ moves their density. For $d X = mu(X) d t + sigma d W$ in one dimension, $partial_t p = -partial_x (mu p) + sigma^2/2 thin partial_x^2 p$: a linear PDE, drift transports the density, noise diffuses it. Setting $partial_t p = 0$ with $mu = -L'$ and $sigma^2 = 2 T$ gives $p prop e^(-L\/T)$ (§PR.5). A4 does not solve it; it uses the OU picture (§PR.4) across the manifold of minima and a slow drift along it.

== AHEAD.4 What muP's coordinate check checks (A7)

Train a network for a few steps at widths $m = 64, dots, 2048$ and plot the typical size of each layer's activation change (the root-mean-square of the entries of $Delta h$) against $m$ on log-log axes (§ALG.4). If the curve is flat in $m$, features keep moving at every width; if it falls like $m^(-1\/2)$, a wide network stops learning features (the kernel regime of AHEAD.2); if it grows, training will blow up at large width. The arithmetic behind the slopes is §PR.2’s: a sum of $m$ independent terms grows like $sqrt(m)$, a sum of $m$ aligned terms like $m$, and a parameterization sets init scales and learning rates per layer so that the update's contribution is of order one. A7 works it out for a two-layer network by hand.

= Answers

#let ans(id, body) = block(above: 0.55em, below: 0.55em)[#text(weight: "bold", id) #h(0.4em) #body]

== ODE

#ans("ODE-E1")[Separable: $integral_1^x d x \/ (-x^3) = 1/2 (1\/x^2 - 1) = t$, so $x = 1\/sqrt(1 + 2 t)$; $x = 0.1$ at $t = 49.5$.]
#ans("ODE-E2")[$t = 1/4 ln (1 dot (2 - 0.02)) / (0.02 dot (2 - 1)) = 1/4 ln 99 = 1.149$.]
#ans("ODE-E3")[$x = 2 (1 - e^(-2 t))$; time constant $1\/2$; $2 e^(-2 t) = 0.01$ at $t = 1/2 ln 200 = 2.649$.]
#ans("ODE-E4")[$A = mat(-1, 1; 1, -1)$: eigenvalue $0$ on $(1, 1)$, $-2$ on $(1, -1)$. So $x + y$ is conserved, $x - y = (x_0 - y_0) e^(-2 t)$, and the state ends at $x = y = (x_0 + y_0)\/2$, on the line of fixed points $x = y$.]

== FP

#ans("FP-E1")[$x^* = 0$ ($F'(0) = 1$, unstable) and $plus.minus 1$ ($F' = 1 - 3 = -2$, stable, $tau = 1\/2$). The pitchfork of §FP.5 at $mu = 1$.]
#ans("FP-E2")[$H = mat(2, 3; 3, 2)$: $5$ on $(1, 1)$, $-1$ on $(1, -1)$; $J = -H$ has $-5$ and $+1$: a saddle, leaving along $(1, -1)$.]
#ans("FP-E3")[$x^* = 0.1$, $F'(x^*) = -0.2$, $tau = 5$. In general $tau = 1\/(2 sqrt(mu)) prop mu^(-1\/2)$.]
#ans("FP-E4")[$nabla L = (x + y - 1)(1, 1) = 0$ on the line. $J = -mat(1, 1; 1, 1)$: $0$ on $(1, -1)$, the tangent; $-2$ on $(1, 1)$.]

== GD

#ans("GD-E1")[$lambda = 4$, $1 - eta lambda = -0.2$: $theta_k = (-0.2)^k theta_0$, converging with sign flips. Stable iff $eta < 2\/4 = 0.5$.]
#ans("GD-E2")[$0.5^4 = 0.0625$ against $e^(-2) = 0.1353$: at $h = 0.5$ the first-order method is off by a factor of two.]
#ans("GD-E3")[$0 < eta < 2\/4 = 0.5$; the zero eigenvalue is neutral (§GD.3).]
#ans("GD-E4")[Fixed points $0, plus.minus 2$. $f'(2) = 1 - eta (3 dot 4 - 4) = 1 - 8 eta$: stable for $0 < eta < 1\/4$ (alternating above $1\/8$).]

== TAY

#ans("TAY-E1")[$2 + c^2\/4$ and $1/2 - c^2\/16$.]
#ans("TAY-E2")[$1 - 4 eta gamma + O(eta^2)$: the $eta^2 r^2$ term is second order.]
#ans("TAY-E3")[$p' = (a - eta b r)(b - eta a r) = p - eta r (a^2 + b^2) + eta^2 r^2 p$, so $r' = r thin [1 - eta (a^2 + b^2) + eta^2 r p]$. Near a minimum ($r$ small) the residual is multiplied by about $1 - eta (a^2 + b^2)$: §GD.2 with curvature $a^2 + b^2$.]
#ans("TAY-E4")[$x approx epsilon\/4 + epsilon^3\/256$ (at $epsilon = 0.1$: $0.0250039$ both exactly and from the series).]

== LA

#ans("LA-E1")[$alpha = 2$, $beta = -1$, $n = 3$: $0$ on $(1, 1, 1)$, and $3$ twice, on the plane $x + y + z = 0$.]
#ans("LA-E2")[$W Sigma - C$ (from $op("tr")(E Sigma W^top) = ⟨W Sigma, E⟩$ twice, halved, and $op("tr")(E C^top) = ⟨C, E⟩$).]
#ans("LA-E3")[$v_k prop (3^k, 1)$, so $v_2 = (9, 1)\/sqrt(82)$, Rayleigh quotient $(3 dot 81 + 1)\/82 = 2.976$.]
#ans("LA-E4")[$psi_alpha (a, b, c) = (alpha a, b\/alpha, c)$, generator $(a, -b, 0)$, conserved $a^2 - b^2$ (and $b^2 - c^2$ from the other pair).]

== PR

#ans("PR-E1")[$"Var"(z^2) = 3 sigma^4 - sigma^4 = 2 sigma^4$; $EE[z^2 w^2] = sigma^4$.]
#ans("PR-E2")[$sqrt(4\/16) = 0.5$; $sqrt(4\/B) = 0.1$ at $B = 400$.]
#ans("PR-E3")[$v_infinity = 1\/(2 dot 2) = 0.25$; $1 - e^(-4 t) = 0.9$ at $t = ln 10 \/ 4 = 0.576$.]
#ans("PR-E4")[$T\/lambda$: $0.01\/4 = 0.0025$ for $x$, $0.01$ for $y$.]

== ALG

#ans("ALG-E1")[$a^2 = (-1.5 + sqrt(2.25 + 4))\/2 = 0.5$, $b^2 = 2$: $(a, b) = (0.7071, 1.4142)$.]
#ans("ALG-E2")[$g(u) = u^2 - u^3$, $g' = 2 u - 3 u^2 = 0$ at $u = 2\/3$, $g(2\/3) = 4\/27 = 0.148$. Two positive roots for $0 < gamma < 4\/27$, a double root at $4\/27$, none above.]
#ans("ALG-E3")[Each decade multiplies $tau$ by $10^(1\/2) = 3.16$: $tau approx 1.0 thin delta^(-1\/2)$, exponent $-1\/2$ (the saddle-node law of §FP.5), constant $1.0$.]
#ans("ALG-E4")[$d t = -d x \/ (x (1 + x^2))$, so $integral_0^infinity x^2 d t = integral_0^(x_0) x \/ (1 + x^2) d x = 1/2 ln(1 + x_0^2)$; $0.3466$ at $x_0 = 1$, as numerical integration confirms.]
