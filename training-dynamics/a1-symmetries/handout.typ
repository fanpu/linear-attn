#import "/template/template.typ": *

#show: doc.with(
  assignment: 1,
  title: "Symmetries and conservation laws",
  kind: "Assignment handout",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "about 5 hours · CPU only · all six checks together run in about 15 seconds",
)

// The fixed test sentence, for a problem whose tests live in one file with several adapters.
#let filetest(file, adapters) = [
  To test your implementation, run #raw("uv run pytest a1-symmetries -k " + file) (the adapters
  #adapters.map(a => raw("run_" + a)).join(", ") in #raw("tests/adapters.py") are pre-wired to these
  signatures; edit them only if yours differ).
]

= Overview

Gradient flow on the anchor $f_(a,b)(x) = a b x$, $L = 1/2 (a b - 1)^2$, never leaves the curve $a^2 - b^2 = "const"$ (A0). This assignment asks _why_, how far the reason reaches, and what it buys. The why is a symmetry: the loss is unchanged when one layer is scaled up and the other down. What it buys is a solved model: the endpoint, the learning curve and the time to learn have closed forms, and the same reduction solves a two-layer linear network with matrices, mode by mode. Real training breaks the law (finite steps, momentum, minibatches, weight decay), and _modified gradient flow_ predicts by how much. A0 supplied the flow, the phase portrait and the one-layer mode decomposition; A2 (weight decay), A3 (finite steps) and A5 (lazy vs rich) start from this assignment's conserved quantities.

==== What you will derive
+ The Noether-style constraint on the gradient and the conserved quantity it implies, for any loss of $a b$ (Problem (#raw("noether")))
+ The exact endpoint of the flow on $a b = s$ from the conserved imbalance (Problem (#raw("endpoint")))
+ The logistic learning curve of a balanced mode and the time to learn it (Problem (#raw("logistic")))
+ The matrix law $d/(d t)(W_1 W_1^top - W_2^top W_2) = 0$ and its exact one-step violation under GD (Problem (#raw("matrix-law")))
+ Saxe's decomposition of a two-layer linear network into independent anchors (Problem (#raw("saxe")))
+ How GD, momentum, minibatch SGD and weight decay change $c$, to first order in $eta$ (Problem (#raw("broken")))

==== What you will implement
+ `noether.py`, `endpoint.py`, `logistic.py` in `symmetries/`: the symmetry, the endpoint, the learning curve (Problems (#raw("noether"))–(#raw("logistic")))
+ `two_layer.py`, `saxe.py`: gradients and the matrix law, Saxe's prediction (Problems (#raw("matrix-law")), (#raw("saxe")))
+ `drift.py`: the exact and first-order breaking of the law (Problem (#raw("broken")))

==== What you will run
+ Six check scripts, one per problem, each comparing a simulation with your derivation
+ A short reading of Simon et al. on where conservation laws sit in a theory of deep learning (Problem (#raw("framing")))

#runin[Setup.] From the course root: `uv run pytest a1-symmetries` (all 48 tests fail with `NotImplementedError` until you fill in the stubs). Checks run as `uv run python a1-symmetries/scripts/check_<name>.py` and write PNGs and their section of `a1-symmetries/out/results.json`; compare with `uv run python -m tdyn.results a1-symmetries/out/results.json a1-symmetries/solutions/results.json`. Three checks have a `FORMULAS` block at the top where you type a derived formula. Worked solutions: `solutions/solutions.pdf`. #runin[Time box.] If a Derive part takes more than 15 minutes with all hints, read that part in `solutions/solutions.pdf`, close it, and redo the part from a blank page before moving on. The solutions are part of the method, not a failure. Rusty on the math? Parts point into the primer `m0-math-primer/handout.pdf` at the point of use, as in #primer("ODE.4", "logistic"); its first page is a one-page self-diagnostic.

#runin[Notation.] As in A0: $theta = (a, b)$, residual $r = a b - 1$ (or $a b - s$ for a target $s > 0$), product $p = a b$, imbalance $c = a^2 - b^2$, step size $eta$, weight decay $gamma$ in the convention $L_gamma = L + gamma (a^2 + b^2)$, momentum $beta$. $⟨u, v⟩$ is the dot product.

= Symmetry and conservation

A _continuous symmetry_ of a loss $L(theta)$ is a family of maps $psi_alpha$, indexed by a real number $alpha$ with $psi_1$ the identity, such that $L(psi_alpha (theta)) = L(theta)$ for every $theta$ and $alpha$. Its _generator_ is the velocity of $psi_alpha (theta)$ as $alpha$ leaves the identity,
$ X(theta) = partial/(partial alpha) psi_alpha (theta) |_(alpha = 1), $ <eq-gen>
a vector field on parameter space. A _conserved quantity_ (or _constant of motion_) is a function $Q(theta)$ that stays constant along every trajectory. Noether's theorem in physics says each continuous symmetry of a Lagrangian gives a conserved quantity; Kunin et al. prove the gradient-flow analogue for neural networks @kunin2021mechanics[§3–4, Theorem 1]: differentiate the symmetry at the identity to get a constraint on the gradient, then project the flow onto the generator. The anchor's symmetry is the _rescaling_
$ psi_alpha (a, b) = (alpha a, b / alpha), $ <eq-sym>
which leaves $a b$, so the network function and every loss built from it, unchanged; for ReLU networks the same map acts on each hidden unit's incoming and outgoing weights @ang2026post[§2.1] @du2018balanced[Theorem 2.1].

#example("scale-by-hand", [The same recipe for a different symmetry])[
  $L(theta) = -theta_2 \/ norm(theta)$ on $RR^2$ depends only on the direction of $theta$ (like weights feeding a normalization layer). Its symmetry is _scaling_, $psi_alpha (theta) = alpha theta$, with generator $X(theta) = theta$. Differentiating $L(alpha theta) = L(theta)$ at $alpha = 1$ gives $⟨nabla L(theta), theta⟩ = 0$: the gradient is always tangent to the circle through $theta$. Under gradient flow, $d/(d t) norm(theta)^2 = 2 ⟨theta, dot(theta)⟩ = -2 ⟨theta, nabla L⟩ = 0$, so $norm(theta)^2$ is conserved. One GD step tells another story. At $theta = (1, 0)$: $nabla L = (0, -1)$; with $eta = 0.5$, $theta' = (1, 0.5)$ and $norm(theta')^2 = 1.25$. Because the step is orthogonal to $theta$, Pythagoras gives $norm(theta')^2 = norm(theta)^2 + eta^2 norm(nabla L)^2$ exactly: every finite step _increases_ the conserved quantity, by an amount of order $eta^2$ @kunin2021mechanics[§6, "scale dynamics"].
]

#problem("noether", [Rescaling symmetry and the conserved imbalance], kind: ("derive", "implement", "check"), time: "40 min", owner: "you")[
  #parts(
    [#tag("derive") Differentiate $L(psi_alpha (theta)) = L(theta)$ in $alpha$ at $alpha = 1$ for @eq-sym #primer("LA.6", "symmetries"). Use the constraint to find a quadratic $Q(a, b)$ with $dot(Q) = 0$ along gradient flow, and state for which losses $L(a, b)$ your argument holds (it should cover any per-example loss of $f_(a,b)$ on any dataset).
     #deliverable[the generator $X(a, b)$; the constraint, as the equation $⟨nabla L(theta), X(theta)⟩ = dots$; $Q(a, b)$; the class of losses, stated as a condition on $L$ (a form $L(a, b) = dots$ that guarantees the symmetry).]
     #hints([Chain rule on $alpha |-> L(psi_alpha (theta))$ at $alpha = 1$ (primer §LA.6).], [Look for $Q$ with $nabla Q prop X$: then $dot(Q) = ⟨nabla Q, dot(theta)⟩ = -⟨nabla Q, nabla L⟩$ vanishes by the constraint.], [First line of the solution: "The generator of @eq-sym is $X(a, b) = partial_alpha (alpha a, b\/alpha)|_(alpha = 1) = (a, -b)$."])],
    [#tag("derive") Differentiate your constraint once more with respect to $theta$ (Kunin et al. do this in §3) #primer("LA.6", "differentiating an identity"). Deduce what $nabla^2 L (theta) X(theta)$ is at any minimum, and connect it to the zero eigenvalue you found in A0’s Problem (#raw("phase-portrait")).
     #deliverable[the identity in the form $nabla^2 L(theta) X(theta) = M nabla L(theta)$ with $M$ a constant $2 times 2$ matrix; the sentence "at a minimum, $X(theta)$ is an eigenvector of $nabla^2 L$ with eigenvalue $dots$", and which eigenvector of A0’s Problem (#raw("phase-portrait")) (b) it is.]
     #hints([The constraint holds for every $theta$, so its gradient in $theta$ vanishes too: product rule on $⟨nabla L(theta), X(theta)⟩$ (primer §LA.6).], [$nabla_theta ⟨nabla L, X⟩ = nabla^2 L thin X + (partial X \/ partial theta)^top nabla L$, and here $partial X \/ partial theta$ is a constant diagonal matrix.], [First line of the solution: "Differentiate the constraint $⟨nabla L(theta), X(theta)⟩ = 0$ with respect to $theta$."])],
    [#tag("derive") _Small initialization gives balanced layers_ @du2018balanced[§2.1, after Corollary 2.1]. For $a_0, b_0$ i.i.d. $cal(N)(0, sigma^2)$, compute $sqrt(EE[c^2])$ #primer("PR.1", "Gaussian moments"). Then use $a_infinity^2 - b_infinity^2 = c$ to find the gap $a_infinity - b_infinity$ to leading order in small $c$ on the positive branch of $a b = 1$.
     #deliverable[$sqrt(EE[c^2])$ as a function of $sigma$; $a_infinity - b_infinity$ to first order in $c$. Type both into the `FORMULAS` block of `scripts/check_noether.py`.]
     #hints([Expand $c^2$ and use Gaussian moments, $EE[a^4] = 3 sigma^4$, and independence (primer §PR.1).], [Factor $a_infinity^2 - b_infinity^2 = (a_infinity - b_infinity)(a_infinity + b_infinity)$ and find $a_infinity + b_infinity$ to zeroth order in $c$ (primer §TAY.4, §ALG.1).], [First line of the solution: "With $a_0, b_0$ i.i.d. $cal(N)(0, sigma^2)$, $EE[a^4] = 3 sigma^4$, so $EE[c^2] = EE[a^4] - 2 EE[a^2] EE[b^2] + EE[b^4]$."])],
    [#tag("implement") In `symmetries/noether.py`, for arrays of points of shape `(..., 2)`:
     #defline[`def rescale(theta, alpha)`  ·  `def generator(theta)`  ·  `def conserved(theta)`]
     #arglist(
       ("rescale", [$psi_alpha (theta)$ by @eq-sym.]),
       ("generator", [$X(theta)$ by @eq-gen, your (a).]),
       ("conserved", [your $Q(theta)$, shape `(...)`.]),
     )
     #filetest("test_noether", ("rescale", "generator", "conserved"))],
    [#tag("check") Run `scripts/check_noether.py`. Expected: $|⟨nabla L, X⟩|$ at round-off for three losses of $a b$, order one for $f = (a b + 0.3 a) x$ (a skip connection); your $Q$ constant to $10^(-9)$ along those three flows, not along the skip-connection flow or GD at $eta = 0.1$; the Hessian identity at round-off; `noether.png` (right) on your formulas for $sigma <= 0.3$.],
  )
]

= Where the flow lands, and how fast

#problem("endpoint", [Where gradient flow lands], kind: ("derive", "implement", "check"), time: "25 min", owner: "you")[
  Gradient flow on $L = 1/2 (s - a b)^2$ with target $s > 0$ ends on the curve $a b = s$ unless it ends at the origin (A0). Ang's post does $s = 1$ @ang2026post[§2.1].
  #parts(
    [#tag("derive") Combine conservation with $a_infinity b_infinity = s$ to find $a_infinity^2$ and $b_infinity^2$ #primer("ALG.1", "two equations"). Decide which branch of $a b = s$ the flow reaches from $(a_0, b_0)$, including $c = 0$, and argue why the sign of $a$ (if $c > 0$) or of $b$ (if $c < 0$) cannot change.
     #deliverable[$(a_infinity, b_infinity)$ in closed form for every start, with one sign rule that covers all cases.]
     #hints([Two equations in $a_infinity^2, b_infinity^2$: their difference is $c$, their product $s^2$; a quadratic in $A = a_infinity^2$ (primer §ALG.1).], [For the signs: if $c > 0$ then $a^2 = c + b^2 >= c > 0$ for all $t$, so $a$ never passes through zero (primer §ODE.1).], [First line of the solution: "Write $A = a_infinity^2$. Conservation gives $b_infinity^2 = A - c$ and the endpoint gives $a_infinity^2 b_infinity^2 = s^2$, so $A^2 - c A - s^2 = 0$."])],
    [#tag("implement") In `symmetries/endpoint.py`:
     #defline[`def flow_endpoint(a0, b0, s=1.0) -> np.ndarray`]
     #arglist(("returns", [$(a_infinity, b_infinity)$, shape `(2,)`; the origin for $a_0 = -b_0$.]))
     #filetest("test_endpoint", ("flow_endpoint",))],
    [#tag("check") Run `scripts/check_endpoint.py`: 450 random starts and $s in {0.5, 1, 2}$, integrated to $t = 60$. Expected: your endpoints within $10^(-9)$ of the integrated ones (`endpoint.png`). It also starts at distance $epsilon$ from the line $a = -b$; note in `notes.md` how the time to leave the origin grows with $epsilon$ (A0’s saddle eigenvalues explain it).],
  )
]

On the line $a = b > 0$ (so $c = 0$), $u = a b = a^2$ is the only variable left. Saxe, McClelland and Ganguli solve exactly this one-dimensional problem @saxe2014exact[§1.3, eqs. (8)–(12)]; it is the learning curve of a single mode of a deep linear network (Problem (#raw("saxe"))). Define the _time to learn_ $t_f (u_0, s)$ as the first time at which $u$ reaches the fraction $f$ of its target,
$ u(t_f) = f s, quad 0 < u_0 < f s, quad t_(1\/2) = t_f "at" f = 1/2. $ <eq-thalf>

#problem("logistic", [Balanced learning is logistic], kind: ("derive", "implement", "check", "predict"), time: "50 min", owner: "you")[
  #parts(
    [#tag("derive") Show that $a_0 = b_0 > 0$ implies $a(t) = b(t)$ for all $t$, and derive a closed ODE for $u = a b$. It is the _logistic equation_ (growth rate proportional to $u$, braked as $u$ approaches its ceiling). Solve it by separation of variables and partial fractions #primer("ODE.4", "logistic").
     #deliverable[the ODE $dot(u) = F(u)$; $u(t)$ in closed form given $u_0$ and $s$.]
     #hints([Show $q = a - b$ obeys a linear ODE $dot(q) = k(t) q$, so $q_0 = 0$ forces $q equiv 0$ by uniqueness (primer §ODE.1); then differentiate $u = a b$ by the product rule.], [Separate variables and use partial fractions (primer §ODE.4, §ALG.3).], [First line of the solution: "For $L = 1/2 (s - a b)^2$: $dot(a) = b(s - a b)$, $dot(b) = a(s - a b)$, so $d\/d t (a - b) = -(a - b)(s - a b)$."])],
    [#tag("derive") From (a), compute $t_f (u_0, s)$, then its leading behaviour for $u_0 << s$ at $f = 1/2$ #primer("ODE.4", "time between two levels"), #primer("TAY.4", "leading order"). Describe the shape of $u(t)$ for $u_0 = 10^(-6)$: how long is the _plateau_ (the stretch where $u$ and the loss barely move) compared with the _drop_ (the time from $u = 0.1 s$ to $u = 0.9 s$)?
     #deliverable[$t_f$ exactly; $t_(1\/2)$ to leading order; how plateau length and drop length each scale with $u_0$ and $s$.]
     #hints([Partial fractions (primer §ODE.4): $1/(u(s - u)) = 1/s (1/u + 1/(s - u))$.], [Write the time from $u_1$ to $u_2$ as $integral_(u_1)^(u_2) d u \/ F(u)$.], [For the drop, both endpoints are fixed fractions of $s$, so $u_0$ drops out.])],
    [#tag("implement") In `symmetries/logistic.py`, broadcasting over numpy arrays:
     #defline[`def balanced_product(u0, s, t)`  ·  `def time_to_learn(u0, s, frac=0.5)`]
     #arglist(
       ("balanced_product", [$u(t)$ from (a); finite for any $t >= 0$ (rewrite it if $e^(2 s t)$ would overflow).]),
       ("time_to_learn", [$t_f (u_0, s)$ with $f = $ `frac`, from (b).]),
     )
     #filetest("test_logistic", ("balanced_product", "time_to_learn"))],
    [#tag("check") Run `scripts/check_logistic.py`. It integrates the full 2-D flow for $s in {0.5, 1, 2, 4}$ and 15 values of $u_0$ from $10^(-8)$ to $10^(-1)$ and finds when $a b$ crosses $s\/2$. Expected: your $t_(1\/2)$ within $10^(-8)$ relative, and the slope of $t_(1\/2)$ against $ln(1\/u_0)$ equal to your prediction from (b) to $1%$ (`logistic.png`, left).],
    [#tag("predict") Before looking at the right panel, write in `notes.md`: at fixed $a_0 b_0 = 10^(-4)$ and $s = 1$, does an _unbalanced_ start, $c = a_0^2 - b_0^2 > 0$, learn faster or slower than the balanced one, and roughly from which $c$ on does it matter? Then derive $dot(p)$ for general $c$ (the post's §2.1 has it) and explain the panel.
     #deliverable[the prediction; $dot(p)$ as a function of $p$, $c$, $s$; two sentences.]],
  )
]

= Two-layer linear networks

Replace the scalars by matrices: $hat(y) = W_2 W_1 x$ with $W_1 in RR^(m times d_"in")$, $W_2 in RR^(d_"out" times m)$, hidden width $m$, and with $Sigma_(x x) = EE[x x^top]$, $Sigma_(y x) = EE[y x^top]$ as in A0,
$ L(W_1, W_2) = 1/2 EE norm(W_2 W_1 x - y)^2 = 1/2 "tr"(W Sigma_(x x) W^top) - "tr"(W Sigma_(y x)^top) + "const", quad W = W_2 W_1. $ <eq-lin>
$L$ depends on $(W_1, W_2)$ only through $W$, so $(W_1, W_2) -> (A W_1, W_2 A^(-1))$ is a symmetry for every invertible $m times m$ matrix $A$ ($m^2$ parameters instead of one), and the conserved quantity is a matrix,
$ D = W_1 W_1^top - W_2^top W_2 in RR^(m times m). $ <eq-dmat>
Du, Hu and Lee prove that gradient flow keeps $D$ fixed for any two consecutive linear layers inside any network @du2018balanced[§2.1, Theorem 2.2]; Simon et al. list it among the "simple empirical laws" of deep learning @simon2026theory[§2.3, p. 13].

#problem("matrix-law", [The matrix conservation law], kind: ("derive", "implement", "check"), time: "45 min", owner: "you")[
  #parts(
    [#tag("derive") With $G = nabla_W L$ the gradient of @eq-lin with respect to the product, find $nabla_(W_1) L$ and $nabla_(W_2) L$ by the chain rule (perturb $W_1 -> W_1 + E$ and keep the first-order term of $L$) #primer("LA.5", "matrix calculus").
     #deliverable[$G$, $nabla_(W_1) L$, $nabla_(W_2) L$ as matrix products.]
     #hints([Matrix gradients by perturbation: write the first-order change of $L$ as $⟨dot, E⟩$ using the cyclic trace (primer §LA.5).], [Perturbing $W_1$ by $E$ perturbs $W = W_2 W_1$ by $W_2 E$, and the change of $L$ is $⟨G, W_2 E⟩$.], [First line of the solution: "$L(W) = 1/2 "tr"(W Sigma_(x x) W^top) - "tr"(W Sigma_(y x)^top)$ has $G = nabla_W L = W Sigma_(x x) - Sigma_(y x)$."])],
    [#tag("derive") Show $dot(D) = 0$ under gradient flow, using only that $L$ depends on $W_2 W_1$ through some $G$. Then derive it a second way, Noether-style: take $A = I + epsilon M$ for an arbitrary $M$ and differentiate at $epsilon = 0$; which matrix does "for all $M$" force to vanish? For $m = d_"in" = d_"out" = 1$, recover Problem (#raw("noether")) #primer("LA.5", "products and \"for all M\""), #primer("LA.6", "symmetries").
     #deliverable[the two-line proof that $dot(D) = 0$; the constraint as an $m times m$ matrix identity $K = 0$, with $K$ written in $W_1$, $W_2$, $nabla_(W_1) L$, $nabla_(W_2) L$.]
     #hints([Product rule: $dot(D) = dot(W)_1 W_1^top + W_1 dot(W)_1^top - dot(W)_2^top W_2 - W_2^top dot(W)_2$; substitute the gradients from (a) (primer §LA.5).], [For the Noether route, $(I + epsilon M)^(-1) = I - epsilon M + O(epsilon^2)$; the derivative at $epsilon = 0$ has the form $"tr"(M K)$, and $"tr"(M K) = 0$ for all $M$ forces $K = 0$ (primer §LA.5, §LA.6).], [First line of the solution: "Under gradient flow $dot(W_1) = -W_2^top G$, $dot(W_2) = -G W_1^top$."])],
    [#tag("derive") Take one GD step, $W_i' = W_i - eta nabla_(W_i) L$. Compute $D' - D$ _exactly_ (no expansion in $eta$) #primer("TAY.3", "a conserved quantity under one step"): the first-order terms cancel; write what remains in terms of $nabla_(W_1) L$ and $nabla_(W_2) L$.
     #deliverable[$D' - D$ in terms of $eta$, $nabla_(W_1) L$, $nabla_(W_2) L$; its scalar case as a function of $c$, $r$, $eta$.]
     #hints([Expand the products exactly and collect powers of $eta$, as in the scalar worked example of primer §TAY.3.], [The $eta^1$ terms of $W_1' W_1'^top$ and of $W_2'^top W_2'$ are equal; check it with your (a).], [First line of the solution: "With $g_i = nabla_(W_i) L$: $W_1' W_1'^top = W_1 W_1^top - eta (g_1 W_1^top + W_1 g_1^top) + eta^2 g_1 g_1^top$."])],
    [#tag("implement") In `symmetries/two_layer.py`:
     #defline[`def two_layer_grads(W1, W2, Sxx, Syx)`  ·  `def imbalance(W1, W2)`  ·  `def gd_imbalance_change(W1, W2, Sxx, Syx, eta)`]
     #arglist(
       ("two_layer_grads", [$(nabla_(W_1) L, nabla_(W_2) L)$ from (a), matrix products only.]),
       ("imbalance", [$D$ by @eq-dmat.]),
       ("gd_imbalance_change", [your exact $D' - D$ from (c), without taking the step.]),
     )
     #filetest("test_two_layer", ("two_layer_grads", "imbalance", "gd_imbalance_change"))],
    [#tag("check") Run `scripts/check_matrix_law.py` ($d_"in" = 4$, $m = 3$, $d_"out" = 2$, correlated inputs). Expected: $norm(D(t) - D(0))_F$ below $10^(-11)$ along RK4 gradient flow; under GD every step's change equal to your formula to round-off; the accumulated drift at $t = 10$ growing like $eta^p$ (`matrix_law.png`). Predict $p$ from (c) before you look: how many steps does it take to reach $t = 10$?],
  )
]

== Saxe's mode decomposition

Now let the inputs be _whitened_, $Sigma_(x x) = I$, and take the _singular value decomposition_ (SVD) $Sigma_(y x) = sum_alpha s_alpha bold(u)_alpha bold(v)_alpha^top = U S V^top$ with singular values $s_1 >= s_2 >= dots > 0$ and orthonormal singular vectors $bold(u)_alpha$ (outputs), $bold(v)_alpha$ (inputs), written in bold to keep them apart from the scalar $u$ of Problem (#raw("logistic")). A0 Problem (#raw("modes")) showed that one layer learns all modes at the same rate. Saxe et al. show that two layers learn them _one at a time_ @saxe2014exact[§1.1–1.3]. The key object is the _aligned, balanced_ initialization: pick any $m times r$ matrix $R$ with orthonormal columns ($r$ the number of modes) and set $W_1 = R thin "diag"(sqrt(u_(0 alpha))) thin V^top$, $W_2 = U "diag"(sqrt(u_(0 alpha))) R^top$. Then (part (a) below) the product stays of the form
$ W(t) = W_2(t) W_1(t) = sum_alpha u_alpha (t) thin bold(u)_alpha bold(v)_alpha^top, $ <eq-saxe>
and each _mode strength_ $u_alpha (t) = bold(u)_alpha^top W(t) bold(v)_alpha$ evolves on its own.

#example("svd-by-hand", [Modes of a $2 times 2$ correlation])[
  $Sigma_(y x) = mat(2, 1; 1, 2)$ is symmetric with eigenvectors $(1, 1)\/sqrt(2)$ and $(1, -1)\/sqrt(2)$, eigenvalues $3$ and $1$. Both are positive, so its SVD has $bold(u)_alpha = bold(v)_alpha$: $s_1 = 3$ along $(1, 1)\/sqrt(2)$, $s_2 = 1$ along $(1, -1)\/sqrt(2)$. A two-layer net learns the map "add the two inputs" ($s_1$) before "subtract them" ($s_2$); part (b) below says how much later.
]

#problem("saxe", [Deep linear networks learn mode by mode], kind: ("derive", "implement", "check"), time: "60 min", owner: "you")[
  #parts(
    [#tag("derive") Change variables to $overline(W)_1 = W_1 V$ and $overline(W)_2 = U^top W_2$ (extend $U, V$ to square orthogonal matrices). From the aligned balanced initialization, show that $overline(W)_1$ and $overline(W)_2$ keep their initial structure, that @eq-saxe holds, and that each pair $(a_alpha, b_alpha)$, defined by column $alpha$ of $overline(W)_1$ being $a_alpha r_alpha$ and row $alpha$ of $overline(W)_2$ being $b_alpha r_alpha^top$ ($r_alpha$ the columns of $R$), follows the anchor with target $s_alpha$.
     #deliverable[the ODEs for $(a_alpha, b_alpha)$; one sentence naming the term that would couple two modes $alpha != beta$ and why it vanishes.]
     #hints([Rotate both sides by the singular vectors so that $Sigma_(y x)$ becomes the diagonal $S$ (primer §LA.4); the ansatz argument is primer §ODE.1.], [Use your gradients from Problem (#raw("matrix-law")) with $Sigma_(x x) = I$; verify that the ansatz is consistent (its velocity has the same structure) and invoke uniqueness of ODE solutions.], [First line of the solution: "With $Sigma_(x x) = I$ the flow is $dot(W_1) = W_2^top (Sigma_(y x) - W_2 W_1)$, $dot(W_2) = (Sigma_(y x) - W_2 W_1) W_1^top$."])],
    [#tag("derive") With Problem (#raw("logistic")), give $u_alpha (t)$, and the ratio $t_(1\/2, alpha) \/ t_(1\/2, beta)$ of half-times for two modes from the same small $u_0$, to leading order #primer("ALG.5", "time scales").
     #deliverable[$u_alpha (t)$; the ratio, exactly and to leading order, as a formula in $s_alpha$, $s_beta$ (and $u_0$).]
     #hints([Each mode is a balanced anchor with target $s_alpha$: reuse Problem (#raw("logistic")) (primer §ODE.4).], [For small $u_0$, $t_(1\/2) approx 1/(2 s) ln(s\/u_0)$; to leading order keep the $1\/s$ outside the logarithm (primer §TAY.4, §ALG.5).], [First line of the solution: "The init is balanced ($a_alpha = b_alpha = sqrt(u_(0 alpha))$), so each mode is Problem (#raw("logistic"))."])],
    [#tag("implement") In `symmetries/saxe.py` ($Sigma_(x x) = I$; modes are singular triples with $s_alpha > 10^(-10) s_1$, in `np.linalg.svd` order; `u0` a scalar or one value per mode):
     #defline[`def aligned_init(Syx, u0, hidden)`  ·  `def saxe_prediction(Syx, u0, t)`  ·  `def effective_u0(W1, W2, Syx)`]
     #arglist(
       ("aligned_init", [$(W_1, W_2)$ as above, with any valid $R$.]),
       ("saxe_prediction", [$W(t)$ by @eq-saxe and (b).]),
       ("effective_u0", [the per-mode $u_0^"eff"$ of (d).]),
     )
     #filetest("test_saxe", ("aligned_init", "saxe_prediction", "effective_u0"))],
    [#tag("derive") Real networks start from small _random_ weights, not aligned ones. For mode $alpha$ let $x = W_1 bold(v)_alpha in RR^m$ and $y = W_2^top bold(u)_alpha in RR^m$, so $u_alpha = x^top y$. While all weights are small, drop the terms cubic in the weights from the flow and solve for $x(t)$, $y(t)$ #primer("ODE.6", "linear systems"). Match the result with the early phase of (b), $u(t) approx u_0 e^(2 s t)$.
     #deliverable[$u_0^"eff"$ for mode $alpha$ as a function of $W_1(0)$, $W_2(0)$, $bold(u)_alpha$, $bold(v)_alpha$.]
     #hints([Dropping the cubic terms leaves a linear system; solve it through its eigenvectors (primer §ODE.6).], [The linearized system is $dot(x) = s_alpha y$, $dot(y) = s_alpha x$; look at $x + y$ and $x - y$, and use $x^top y = 1/4 (norm(x + y)^2 - norm(x - y)^2)$.], [First line of the solution: "Linearize: $dot(x) = dot(W_1) bold(v)_alpha = W_2^top (Sigma_(y x) - W) bold(v)_alpha approx W_2^top Sigma_(y x) bold(v)_alpha = s_alpha W_2^top bold(u)_alpha = s_alpha y$."])],
    [#tag("check") Run `scripts/check_saxe.py`: $d_"in" = 8$, $d_"out" = 6$, $m = 6$, $s = (4, 2, 1, 0.5)$. Expected: from the aligned init your $W(t)$ within $10^(-7)$ of the integrated flow; from a random init of scale $10^(-3)$, half-times within $5%$ of your prediction with $u_0^"eff"$, a small delay growing for weaker modes (competition between modes, Saxe et al. Fig. 3 right), and a loss made of plateaus and drops (`saxe.png`).],
  )
]

= How the law breaks

Gradient flow is a model. Real training takes finite steps $eta$, often with heavy-ball _momentum_ $theta_(k+1) = theta_k - eta nabla L(theta_k) + beta (theta_k - theta_(k-1))$, on minibatches, with weight decay. In GD with weight decay, one step on the anchor reads
$ a_(k+1) = a_k - eta (b_k r_k + 2 gamma a_k), quad b_(k+1) = b_k - eta (a_k r_k + 2 gamma b_k). $ <eq-gdwd>
To track a slowly drifting quantity, find an ODE that the iterates follow more closely than gradient flow. _Backward error analysis_ (A3 derives it) gives, for GD, the _modified gradient flow_
$ dot(theta) = -nabla L(theta) - eta/2 nabla^2 L(theta) nabla L(theta), $ <eq-mflow>
which matches GD to one order higher in $eta$; time is $t = k eta$. For heavy-ball momentum started at rest, Kunin et al. use the second-order _modified momentum flow_ @kunin2021mechanics[§5 and App. G, with no damping]
$ eta (1 + beta)/2 dot.double(theta) + (1 - beta) dot(theta) = -nabla L(theta), quad dot(theta)(0) = 0, $ <eq-mmom>
again with $t = k eta$. To leading order it is gradient flow slowed by $1 - beta$, i.e. with effective step $eta\/(1 - beta)$. Kunin et al. combine these with the symmetry constraints and check the result on VGG-16 @kunin2021mechanics[§6, Fig. 5]; here the anchor gives closed forms.

#problem("broken", [Finite steps, momentum, minibatches and weight decay], kind: ("derive", "implement", "check"), time: "75 min", owner: "you")[
  #parts(
    [#tag("derive") From @eq-gdwd, compute $c_(k+1)$ exactly in terms of $c_k$, $r_k$, $eta$, $gamma$ #primer("TAY.3", "one GD step"). Separately, for gradient flow on $L_gamma$ find $dot(c)$ and solve it #primer("ODE.3", "linear decay"). (Ang's post states the $gamma = 0$ step @ang2026post[§3.1]; the brief writes weight decay as $-gamma a$, a factor $2$ apart from this course's convention.)
     #deliverable[$c_(k+1)$; $c(t)$ under the flow with weight decay (type it into `wd_imbalance` in `scripts/check_drift.py`); for $gamma = 0$, the condition on $eta r_k$, as an inequality, under which $|c|$ cannot grow.]
     #hints([Expand $a_(k+1)^2 - b_(k+1)^2$ exactly, as in primer §TAY.3; the flow part is a linear ODE (primer §ODE.3).], [Write the step as $a' = (1 - 2 eta gamma) a - eta b r$; the cross terms cancel because $a b - b a = 0$.], [First line of the solution: "Write @eq-gdwd as $a' = (1 - 2 eta gamma) a - eta b r$, $b' = (1 - 2 eta gamma) b - eta a r$."])],
    [#tag("derive") Along @eq-mflow ($gamma = 0$), compute $dot(c)$ #primer("LA.6", "the generator"). Use the constraint from Problem (#raw("noether")) to kill the $nabla L$ term; for the correction term you need $nabla^2 L thin nabla L$ explicitly.
     #deliverable[$dot(c)$ as a function of $eta$, $r$, $c$.]
     #hints([$dot(c) = ⟨nabla c, dot(theta)⟩ = 2 ⟨X, dot(theta)⟩$ with $X = (a, -b)$ (primer §LA.6).], [$⟨X, nabla^2 L nabla L⟩ = ⟨nabla^2 L thin X, nabla L⟩$ since $nabla^2 L$ is symmetric; use your identity from Problem (#raw("noether")) (b).], [First line of the solution: "Along @eq-mflow, $dot(c) = -2⟨X, nabla L⟩ - eta ⟨X, nabla^2 L nabla L⟩$ with $X = (a, -b)$."])],
    [#tag("derive") Integrate (b) to first order in $eta$: replace $r(t)$ by its gradient-flow value. Show that $ln(c_infinity \/ c_0) approx -eta thin I(p_0, c_0)$ with $I = integral_0^infinity r(t)^2 thin d t$, change variables from $t$ to $p$ using $dot(p)$ from Problem (#raw("logistic")) (e), and evaluate the integral in closed form #primer("ODE.3", "time-varying rate"), #primer("ALG.3", "integrating along a trajectory"). How does $|ln(c_1\/c_0)|$ after a single step scale with $eta$, and how does the total $|ln(c_infinity \/ c_0)|$ scale?
     #deliverable[$I(p_0, c)$ in closed form ($c != 0$); the two log-log slopes against $eta$.]
     #hints([$d t = d p \/ dot(p)$ (primer §ALG.3); $p$ moves monotonically from $p_0$ to $1$.], [$integral d p \/ sqrt(c^2 + 4 p^2) = 1/2 "arcsinh"(2 p\/|c|)$.], [$integral p thin d p \/ sqrt(c^2 + 4 p^2) = 1/4 sqrt(c^2 + 4 p^2)$.])],
    [#tag("derive") Repeat (b)–(c) for momentum, starting from @eq-mmom. Take the inner product with $(a, -b)$ to get an equation for $c$, keep terms to first order in $eta$ #primer("TAY.2", "first order"), and integrate over the whole run.
     #deliverable[the factor by which momentum $beta$ multiplies the first-order drift of $ln c$ at a fixed $eta$.]
     #hints([Differentiate $c$ twice (primer §TAY.1, chain rule): $⟨(a, -b), dot.double(theta)⟩ = 1/2 dot.double(c) - (dot(a)^2 - dot(b)^2)$.], [To leading order $dot(theta) = -nabla L \/ (1 - beta)$, and the run takes $1\/(1-beta)$ times longer in $t$ than gradient flow takes.], [The $dot.double(c)$ term integrates to $dot(c)(infinity) - dot(c)(0)$; both vanish.])],
    [#tag("derive") _Minibatches._ Take $n$ examples with $1/n sum_i x_i^2 = 1$ and $y_i = x_i$, per-example loss $1/2 (a b x_i - y_i)^2$, and SGD on batches of $B$ indices drawn with replacement. Show that the batch gradient is $m_B r (b, a)$ with $m_B$ the batch mean of $x_i^2$, so it is still orthogonal to $X$, and find $c_(k+1)$. To first order in $eta$, by what factor does SGD's drift exceed GD's? #primer("PR.2", "sample means")
     #deliverable[$c_(k+1)$ for SGD; the factor in terms of $B$ and the variance of $x_i^2$ over the dataset (type it into `sgd_drift_factor`).]
     #hints([The batch mean of $x_i^2$ multiplies the GD step; the drift needs the mean of its square (primer §PR.2).], [$EE[m_B^2] = (EE m_B)^2 + "Var"(m_B)$, and $"Var"(m_B) = "Var"(x_i^2)\/B$ for draws with replacement.], [First line of the solution: "Per-example gradient: $partial_a 1/2 (a b x_i - x_i)^2 = b x_i^2 r$, so the batch gradient is $m_B r (b, a)$ with $m_B = 1/B sum_(i in B) x_i^2$."])],
    [#tag("implement") In `symmetries/drift.py`:
     #defline[`def imbalance_after_step(a, b, eta, gamma=0.0)`  ·  `def drift_integral(p0, c)`  ·  `def predicted_log_drift(a0, b0, eta, beta=0.0)`]
     #arglist(
       ("imbalance_after_step", [$c_(k+1)$ by your (a), as a formula; do not take the step.]),
       ("drift_integral", [your closed form $I(p_0, c)$ from (c).]),
       ("predicted_log_drift", [the first-order prediction of $ln(c_infinity \/ c_0)$, from (c) and (d).]),
     )
     #filetest("test_drift", ("imbalance_after_step", "drift_integral", "predicted_log_drift"))],
    [#tag("check") Run `scripts/check_drift.py` from $(a_0, b_0) = (0.3, 0.1)$ with $eta$ from $0.002$ to $0.1$. Expected: the one-step formula exact to round-off; $c(t)$ under weight decay on your formula; both log-log slopes within $0.05$ of your (c); measured over predicted drift tending to $1$ as $eta -> 0$ (within $1%$ at $eta = 0.002$) for GD and $beta = 0.3, 0.5$; SGD's ratio on your factor within two standard errors for $B = 1, 4, 16$ (`drift.png`). A ratio tending to a constant other than $1$ means a wrong factor in (c) or (d).],
  )
  #note[Every minibatch gradient obeys the same constraint, so gradient noise is orthogonal to the generator @kunin2021mechanics[§5]: it is the _finite step_ that breaks the law. Which way the drift pushes $|c|$ is a first glimpse of the implicit regularization of A3 and A4.]
]

= Framing

#problem("framing", [Conservation laws in the larger map], kind: ("read",), time: "20 min", owner: "you")[
  Read Simon et al. @simon2026theory §2.1 "Linearization in the data" with Figure 1 (pp. 6–7) and the paragraph "Gradient flow conservation laws" in §2.3 (p. 13). Answer briefly in `notes.md`:
  #parts(
    [Figure 1(a) assumes "task-aligned initialization" and whitened inputs. Which did `check_saxe.py` drop in its second run, and what did it cost, in numbers?],
    [Conservation laws are "weakly broken by SGD in predictable ways". Which breakings did you predict, to which order in $eta$, and what in real training would your predictions miss?],
    [Attention scores depend on $W_Q W_K^top$. Write the symmetry this gives and, with Problem (#raw("matrix-law")) (b), the matrix conserved by gradient flow.],
  )
]

= What to write down

In `notes.md`, three to five sentences: what a symmetry gives you under gradient flow and under GD; why small init gives balanced layers and a long plateau; what "a deep linear network is a bundle of anchors" means and when it fails; one question for a tutorial.

#bibliography("/template/refs.bib", style: "/template/handout.csl", title: "References")
