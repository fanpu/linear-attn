#import "/template/template.typ": *

#show: doc.with(
  assignment: 2,
  title: "Weight decay and phase transitions",
  kind: "Worked solutions",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "reference numbers: solutions/results.json · figures: solutions/figures/ · equation numbers (n) refer to the handout",
)

Every derivation here is complete; read only as far as you need. "Measured" numbers come from running each check script with the reference implementations (`--solutions`) on CPU; your numbers should agree within the tolerances stated in each problem. Diff yours with `uv run python -m tdyn.results a2-weight-decay/out/results.json a2-weight-decay/solutions/results.json`. Equation numbers such as (1) refer to the handout; this document numbers its own equations (S1), (S2), ... Every closed form below was checked symbolically (sympy) and numerically before it was written down.

#set math.equation(numbering: n => "(S" + str(n) + ")")

= Problem (#raw("balance")): Weight decay balances the layers

#runin[(a)] With $nabla ell(a b) = ell'(a b)(b, a)$, the loss part cancels as in A1 and only weight decay is left:
$ a thin partial_a L_gamma - b thin partial_b L_gamma = a (ell' b + 2 gamma a) - b (ell' a + 2 gamma b) = 2 gamma (a^2 - b^2) = 2 gamma c. $
At a stationary point the left side is zero, so for $gamma > 0$ every stationary point has $a^2 = b^2$, whatever $ell$ and the data. Along the flow $dot(c) = 2 a dot(a) - 2 b dot(b) = -2(a partial_a L_gamma - b partial_b L_gamma) = -4 gamma c$, so $c(t) = c_0 e^(-4 gamma t)$ (A1 Problem (#raw("broken")) (a)). The conserved quantity of A1 becomes a decaying one, and its only fixed value is $0$.

#runin[(b)] With $G = nabla_W L$ at $W = W_2 W_1$, the gradients are $nabla_(W_1) = W_2^top G + 2 gamma W_1$ and $nabla_(W_2) = G W_1^top + 2 gamma W_2$. The $G$ terms cancel in $dot(D)$ exactly as in A1 Problem (#raw("matrix-law")) (b); the weight-decay terms give $dot(W_1) W_1^top + W_1 dot(W_1)^top supset -4 gamma W_1 W_1^top$ and $-(dot(W_2)^top W_2 + W_2^top dot(W_2)) supset +4 gamma W_2^top W_2$. So
$ dot(D) = -4 gamma D, quad D(t) = D(0) e^(-4 gamma t). $
At any stationary point $D = 0$: $W_1 W_1^top = W_2^top W_2$. The $(j, j)$ entry reads $norm(W_1[j, :])^2 = norm(W_2[:, j])^2$: each hidden neuron has the same norm coming in as going out. This is Ziyin et al.’s Lemma 1 with $gamma_u = gamma_w$ (with unequal decay strengths the flow gives $dot(D) = -4(gamma_w W_1 W_1^top - gamma_u W_2^top W_2)$ and the stationary relation becomes their $gamma_u U_j^2 = gamma_w norm(W_(j :))^2$). Here it holds at _every_ stationary point, not only the global minimum. The off-diagonal entries say more: the hidden representations of the two layers have equal Gram matrices.

#runin[(c)] For $f = a b c x$, $partial_a L_gamma = b c thin ell' + 2 gamma a$ and so on, and $a partial_a L_gamma - b partial_b L_gamma = 2 gamma (a^2 - b^2)$. Hence $a^2 - b^2$, $b^2 - c^2$ (and $a^2 - c^2$) each decay as $e^(-4 gamma t)$, and every stationary point has $a^2 = b^2 = c^2$.

#runin[(d) Expected.] Maximum relative error of $D(t)$ on $[0, 20]$: $2.2 times 10^(-11)$ ($gamma = 0.05$) and $1.9 times 10^(-10)$ ($gamma = 0.2$); depth-3 differences $1.1 times 10^(-10)$. $max ||a_infinity| - |b_infinity||$ over 200 starts: $1.5$ at $gamma = 0$ (endpoints spread along $a b = 1$ between $(0.5, 2)$ and $(2, 0.5)$) and $7.8 times 10^(-16)$ at $gamma = 0.1$ (all at $(plus.minus 0.894, plus.minus 0.894)$). Matrix network at $t = 200$: $norm(D)_F = 3.2 times 10^(-16)$, per-neuron gap $1.1 times 10^(-16)$.

#figure(image("figures/balance.png", width: 88%), caption: [`balance.png`. Left: imbalances decaying at rate $4 gamma$ (dashed: the formula). Right: where 200 flows end without (blue) and with (orange) weight decay.])

= Problem (#raw("stationary")): Stationary points and the bifurcation at one half

#runin[(a)] By (balance) (a), a stationary point has $b = a$ or $b = -a$.
- $b = a$: $partial_a L_gamma = a r + 2 gamma a = a (a^2 - 1 + 2 gamma) = 0$, so $a = 0$ or $a^2 = 1 - 2 gamma$, possible only for $gamma < 1/2$.
- $b = -a$: $r = -a^2 - 1$ and $partial_a L_gamma = b r + 2 gamma a = a (a^2 + 1 + 2 gamma)$, zero only at $a = 0$.
So the stationary points are the origin for every $gamma$, and $plus.minus (u, u)$ with $u = sqrt(1 - 2 gamma)$ for $gamma < 1/2$. From any start off the line $a = -b$ the flow ends at one of $plus.minus (u, u)$ for $gamma < 1/2$ (the origin is a saddle, below) and at the origin for $gamma > 1/2$:
$ p_infinity (gamma) = cases(1 - 2 gamma & gamma < 1/2, 0 & gamma >= 1/2.) $ <s-pinf>
The Hessian of (1) is
$ H(a, b) = mat(b^2 + 2 gamma, 2 a b - 1; 2 a b - 1, a^2 + 2 gamma). $
At the origin $H = mat(2 gamma, -1; -1, 2 gamma)$: eigenvalue $2 gamma - 1$ on $(1, 1)$ and $2 gamma + 1$ on $(1, -1)$. At $plus.minus (u, u)$, $a^2 = b^2 = a b = 1 - 2 gamma$, so $H = mat(1, 1 - 4 gamma; 1 - 4 gamma, 1)$: eigenvalue $2 - 4 gamma$ on $(1, 1)$ and $4 gamma$ on $(1, -1)$. Classification:
- $gamma < 1/2$: origin a saddle ($2 gamma - 1 < 0 < 2 gamma + 1$); $plus.minus (u, u)$ minima ($4 gamma > 0$, $2 - 4 gamma > 0$).
- $gamma = 1/2$: only the origin, degenerate (eigenvalues $0$ and $2$). It is still the global minimum: on $a = b$, $L_(1\/2) = 1/2 (u^2 - 1)^2 + u^2 = 1/2 + u^4\/2$, flat to fourth order.
- $gamma > 1/2$: the origin is the only stationary point and a minimum.
As $gamma arrow.t 1/2$ the two minima slide along $a = b$ into the origin, and their eigenvalue $2 - 4 gamma$ along the line goes to zero together with the origin's $2 gamma - 1$. This is a _supercritical pitchfork bifurcation_: one fixed point (the origin) changes stability while two symmetric ones are born from it. The order parameter $p_infinity = 1 - 2 gamma = 2(1/2 - gamma)$ vanishes linearly; in terms of the weights, $u = sqrt(2(1/2 - gamma))$ vanishes like a square root, the mean-field exponent of a magnet.

#runin[(b)] Reference: `solutions/stationary_solution.py`.

#runin[(c) Expected.] Maximum $|a b - p_infinity|$ over 79 values of $gamma$: $9.8 times 10^(-9)$ (the run at $gamma = 0.499$ is the slowest). Autograd eigenvalues equal to the formulas to $2.2 times 10^(-16)$; e.g. at $gamma = 0.45$: origin $(-0.1, 1.9)$, minima $(0.2, 1.8)$. Log-log slope of $p_infinity$ against $1/2 - gamma$: $1.0000$. Against the ridge curve $1\/(1 + 2 gamma)$ of the example, depth 2 shrinks faster and hits zero.

#figure(image("figures/bifurcation.png", width: 100%), caption: [`bifurcation.png`. Left: learned $a b$ against $gamma$. Middle: Hessian eigenvalues at the origin and the minima. Right: the order parameter near $gamma = 1/2$.])

= Problem (#raw("slowing")): Critical slowing down

#runin[(a)] On $a = b = u$, (2) gives $dot(u) = -u (u^2 - 1 + 2 gamma)$, and $dot(p) = 2 u dot(u) = 2 p (s_"eff" - p)$ with
$ s_"eff" = 1 - 2 gamma. $
Weight decay lowers the effective target; above $gamma = 1/2$ the target is negative and the logistic equation drives $p$ to $0$.

#runin[(b)] The flow converges to the minimum for $gamma < 1/2$ and to the origin for $gamma > 1/2$, so with (stationary) (a):
$ tau(gamma) = cases(1 \/ (4 gamma) & "if" 0 < gamma <= 1/4,
  1 \/ (2 - 4 gamma) = 1 \/ (4 (1/2 - gamma)) quad & "if" 1/4 <= gamma < 1/2,
  1 \/ (2 gamma - 1) = 1 \/ (2 (gamma - 1/2)) & "if" gamma > 1/2.) $ <s-tau>
The slow direction is $(1, -1)$ on the first piece and $(1, 1)$ on the other two.
So $C_minus = 1/4$, $C_plus = 1/2$, and $C_plus \/ C_minus = 2$: at equal distance from the critical point, relaxation is twice as slow on the zero side. (The same factor $2$ appears in the Landau theory of a magnet, for the same reason: below the transition the restoring curvature comes from both the quadratic and the quartic term.) The other divergence is $gamma -> 0$, along $(1, -1)$: that is the imbalance $c$, which decays only at rate $4 gamma$ (balance (a)); at $gamma = 0$ it is A1’s conserved quantity and the minima form a flat valley.

#runin[(c)] At $gamma = 1/2$, $dot(p) = -2 p^2$, so $1\/p = 1\/p_0 + 2 t$ and
$ p(t) = p_0 / (1 + 2 p_0 t), quad norm(theta(t)) = sqrt(2 p) approx t^(-1\/2). $
Convergence is algebraic, not exponential: the linear term vanished and only the cubic force $-u^3$ is left.

#runin[(d)] The distance on the line is $sqrt(2) |u - u^*|$ with $u^* = sqrt(s_"eff")$ (or $0$ for $gamma >= 1/2$). The flow reaches distance $epsilon$ at $u_epsilon = u^* + "sign"(u_0 - u^*) thin epsilon \/ sqrt(2)$, i.e. $p_epsilon = u_epsilon^2$. Inverting A1’s logistic solution with target $s = s_"eff"$,
$ t_epsilon = 1/(2 s) ln (p_epsilon (s - p_0)) / (p_0 (s - p_epsilon)) quad (s != 0), quad t_epsilon = 1/2 (1/p_epsilon - 1/p_0) quad (s = 0), $ <s-teps>
and $t_epsilon = 0$ if $sqrt(2) |u_0 - u^*| <= epsilon$. For $s < 0$ both $s - p_0$ and $s - p_epsilon$ are negative, so the logarithm is real. The slope is not exactly $-1$: $t_epsilon approx tau ln(K\/epsilon)$ and $K$ depends on $gamma$. Below, $s - p_epsilon approx -sqrt(2 s) thin epsilon$ gives $K prop sqrt(s)$; above, $p_epsilon = epsilon^2\/2$ gives $t_epsilon approx 1/(2|s|) ln(2 |s| p_0 \/ ((p_0 + |s|) epsilon^2))$. In both, $d ln t_epsilon \/ d ln |gamma - 1/2| = -1 + O(1 \/ ln(K\/epsilon))$, slightly shallower than $-1$. The late-time ratio used in the check removes $K$.

#runin[(e)] Reference: `solutions/slowing_solution.py`.

#runin[(f) Expected.] Measured over predicted $tau$: within $2.6 times 10^(-4)$ below and $1.4 times 10^(-6)$ above; slopes $-0.9999$ and $-1.0000$; $tau |gamma - 1/2| = 0.2500$ and $0.5000$, ratio $2.0002$. $t_epsilon$ ($epsilon = 10^(-4)$, $u_0 = 0.5$) matches @s-teps to $4 times 10^(-9)$ relative; its slopes are $-0.962$ (below) and $-0.940$ (above). At $gamma = 1/2$: $p(t)$ matches to $1.5 times 10^(-10)$, late-time exponent $-0.4998$.

#figure(image("figures/slowing.png", width: 100%), caption: [`slowing.png`. Left: relaxation time on both sides (the upper line is the zero side). Middle: time to within $10^(-4)$. Right: algebraic decay at $gamma = 1/2$ against exponential decay just above it.])

= Problem (#raw("depth3")): The jump at depth 3

#runin[(a)] $1/2 (a b c - s)^2 = 1/2 s^2 - s thin a b c + 1/2 a^2 b^2 c^2$. Apart from the constant, every term has degree $3$ or $6$, so all second derivatives of the data term vanish at the origin, and
$ nabla^2 L_gamma (0, 0, 0) = 2 gamma I. $
The origin is a strict local minimum for every $gamma > 0$. At depth 2 the term $-s thin a b$ is quadratic and gives the off-diagonal $-s$ that, for $2 gamma < s$, makes the origin a saddle. At depth 3 the signal enters only at cubic order, so near zero weight decay always wins (Ziyin et al., Proposition 2).

#runin[(b)] Multiplying the components of $nabla L_gamma = 0$ by $a$, $b$, $c$: $p r + 2 gamma a^2 = p r + 2 gamma b^2 = p r + 2 gamma c^2 = 0$, so $a^2 = b^2 = c^2 = u^2$ and $p r = -2 gamma u^2 < 0$ at a nonzero point. With $r = p - s$ that forces $0 < p < s$, hence $p = a b c = u^3 > 0$ (an even number of negative weights). If one weight is zero, the others' equations reduce to $2 gamma b = 0$ and so on, which gives the origin. Then $p r = -2 gamma u^2$ reads $u^3 (u^3 - s) = -2 gamma u^2$, i.e.
$ u^4 - s u + 2 gamma = 0 quad <=> quad gamma = g(u) = u (s - u^3) / 2. $ <s-quartic>
$g(0) = 0$, $g$ rises to a maximum and falls to $0$ at $u = s^(1\/3)$: two positive roots for $0 < gamma < max g$, none above. $g'(u) = (s - 4 u^3)\/2 = 0$ at $(u^*)^3 = s\/4$, so
$ gamma^*(s) = g(u^*) = (s\/4)^(1\/3) dot (3 s\/4) / 2 = 3/8 thin 4^(-1\/3) s^(4\/3), quad gamma^*(1) = (3 dot 2^(1\/3))/16 = 0.23624, $
and the product just below $gamma^*$ is $p = (u^*)^3 = s\/4$. Followed from $gamma = 0$ (where $p = s$), the learned product decreases to $s\/4$ and then jumps to $0$: a discontinuous transition.

#runin[(c)] At $(u, u, u)$: $partial_a^2 L_gamma = (b c)^2 + 2 gamma = u^4 + 2 gamma$ and $partial_a partial_b L_gamma = c r + a b c^2 = u (u^3 - s) + u^4$. Eliminating $2 gamma = s u - u^4$ with @s-quartic: $alpha = s u$, $beta = 2 u^4 - s u$. The eigenvalues of $alpha I + beta (bold(1) bold(1)^top - I)$ are $alpha + 2 beta$ on $(1, 1, 1)$ and $alpha - beta$ (twice) across it:
$ lambda_parallel = u (4 u^3 - s), quad lambda_perp = 2 u (s - u^3) = 4 gamma > 0. $
So the larger root ($u^3 > s\/4$) is a minimum and the smaller root ($u^3 < s\/4$) is a saddle with one unstable direction, along the ray. At $gamma^*$ the two meet at $u^3 = s\/4$ where $lambda_parallel = 0$ and annihilate: a _saddle-node bifurcation_.

#runin[(d)] With $q = u^3$ and @s-quartic, $3 gamma u^2 = 3/2 q (s - q)$ and
$ L_gamma = 1/2 (q - s)^2 + 3/2 q (s - q) = s^2/2 + (q s)/2 - q^2. $
This equals $s^2\/2$ at $q = s\/2$, so $u_c = (s\/2)^(1\/3)$ and
$ gamma_c (s) = g(u_c) = s/4 (s/2)^(1\/3), quad gamma_c (1) = 2^(-1\/3) \/ 4 = 0.19843. $
On the minimum branch $q$ decreases with $gamma$, so for $gamma_c < gamma < gamma^*$ we have $q < s\/2$ and $L > s^2\/2$: the origin is the global minimum and the learned solution is only _metastable_. Gradient flow does not see $gamma_c$; it stays on the nonzero minimum until that minimum disappears at $gamma^*$.

#runin[(e) Expected prediction.] Going up, $a b c$ decreases smoothly to $1\/4$ and then drops to $0$ at $gamma^*$. Going down, the depth-3 net never learns again, down to $gamma -> 0$: the origin is a minimum for every $gamma > 0$, and a $10^(-3)$ kick is far inside its basin (radius about $2 gamma$, Problem (#raw("basins"))) except at the very smallest $gamma$. The depth-2 net retraces its curve exactly: its origin is a saddle below $1\/2$ and any kick escapes.

#runin[(f)] Reference: `solutions/depth3_solution.py`.

#runin[(g) Expected.] Hessian at the origin $"diag"(0.02, 0.02, 0.02)$, $(0.2, 0.2, 0.2)$, $(1, 1, 1)$ with zero off-diagonals. Learned $a b c$ on the larger root cubed to $2.3 times 10^(-10)$ away from $gamma^*$. Bisection $gamma^*$ against the formula: $0.0372048$ vs $0.0372047$ ($s = 0.25$), $0.0937501$ vs $0.0937500$, $0.2362353$ vs $0.2362352$, $0.5952752$ vs $0.5952754$ ($s = 2$), all within $2 times 10^(-6)$ relative; products just before the jump $0.0625, 0.1250, 0.2503, 0.5009$ against $s\/4$ (the last bisection point is slightly below $gamma^*$). Log-log slope of $gamma^*$ in $s$: $1.3333 = 4\/3$. Ziyin et al.’s Proposition 3 (their (15)) has $norm(EE[x y])^(1 + (D-1)\/(D+1))$, which is $s^(4\/3)$ at $D = 2$: the same power. At depth 2 the threshold is $s\/2$, linear in $s$, so weak signals lose much sooner at depth 3. Sweep: depth 2 gives $p = 0.800, 0.400$ at $gamma = 0.1, 0.3$ in both directions; depth 3 gives $0.783$ and $0$ going up and $0$, $0$ coming down.

#figure(image("figures/depth3.png", width: 100%), caption: [`depth3.png`. Left: learned product from $(1, 1, 1)$. Middle: adiabatic sweeps; the depth-3 loop never closes. Right: $gamma^*(s)$ against the depth-2 threshold $s\/2$.])

= Problem (#raw("basins")): Basins of the zero solution

#runin[(a)] $L_gamma$ is symmetric under permutations of $(a, b, c)$, so at a point with $a = b = c$ the three gradient components are equal and the flow stays on the ray. There
$ dot(u) = F(u) = -u (u^4 - s u + 2 gamma). $
The bracket equals $2 gamma > 0$ at $u = 0$, is positive for all $u < 0$ (all three terms are), and for $gamma < gamma^*$ is negative exactly between the roots $u_s < u_+$ of @s-quartic. Phase line: $F > 0$ for $u < 0$, $F < 0$ on $(0, u_s)$, $F > 0$ on $(u_s, u_+)$, $F < 0$ above $u_+$. Endpoint rule: $u_0 <= 0 -> 0$; $0 < u_0 < u_s -> 0$; $u_0 = u_s$ stays; $u_0 > u_s -> u_+$. For $gamma > gamma^*$ every start goes to $0$. For small $gamma$ the root $u_s$ is small, the $u^4$ term is negligible, and
$ u_s approx (2 gamma)/s quad ("next term:" u_s = (2 gamma)/s + (2 gamma)^4/s^5 + dots). $
The zero basin's radius grows linearly in the weight decay and inversely in the signal, the $gamma \/ norm(EE[x y])$ scaling of Ziyin et al.’s bound (16) at $D = 2$.

#runin[(b)] Swapping $a$ and $b$ is a symmetry, so the plane $a = b$ is invariant. In it the basin boundary is the stable manifold of the balanced saddle $(u_s, u_s, u_s)$: in 3-D it is a surface (the saddle has two stable directions, eigenvalue $-4 gamma$ of the flow's Jacobian, and one unstable), and the plane cuts it in a curve through the saddle tangent to $(1, 1, -2)$. Flipping the signs of two weights, $(a, b, c) -> (-a, -b, c)$, leaves $a b c$ and the loss unchanged and maps the plane to itself by $(x, y) -> (-x, y)$; it takes the saddle to $(-u_s, -u_s, u_s)$, whose stable manifold is the mirrored boundary.

#runin[(c)] Reference: `solutions/basins_solution.py`.

#runin[(d) Expected.] Balanced starts: maximum error $1.3 times 10^(-11)$, measured edge between $0.200$ and $0.210$ around $u_s = 0.20165$ ($gamma = 0.1$); $67%$ of the plane's grid learns. Gaussian starts at $gamma = 0.03$: $u_s = 0.0600$ for $s = 1$ and $0.1513$ for $s = 0.4$; the learning fraction is $0.44$ at $sigma = 0.100$ and $0.65$ at $0.133$ for $s = 1$, and $0.40$ at $0.237$, $0.57$ at $0.316$ for $s = 0.4$, so $sigma_(1\/2) approx 0.11$ and $0.28$: about $1.8 u_s$ in both cases. Exact over small-$gamma$ radius: $1.0000$ up to $gamma = 0.01$, $1.0002$ at $0.03$, $1.0083$ at $0.1$.

#figure(image("figures/basins.png", width: 100%), caption: [`basins.png`. Left: endpoints on the balanced ray. Middle: the plane $a = b$, blue learns, grey ends at zero, with the saddles' stable manifolds. Right: fraction of Gaussian starts that learn, with the basin radius dotted.])

= Problem (#raw("finite-step")): The phase diagram of GD with weight decay

#runin[(a)] Near a fixed point, $theta_(k+1) - theta^* approx (I - eta H)(theta_k - theta^*)$, which contracts iff $|1 - eta lambda| < 1$, i.e. $0 < lambda < 2\/eta$, for every eigenvalue. With (stationary) (a):
$ eta_"max" (gamma) = cases(2 \/ (2 - 4 gamma) = 1 \/ (1 - 2 gamma) quad & "if" 0 < gamma <= 1/4,
  2 \/ (4 gamma) = 1 \/ (2 gamma) & "if" 1/4 <= gamma < 1/2,
  2 \/ (2 gamma + 1) & "if" gamma > 1/2.) $ <s-etamax>
The direction that loses stability is $(1, 1)$ at the minimum on the first piece, $(1, -1)$ at the minimum on the second, and $(1, -1)$ at the origin on the third.
The maximum is $eta_"max" (1/4) = 2$, where the two eigenvalues at the minimum are equal (both $1$). The limits at $gamma -> 1/2$ are $1$ from both sides, so the boundary is continuous across the transition, and $eta_"max" -> 1$ as $gamma -> 0$ (the $(1, 1)$ minimum of the plain anchor has sharpness $a^2 + b^2 = 2$, A3). For $gamma < 1/2$ the origin cannot rescue GD: its eigenvalue $2 gamma - 1 < 0$ makes $|1 - eta lambda| > 1$ for every $eta$.

#runin[(b) Expected prediction.] A band of bounded oscillation only for $gamma < 1/4$. There the unstable direction is $(1, 1)$, along the product, where the cubic nonlinearity folds the iterates back (the same mechanism as edge-of-stability oscillations in A3). For $gamma > 1/4$ the unstable direction is the imbalance: at the minimum $r = -2 gamma$ and A1’s factor is $c_(k+1)\/c_k = (1 - 2 eta gamma)^2 - eta^2 r^2 = 1 - 4 eta gamma$, which drops below $-1$ exactly at $eta = 1\/(2 gamma)$. Then $|c|$ grows, $a^2 + b^2 >= |c|$ grows, the curvature grows with it, and nothing saturates: divergence. Likewise past $2\/(2 gamma + 1)$ at the origin.

#runin[(c)] On $a = b = u$ the GD map is $u |-> u - eta u (u^2 - s)$, $s = s_"eff"$. Its derivative at $u^* = sqrt(s)$ is $1 - eta (3 (u^*)^2 - s) = 1 - 2 eta s$, so the fixed point is stable iff $eta < 1\/s$: oscillation onset $eta = 1\/(1 - 2 gamma)$ (the first piece of @s-etamax, as it should be). With $u = k v$ and $eta k^2 = mu = 1 + eta s$, the map is $v |-> mu (v - v^3)$. At $v_e = sqrt(1 + 1\/mu)$, $mu (v_e - v_e^3) = mu v_e (1 - v_e^2) = -v_e$, so the endpoints of $[-v_e, v_e]$ map to each other, and outside it $|f(v)| > |v|$ grows without bound. The interval maps into itself iff the local maximum $f(1\/sqrt(3)) = 2 mu \/ (3 sqrt(3))$ is at most $v_e$:
$ (4 mu^2)/27 <= 1 + 1/mu quad <=> quad 4 mu^3 - 27 mu - 27 = (mu - 3)(2 mu + 3)^2 <= 0 quad <=> quad mu <= 3. $
For $mu > 3$ the top of the map pokes out of the interval, and almost every orbit eventually lands outside and escapes. With $mu = 1 + eta s$: divergence onset $eta = 2\/s = 2\/(1 - 2 gamma)$. Between $1\/s$ and $2\/s$ the map goes through period doubling and chaos, the same route as the logistic map.

#runin[(d)] Reference: `solutions/finite_step_solution.py`.

#runin[(e) Expected.] From $(0.6, 0.4)$: 2746 cells converge to the nonzero minimum, 1594 to zero, 818 oscillate and 4842 diverge; agreement with `predicted_phase` is $1.0000$ on the 9739 cells more than $3%$ from the boundary and $0.9998$ overall. The two disagreeing cells ($gamma = 0.525$, $eta = 0.975$ and $gamma = 0.675$, $eta = 0.850$) sit within $0.3%$ below $eta_"max"$ and diverge: the origin is linearly stable there, but its basin shrinks to nothing at the boundary and $(0.6, 0.4)$ is outside it. Oscillating cells occur only for $gamma in [0.005, 0.245]$, below $1\/4$. Their upper edge (between $eta approx 2$ and $2.5$) is ragged: it is set by when a chaotic orbit lands outside the trapping region, which the linear analysis does not predict, and from an unbalanced start it lies below the balanced line's $2\/(1 - 2 gamma)$ for larger $gamma$ (the imbalance is not stable along a chaotic orbit). From $(0.45, 0.45)$: both onsets match (c) within $0.8%$ and $0.4%$, the $eta$ grid step. The post's caption calls the stable-solution boundaries analytic and the oscillation and divergence regions numerical; here the stability boundaries are @s-etamax and the line $gamma = 1/2$, and on the balanced line the divergence edge is analytic too.

#figure(image("figures/finite_step.png", width: 96%), caption: [`finite_step.png`. Left: the phase diagram of GD from $(0.6, 0.4)$ with $eta_"max" (gamma)$. Right: the balanced line, with the oscillation and divergence onsets.])

= Problem (#raw("ziyin")): Exact solutions of a deep linear network

#runin[(a)] $EE_epsilon [1/2 (a epsilon b - 1)^2] = 1/2 ((1 + sigma^2) a^2 b^2 - 2 a b + 1) = 1/2 (a b - 1)^2 + sigma^2/2 thin a^2 b^2$: the noise is a penalty on the _product_ (on the function), not on the weights. Using balance, $a^2 + b^2 >= 2|a b|$ with equality at the optimum, so minimize $1/2 (p - 1)^2 + sigma^2 p^2\/2 + 2 gamma p$ over $p >= 0$: $p_infinity = (1 - 2 gamma)\/(1 + sigma^2)$ for $gamma < 1/2$ and $0$ above. The critical $gamma$ stays at $1/2$ because the noise term is quartic in the weights and does not touch the Hessian at the origin; it only shrinks the learned product. This is their remark in §5 that the two-layer existence condition $norm(EE[x y])^2 > gamma_u gamma_w$ is independent of the noise (and of width).

#runin[(b)] At the origin the Hessian only sees the quadratic part of the loss. With one hidden layer, $-s thin a b$ is quadratic and gives off-diagonal entries $-s$; when the signal beats the weight decay ($s > 2 gamma$ here) they make the origin a saddle, otherwise it is the global minimum (Theorem 1). With two or more hidden layers the signal term $-s thin a b c$ is cubic or higher, the Hessian is $2 gamma I$, and the origin is a local minimum for every $gamma > 0$ (Proposition 2), a bad one whenever a nonzero solution has lower loss ($gamma < gamma_c$ here).

#runin[(c)] The Gaussian-start fractions in `check_basins.py` part C: starts of scale $sigma$ below about $u_s$ almost never learn, and $u_s approx 2 gamma \/ s$ grows as the signal weakens, so a fixed init scale that works for $s = 1$ fails for $s = 0.4$. Ziyin et al.’s bound (16) puts the nonzero solutions at radius at least about $(gamma\/norm(EE[x y]))^(1\/(D-1))\/d_0$, which for $D = 2$, $d_0 = 1$ is the same $gamma \/ s$ scaling; Kaiming’s $1\/sqrt(d_0)$ radius sits inside it when the signal is weak (their Figure 3).
