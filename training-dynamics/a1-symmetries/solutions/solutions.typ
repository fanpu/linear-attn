#import "/template/template.typ": *

#show: doc.with(
  assignment: 1,
  title: "Symmetries and conservation laws",
  kind: "Worked solutions",
  author: "Training dynamics, a self-study course · written by Claude for Fan Pu",
  date: "October 29, 2026",
  banner: "reference numbers: solutions/results.json · figures: solutions/figures/ · equation numbers (n) refer to the handout",
)

Every derivation here is complete; read only as far as you need. "Measured" numbers come from running each check script with the reference implementations (`--solutions`) on CPU; your numbers should agree within the tolerances stated in each problem. Diff yours with `uv run python -m tdyn.results a1-symmetries/out/results.json a1-symmetries/solutions/results.json`. Equation numbers such as (2) refer to the handout; this document numbers its own equations (S1), (S2), ... Every formula below was also checked symbolically (sympy) or numerically before it was written down.

#set math.equation(numbering: n => "(S" + str(n) + ")")

= Problem (#raw("noether")): Rescaling symmetry and the conserved imbalance

#runin[(a)] The generator of (2) is $X(a, b) = partial_alpha (alpha a, b\/alpha)|_(alpha = 1) = (a, -b)$. Differentiating $L(alpha a, b\/alpha) = L(a, b)$ at $alpha = 1$ by the chain rule:
$ ⟨nabla L(theta), X(theta)⟩ = a thin partial_a L - b thin partial_b L = 0 quad "for every" theta. $ <s-constraint>
Along gradient flow, with $Q = a^2 - b^2 = c$,
$ dot(Q) = 2 a dot(a) - 2 b dot(b) = -2 (a thin partial_a L - b thin partial_b L) = -2 ⟨nabla L, X⟩ = 0. $
Note $nabla Q = 2 X$: the conserved quantity is the function whose gradient is the generator, which works because $X$ here is itself a gradient field (that is the "Noether" step; Kunin et al.'s Theorem 1 states the general condition). Any function of $c$ is also conserved. The argument used only that $L$ is invariant under (2), which holds for every $L(a, b) = ell(a b)$: then $nabla L = ell'(a b)(b, a)$ and $⟨nabla L, X⟩ = ell'(a b)(a b - b a) = 0$. This covers any per-example loss of $f_(a,b)(x) = a b x$ on any dataset (squared, logistic, cross-entropy), since the data enter only through the scalar $a b$.

#runin[(b)] Differentiate @s-constraint, $⟨nabla L(theta), X(theta)⟩ = 0$, with respect to $theta$. The product rule gives
$ nabla^2 L(theta) thin X(theta) + (partial X \/ partial theta)^top nabla L(theta) = 0, quad "here" quad nabla^2 L thin X = -"diag"(1, -1) nabla L. $ <s-hessid>
At any minimum $nabla L = 0$, so $nabla^2 L thin X = 0$: the generator is a zero-eigenvalue direction of the Hessian. On $a b = 1$, $X = (a, -1\/a) = a (1, -1\/a^2)$, which is exactly A0's zero-eigenvalue eigenvector, the tangent to the curve of minima. The symmetry _is_ the flat direction: moving along the orbit of $psi_alpha$ moves along the minima without changing the loss.

#runin[(c)] With $a_0, b_0$ i.i.d. $cal(N)(0, sigma^2)$, $EE[a^4] = 3 sigma^4$, so
$ EE[c^2] = EE[a^4] - 2 EE[a^2] EE[b^2] + EE[b^4] = 6 sigma^4 - 2 sigma^4 = 4 sigma^4, quad sqrt(EE[c^2]) = 2 sigma^2. $
At the endpoint $a_infinity^2 - b_infinity^2 = (a_infinity - b_infinity)(a_infinity + b_infinity) = c$ and $(a_infinity + b_infinity)^2 = a_infinity^2 + b_infinity^2 + 2 = sqrt(c^2 + 4) + 2 = 4 + O(c^2)$, so
$ a_infinity - b_infinity = c / sqrt(sqrt(c^2 + 4) + 2) = c/2 + O(c^3). $
The init scale enters squared: an init of scale $10^(-2)$ leaves the two layers equal to about $10^(-4)$ at the end of training. This is Du, Hu and Lee's "layers are automatically balanced".

#runin[(d)] Reference: `solutions/noether_solution.py` (three one-liners on `theta[..., 0]`, `theta[..., 1]`).

#runin[(e) Expected.] $|⟨nabla L, X⟩| \/ (norm(nabla L) norm(X)) approx 1.3 times 10^(-16)$ for the three losses of $a b$ and $0.998$ for the skip connection. $max_t |Q(t) - Q(0)|$ on $[0, 10]$: $5.0 times 10^(-11)$, $1.4 times 10^(-10)$, $3.2 times 10^(-11)$ (integrator error) for the three flows, $0.103$ for the skip connection and $0.096$ for GD at $eta = 0.1$. Hessian identity residual $1.0 times 10^(-9)$ (finite-difference error), $norm(nabla^2 L thin X) = 4.6 times 10^(-16)$ on $a b = 1$. Root-mean-square $c$: $2.03, 18.1, 202, 1811 (times 10^(-4))$ and $1.98$ against $2 sigma^2$; the endpoint gap matches $c\/2$ to $0.5%$ for $sigma <= 0.3$ and is $0.66$ against $0.80$ at $sigma = 1$, where $c$ is not small.

#figure(image("figures/noether.png", width: 92%), caption: [`noether.png`. Left: $|Q(t) - Q(0)|$ along four flows and one GD run. Right: the size of $c$ and of the endpoint gap against the init scale.])

= Problem (#raw("endpoint")): Where gradient flow lands

#runin[(a)] Write $A = a_infinity^2$. Conservation gives $b_infinity^2 = A - c$ and the endpoint gives $a_infinity^2 b_infinity^2 = s^2$, so $A^2 - c A - s^2 = 0$. Only the root with $+$ is positive:
$ a_infinity^2 = (c + sqrt(c^2 + 4 s^2)) / 2, quad b_infinity^2 = (-c + sqrt(c^2 + 4 s^2)) / 2. $
Signs: if $c > 0$ then $a^2 = c + b^2 >= c > 0$ for all $t$, so $a$ never crosses zero and keeps the sign of $a_0$; $b_infinity$ has the same sign since $a_infinity b_infinity = s > 0$. If $c < 0$, likewise $b$ keeps its sign. In both cases $"sign"(a_0) = "sign"(a_0 + b_0)$ (resp. $"sign"(b_0)$), because the larger of $|a_0|, |b_0|$ wins the sum. If $c = 0$ the start is on $a = b$ (stays there, ends at $plus.minus sqrt(s)$ with the sign of $a_0 + b_0$) or on $a = -b$ (A0: flows to the origin). One rule covers everything:
$ (a_infinity, b_infinity) = "sign"(a_0 + b_0) thin (sqrt((c + sqrt(c^2 + 4 s^2))\/2), sqrt((-c + sqrt(c^2 + 4 s^2))\/2)), $
with $"sign"(0) = 0$ giving the origin. (Convergence to $a b = s$ for $c != 0$: $dot(p) = sqrt(c^2 + 4 p^2)(s - p)$ with a strictly positive square root; see Problem (#raw("logistic")) (e).)

#runin[(b)] Reference: `solutions/endpoint_solution.py`.

#runin[(c) Expected.] Maximum error $1.0$–$1.2 times 10^(-10)$ over 150 starts for each of $s = 0.5, 1, 2$. From $(0.7, -0.7 + epsilon)$, the loss falls below $1\/4$ at $t = 4.67, 9.27, 13.87$ for $epsilon = 10^(-2), 10^(-4), 10^(-6)$: each factor $100$ in $epsilon$ adds $4.6 = ln 100$. The start sits near the origin's stable direction $a = -b$ (eigenvalue $-1$), is pulled to the saddle, and leaves along $a = b$ at rate $+1$ starting from a component of size about $epsilon$, so it needs time $ln(1\/epsilon)$.

#figure(image("figures/endpoint.png", width: 85%), caption: [`endpoint.png`. Left: 40 flows for $s = 1$, coloured by $"sign"(a_0 + b_0)$, with the predicted endpoints. Right: predicted against integrated $a_infinity$ for all 450 starts.])

= Problem (#raw("logistic")): Balanced learning is logistic

#runin[(a)] For $L = 1/2 (s - a b)^2$: $dot(a) = b(s - a b)$, $dot(b) = a(s - a b)$, so $d/(d t)(a - b) = -(a - b)(s - a b)$. This is linear in $a - b$; with $a_0 - b_0 = 0$ the unique solution is $a - b equiv 0$. On $a = b$, $u = a b = a^2$ and
$ dot(u) = dot(a) b + a dot(b) = (a^2 + b^2)(s - u) = 2 u (s - u). $
Separate and use partial fractions, $integral d u \/ (u(s - u)) = 1/s ln(u\/(s - u))$:
$ 1/(2 s) ln (u (s - u_0)) / (u_0 (s - u)) = t quad => quad u(t) = (s u_0 e^(2 s t)) / (s + u_0 (e^(2 s t) - 1)) = s / (1 + (s\/u_0 - 1) e^(-2 s t)). $ <s-logistic>
The second form never overflows. This is Saxe et al.'s eq. (12) with $tau = 1$.

#runin[(b)] Solving @s-logistic for $u(t_f) = f s$:
$ t_f (u_0, s) = 1/(2 s) ln (f (s - u_0)) / ((1 - f) u_0), quad t_(1\/2) = 1/(2 s) ln (s - u_0)/u_0 = 1/(2 s) ln s/u_0 + O(u_0 \/ s). $
_Plateau_ (time to reach $0.1 s$): $1/(2 s) ln(s\/(9 u_0))$, logarithmic in $1\/u_0$ and inversely proportional to $s$. _Drop_ (from $0.1 s$ to $0.9 s$): $1/(2 s)(ln 9 - ln(1\/9)) = ln 81 \/ (2 s) = 2.20\/s$, independent of $u_0$. For $s = 1$, $u_0 = 10^(-6)$: plateau $5.82$, drop $2.20$. Smaller init lengthens only the plateau, so the curve looks ever more like a step.

#runin[(c)] Reference: `solutions/logistic_solution.py`.

#runin[(d) Expected.] Maximum relative error of $t_(1\/2)$ over 60 pairs: $1.2 times 10^(-10)$. Fitted slopes of $t_(1\/2)$ against $ln(1\/u_0)$: $1.0065, 0.5015, 0.2504, 0.1251$ for $s = 0.5, 1, 2, 4$, i.e. $1\/(2 s)$ (the small excess at $s = 0.5$ is the $O(u_0\/s)$ term at $u_0 = 0.1$).

#runin[(e)] Expected prediction: faster. In general $dot(p) = dot(a) b + a dot(b) = (a^2 + b^2)(s - p)$ and $(a^2 + b^2)^2 = (a^2 - b^2)^2 + 4 a^2 b^2$, so
$ dot(p) = sqrt(c^2 + 4 p^2) thin (s - p). $
For $p << |c|$ the rate is about $|c| s$, a constant push, instead of $2 p s$, which is tiny at small $p$; the plateau is the time the balanced net spends with a vanishing rate. Imbalance matters once $|c|$ is comparable to $2 p_0 = 2 times 10^(-4)$. Measured half-times at $p_0 = 10^(-4)$, $s = 1$: $4.61$ ($c = 0$, $= ln(10^4)\/2$), $4.58, 4.05, 2.98, 2.44, 1.82, 1.23, 0.60$ for $c = 10^(-4), 10^(-3), 10^(-2), 0.03, 0.1, 0.3, 1$. The kernel of the anchor is $a^2 + b^2 >= |c|$ (A5), so imbalance also keeps it from starting at zero.

#figure(image("figures/logistic.png", width: 100%), caption: [`logistic.png`. Left: half-time against $ln(1\/u_0)$, slopes $1\/(2s)$. Middle: $u(t)\/s$ for $u_0 = 10^(-6)$. Right: half-time against the imbalance at fixed $a_0 b_0 = 10^(-4)$.])

= Problem (#raw("matrix-law")): The matrix conservation law

#runin[(a)] $L(W) = 1/2 "tr"(W Sigma_(x x) W^top) - "tr"(W Sigma_(y x)^top)$ has $G = nabla_W L = W Sigma_(x x) - Sigma_(y x)$. Perturbing $W_1 -> W_1 + E$ changes $W$ by $W_2 E$ and $L$ by $"tr"(G^top W_2 E) = ⟨W_2^top G, E⟩$ to first order; perturbing $W_2 -> W_2 + E$ changes $W$ by $E W_1$ and $L$ by $⟨G W_1^top, E⟩$:
$ nabla_(W_1) L = W_2^top G, quad nabla_(W_2) L = G W_1^top. $

#runin[(b)] Under gradient flow $dot(W_1) = -W_2^top G$, $dot(W_2) = -G W_1^top$, so
$ dot(D) &= dot(W_1) W_1^top + W_1 dot(W_1)^top - dot(W_2)^top W_2 - W_2^top dot(W_2) \
  &= -W_2^top G W_1^top - W_1 G^top W_2 + W_1 G^top W_2 + W_2^top G W_1^top = 0. $
Only the form of the two gradients was used, so any loss of the product $W_2 W_1$ works. Noether-style: $L(A W_1, W_2 A^(-1))$ is constant; with $A = I + epsilon M$, $A^(-1) = I - epsilon M + O(epsilon^2)$, and differentiating at $epsilon = 0$,
$ ⟨nabla_(W_1) L, M W_1⟩ - ⟨nabla_(W_2) L, W_2 M⟩ = "tr"(M (W_1 nabla_(W_1) L^top - nabla_(W_2) L^top W_2)) = 0 quad "for all" M, $
which forces the $m times m$ matrix $W_1 nabla_(W_1) L^top - nabla_(W_2) L^top W_2$ to vanish (here both terms equal $W_1 G^top W_2$). Then $dot(D) = -(nabla_(W_1) L thin W_1^top + W_1 nabla_(W_1) L^top) + (nabla_(W_2) L^top W_2 + W_2^top nabla_(W_2) L) = 0$ by the constraint and its transpose: one conserved number per entry of $D$, one per generator $M$. For $m = d_"in" = d_"out" = 1$, $M$ is a scalar, the constraint is $a partial_a L - b partial_b L = 0$, and $D = a^2 - b^2$.

#runin[(c)] With $g_i = nabla_(W_i) L$: $W_1' W_1'^top = W_1 W_1^top - eta (g_1 W_1^top + W_1 g_1^top) + eta^2 g_1 g_1^top$ and $W_2'^top W_2' = W_2^top W_2 - eta (g_2^top W_2 + W_2^top g_2) + eta^2 g_2^top g_2$. The two first-order terms are both $W_2^top G W_1^top + W_1 G^top W_2$, so they cancel _exactly_, and
$ D' - D = eta^2 (nabla_(W_1) L thin nabla_(W_1) L^top - nabla_(W_2) L^top nabla_(W_2) L). $ <s-dstep>
Scalar case: $eta^2 (b^2 r^2 - a^2 r^2) = -eta^2 r^2 c$, i.e. $c' = c(1 - eta^2 r^2)$, the post's §3.1 formula.

#runin[(d)] Reference: `solutions/two_layer_solution.py`.

#runin[(e) Expected.] Flow drift $5.3 times 10^(-13)$; @s-dstep matches every GD step to $1.1 times 10^(-15)$; drift at $t = 10$: $0.261, 0.115, 0.0545, 0.0212, 0.0105, 0.00521$ for $eta = 0.2, dots, 0.005$, fitted slope $1.05$. Prediction $p = 1$: each step adds $O(eta^2)$ and reaching $t = 10$ takes $10\/eta$ steps.

#figure(image("figures/matrix_law.png", width: 85%), caption: [`matrix_law.png`: drift of $D$ along the flow and GD (left), and at $t = 10$ against $eta$ (right).])

= Problem (#raw("saxe")): Deep linear networks learn mode by mode

#runin[(a)] With $Sigma_(x x) = I$ the flow is $dot(W_1) = W_2^top (Sigma_(y x) - W_2 W_1)$, $dot(W_2) = (Sigma_(y x) - W_2 W_1) W_1^top$. Multiply by the orthogonal $U^top$, $V$: with $S = U^top Sigma_(y x) V$ (diagonal, $s_alpha$ on the diagonal),
$ dot(overline(W)_1) = overline(W)_2^top (S - overline(W)_2 overline(W)_1), quad dot(overline(W)_2) = (S - overline(W)_2 overline(W)_1) overline(W)_1^top. $
Ansatz: $overline(W)_1 = R thin "diag"(a_alpha)$, $overline(W)_2 = "diag"(b_alpha) R^top$ (padded with zeros for the other directions). Then $overline(W)_2 overline(W)_1 = "diag"(b_alpha) R^top R "diag"(a_alpha) = "diag"(a_alpha b_alpha)$ because $R^top R = I$, so $S - overline(W)_2 overline(W)_1 = "diag"(s_alpha - a_alpha b_alpha)$ and
$ dot(overline(W)_1) = R thin "diag"(b_alpha (s_alpha - a_alpha b_alpha)), quad dot(overline(W)_2) = "diag"(a_alpha (s_alpha - a_alpha b_alpha)) R^top. $
The velocity has the ansatz's form, so (by uniqueness of solutions) the ansatz holds for all time with
$ dot(a)_alpha = b_alpha (s_alpha - a_alpha b_alpha), quad dot(b)_alpha = a_alpha (s_alpha - a_alpha b_alpha), $
the anchor with target $s_alpha$. Modes do not interact because the cross terms $r_alpha^top r_beta$ vanish (Saxe's competitive terms in their eq. (5) are zero on this manifold). Undoing the rotation, $W = U thin "diag"(a_alpha b_alpha) V^top$, which is (6).

#runin[(b)] The init is balanced ($a_alpha = b_alpha = sqrt(u_(0 alpha))$), so each mode is Problem (#raw("logistic")):
$ u_alpha (t) = s_alpha / (1 + (s_alpha \/ u_(0 alpha) - 1) e^(-2 s_alpha t)), quad t_(1\/2, alpha) / t_(1\/2, beta) = (s_beta ln(s_alpha\/u_0)) / (s_alpha ln(s_beta \/u_0)) approx s_beta / s_alpha. $
Strong modes are learned first, at a time inversely proportional to their singular value (up to the logarithm). In the $2 times 2$ example the "subtract" mode comes about $3$ times later.

#runin[(c)] Reference: `solutions/saxe_solution.py` (`R` = the first $r$ columns of $I_m$).

#runin[(d)] Linearize: $dot(x) = dot(W_1) bold(v)_alpha = W_2^top (Sigma_(y x) - W) bold(v)_alpha approx W_2^top Sigma_(y x) bold(v)_alpha = s_alpha W_2^top bold(u)_alpha = s_alpha y$, and likewise $dot(y) = dot(W_2)^top bold(u)_alpha approx W_1 Sigma_(y x)^top bold(u)_alpha = s_alpha x$. Hence $x + y = (x_0 + y_0) e^(s_alpha t)$ and $x - y = (x_0 - y_0) e^(-s_alpha t)$, and
$ u_alpha = x^top y = 1/4 (norm(x + y)^2 - norm(x - y)^2) approx 1/4 norm(x_0 + y_0)^2 e^(2 s_alpha t), quad u_0^"eff" = 1/4 norm(W_1(0) bold(v)_alpha + W_2(0)^top bold(u)_alpha)^2. $
For entries of variance $sigma^2$ this is about $m sigma^2 \/ 2$. A random init does not start on the aligned manifold, but the growing direction $x + y$ aligns it within the plateau; the decaying part is forgotten.

#runin[(e) Expected.] Aligned init: $max |W_"flow" - W_"pred"| = 1.7 times 10^(-9)$ and half-times $1.900, 3.627, 6.908, 13.122$ equal to the formula. Random init ($u_0^"eff" = 5.3, 5.3, 1.4, 2.0 times 10^(-6)$): measured half-times $1.691, 3.223, 6.852, 12.792$ against predicted $1.691, 3.209, 6.725, 12.423$, relative delays $0.0%, 0.4%, 1.9%, 3.0%$, growing for the weaker modes as in Saxe et al.'s Fig. 3 (right). $s_alpha t_(1\/2)$ stays between $6.4$ and $6.9$: the time scale is $1\/s_alpha$, the rest is the logarithm.

#figure(image("figures/saxe.png", width: 92%), caption: [`saxe.png`. Left: mode strengths from a random init against the prediction with $u_0^"eff"$. Right: the loss, a staircase of plateaus and drops.])

= Problem (#raw("broken")): Finite steps, momentum, minibatches and weight decay

#runin[(a)] Write (7) as $a' = (1 - 2 eta gamma) a - eta b r$, $b' = (1 - 2 eta gamma) b - eta a r$. Then
$ a'^2 - b'^2 = (1 - 2 eta gamma)^2 (a^2 - b^2) - 2 (1 - 2 eta gamma) eta r (a b - b a) + eta^2 r^2 (b^2 - a^2), $
$ c_(k+1) = c_k [(1 - 2 eta gamma)^2 - eta^2 r_k^2]. $ <s-cstep>
With $gamma = 0$, $|c|$ cannot grow iff $|1 - eta^2 r_k^2| <= 1$, i.e. $eta |r_k| <= sqrt(2)$; for $eta |r_k| > sqrt(2)$ the factor is below $-1$ and $c$ flips sign and grows (the post's diverging example). Under the flow on $L_gamma$, $dot(a) = -b r - 2 gamma a$, $dot(b) = -a r - 2 gamma b$, and
$ dot(c) = 2 a dot(a) - 2 b dot(b) = -4 gamma c, quad c(t) = c_0 e^(-4 gamma t). $
(The brief's convention $dot(a) = -partial_a L - gamma a$ gives $-2 gamma c$; Kunin et al.'s $L + lambda\/2 norm(theta)^2$ gives $-2 lambda c$, App. E.) Expanding @s-cstep: $(1 - 2 eta gamma)^2 approx e^(-4 eta gamma)$, the flow's decay over a time $eta$.

#runin[(b)] Along (8), $dot(c) = -2⟨X, nabla L⟩ - eta ⟨X, nabla^2 L nabla L⟩$ with $X = (a, -b)$. The first term vanishes by @s-constraint. For the second, @s-hessid gives $⟨X, nabla^2 L nabla L⟩ = ⟨nabla^2 L X, nabla L⟩ = -⟨"diag"(1, -1) nabla L, nabla L⟩ = -(b^2 r^2 - a^2 r^2) = r^2 c$. So
$ dot(c) = -eta r^2 c. $ <s-mflowc>
(Directly: $nabla^2 L nabla L = r (b^3 + (2 a b - 1) a, (2 a b - 1) b + a^3)$, dotted with $(a, -b)$ gives $r(a b (a^2 - b^2) - (a^2 - b^2)) = r^2 c$.) This is Kunin et al.'s rescale result (App. E, eq. (30)) with $|g_a|^2 - |g_b|^2 = r^2 (b^2 - a^2)$.

#runin[(c)] Dividing @s-mflowc by $c$ and integrating, $ln(c_infinity \/ c_0) = -eta integral_0^infinity r^2 thin d t$; to first order in $eta$ the integral may be taken along gradient flow. There $dot(p) = sqrt(c^2 + 4 p^2)(1 - p) = -r sqrt(c^2 + 4 p^2)$, monotone from $p_0$ to $1$, so $r^2 d t = r^2 d p \/ dot(p) = (1 - p) d p \/ sqrt(c^2 + 4 p^2)$ and
$ I(p_0, c) = integral_(p_0)^1 (1 - p) / sqrt(c^2 + 4 p^2) d p = 1/2 ["arcsinh"(2/(|c|)) - "arcsinh"((2 p_0)/(|c|))] - 1/4 [sqrt(c^2 + 4) - sqrt(c^2 + 4 p_0^2)], $
$ ln(c_infinity \/ c_0) approx -eta thin I(p_0, c_0). $
Scaling: one step changes $ln c$ by $ln(1 - eta^2 r_0^2) approx -eta^2 r_0^2$, slope $2$ in $eta$; a whole run takes about $1\/eta$ steps (to a fixed time), so the total is $O(eta)$, slope $1$. The drift is always negative: finite steps shrink $|c|$, pushing toward balance, and more so the longer the run spends with large $r$ (for $p_0, |c| << 1$, $I approx 1/2 ln(4\/|c|) - 1/2$, which grows as $c -> 0$).

#runin[(d)] Take the inner product of (9) with $X = (a, -b)$. Since $⟨X, dot(theta)⟩ = dot(c)\/2$, $⟨X, dot.double(theta)⟩ = 1/2 dot.double(c) - (dot(a)^2 - dot(b)^2)$ and $⟨X, nabla L⟩ = 0$:
$ eta (1 + beta)/2 [1/2 dot.double(c) - (dot(a)^2 - dot(b)^2)] + (1 - beta)/2 dot(c) = 0. $
To leading order $dot(theta) = -nabla L \/ (1 - beta)$, so $dot(a)^2 - dot(b)^2 = r^2 (b^2 - a^2) \/ (1 - beta)^2 = -r^2 c \/ (1 - beta)^2$, and
$ dot(c) = -eta (1 + beta)/(1 - beta)^3 r^2 c - eta (1 + beta)/(2 (1 - beta)) dot.double(c) + O(eta^2). $
The last term integrates to $dot(c)(infinity) - dot(c)(0) = 0$ ($dot(theta)(0) = 0$). In $t$, the leading-order trajectory is gradient flow run at speed $1\/(1 - beta)$, so $integral r^2 d t = (1 - beta) I(p_0, c_0)$. Hence
$ ln(c_infinity \/ c_0) approx -eta (1 + beta)/(1 - beta)^2 I(p_0, c_0), $
a factor $(1 + beta)\/(1 - beta)^2$ over GD at the same $eta$; in terms of the effective step $eta_"eff" = eta\/(1 - beta)$ it is $(1 + beta)\/(1 - beta)$ times GD at $eta_"eff"$. Momentum breaks the law more than plain GD with the same effective step, because of the inertia term.

#runin[(e)] Per-example gradient: $partial_a 1/2 (a b x_i - x_i)^2 = b x_i^2 r$, so the batch gradient is $m_B r (b, a)$ with $m_B = 1/B sum_(i in B) x_i^2$: a multiple of $(b, a)$, orthogonal to $X$. One step is the GD step with $eta$ replaced by $eta m_B$:
$ c_(k+1) = c_k (1 - eta^2 m_B^2 r_k^2). $
To first order the drift is $-eta^2 sum_k EE[m_B^2] r_k^2 approx -eta EE[m_B^2] I$ (the mean path is the flow since $EE[m_B] = 1$), and for $B$ draws with replacement $EE[m_B^2] = 1 + "Var"(x_i^2)\/B$. So SGD multiplies the drift by $1 + "Var"(x^2)\/B$; for Gaussian $x$ that is $1 + 2\/B$.

#runin[(f)] Reference: `solutions/drift_solution.py`.

#runin[(g) Expected.] (A) @s-cstep exact to $9 times 10^(-16)$. (B) weight decay: relative error $2 times 10^(-11)$ ($gamma = 0.01$) and $2 times 10^(-8)$ ($gamma = 0.05$, after $c$ has decayed by $e^(-8)$). (C) $I(0.03, 0.08) = 1.13424$; slopes $2.001$ (one step) and $1.015$ (whole run). Measured over predicted $ln(c_infinity\/c_0)$:

#figure(
  table(
    columns: 7,
    [$eta$], [$0.1$], [$0.05$], [$0.02$], [$0.01$], [$0.005$], [$0.002$],
    [GD], [$1.064$], [$1.031$], [$1.012$], [$1.006$], [$1.003$], [$1.001$],
    [$beta = 0.3$], [$1.164$], [$1.070$], [$1.026$], [$1.012$], [$1.006$], [$1.002$],
    [$beta = 0.5$], [$1.603$], [$1.186$], [$1.059$], [$1.027$], [$1.013$], [$1.005$],
  ),
  caption: [The ratio tends to $1$ linearly in $eta$ (the neglected $O(eta^2)$ term), faster for GD than for momentum, whose effective step is larger.],
)
(E) dataset with $"Var"(x_i^2) = 3.24$: SGD ratio $4.28 plus.minus 0.06$, $1.83 plus.minus 0.01$, $1.198 plus.minus 0.005$ for $B = 1, 4, 16$ against $4.24, 1.81, 1.20$.

#figure(image("figures/drift.png", width: 100%), caption: [`drift.png`. Left: $|Delta ln c|$ against $eta$ for GD and momentum (one run each to convergence) and for a single GD step. Middle: $c(t)$ under weight decay. Right: SGD's drift relative to $-eta I$ against batch size.])

= Problem (#raw("framing")): reading notes

No single right answer; a good answer covers the following. (a) The second run of `check_saxe.py` dropped the task-aligned initialization (inputs stay whitened). The cost: one needs $u_0^"eff"$ instead of the init scale, and the half-times come $0$ to $3%$ late, more for weaker modes, from competition between modes during the plateau. Simon et al.'s Fig. 1(a) uses the aligned init, where the prediction is exact. (b) Predicted: GD's exact one-step factor and its $O(eta)$ total drift, momentum's factor $(1 + beta)\/(1 - beta)^2$, SGD's factor $1 + "Var"\/B$, weight decay's exact exponential. Not covered: steps so large that the run is far from the flow (A3's edge of stability, where $eta r$ is not small), label noise and other noise that does not vanish at the minimum (A4), adaptive optimizers such as Adam, whose per-coordinate rescaling makes the update no longer a multiple of a gradient orthogonal to $X$, and normalization layers that add other symmetries. (c) Scores depend on $x W_Q W_K^top x'^top$, invariant under $W_Q -> W_Q A$, $W_K -> W_K A^(-top)$ for invertible $A$. With $W_2 = W_Q$, $W_1 = W_K^top$ this is Problem (#raw("matrix-law")), and gradient flow conserves $W_K^top W_K - W_Q^top W_Q$ (Simon et al. §2.3; Kunin et al. treat such symmetries in §3).
