# The optimizer chooses

<p class="subtitle">When infinitely many solutions fit the data perfectly, the loss can't tell you which one training finds. The optimizer, the parameterization and the initialization decide. A tour with measurements, from logistic regression to Adam and Muon.</p>

<style>
.ib-controls { display: flex; flex-wrap: wrap; gap: .45rem 1.1rem; align-items: center; margin-bottom: .55rem; font-size: .86rem; }
.ib-row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; justify-content: center; }
.ib-row > div { flex: 1 1 380px; }
.ib-val { font-variant-numeric: tabular-nums; min-width: 3.2em; display: inline-block; color: #1d1d1f; }
.ib-read { color: #52514e; font-variant-numeric: tabular-nums; font-size: .84rem; }
.ib-sub { color: #8b8984; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.ib-test { display: none; font-size: .75rem; background: #f3f0ea; padding: .5rem; }
.widget button { font: inherit; font-size: .84rem; padding: .28rem .8rem; border: 1px solid #d8d3ca; border-radius: 6px; background: #fcfbf8; cursor: pointer; }
.widget button:hover { background: #f3f0ea; }
.widget input[type=range] { width: 130px; accent-color: #2a78d6; }
.widget select { font: inherit; font-size: .84rem; }
.widget .wcap { color: #6b6b70; font-size: .84rem; line-height: 1.45; margin: .6rem 0 0; }
figure.hero { background: #0b0d12; border-radius: 10px; padding: 0; overflow: hidden; }
figure.hero figcaption { color: #8e929c; padding: .2rem 1.1rem .9rem; }
.tag { display: inline-block; font: 600 .68rem/1 "Inter", "Helvetica Neue", Arial, sans-serif; letter-spacing: .06em; text-transform: uppercase; padding: .28rem .5rem; border-radius: 4px; vertical-align: middle; margin-right: .4rem; }
.tag.new { background: #fde8dd; color: #9c3a12; }
.tag.lit { background: #e3edf9; color: #1c5cab; }
.keyeq { background: #fff; border: 1px solid #e6e2da; border-radius: 8px; padding: .3rem 1rem; margin: 1.2rem 0; }
</style>

<figure class="hero wide">
<video autoplay loop muted playsinline poster="figures/hero_still.png" src="figures/hero.mp4"></video>
<figcaption>Logistic regression trained by plain gradient descent on data a line can separate. The white line is the decision boundary; the dashed lilac line is the maximum-margin (hard SVM) boundary. Watch the clock on the right: it counts to $10^{100}$ steps, and the boundary is <em>still</em> not there. The dashed curve in the angle panel is the closed-form asymptote from Soudry et al. (2018); the measured curve lies on it to seven digits.</figcaption>
</figure>

Here is a small puzzle. Take a dataset that a straight line can separate, and train logistic regression on it with gradient descent. The loss goes to zero. But logistic loss never actually *reaches* zero: you can always make it smaller by scaling up the weights. So there is no minimum. Instead there is a whole cone of directions, every one of which drives the loss to zero as $\|w\| \to \infty$.

So which direction does gradient descent pick? The animation above gives the answer: the one that makes the **margin** as large as possible, the same boundary a support vector machine would draw. Nobody asked for that. There is no margin term anywhere in the loss. The preference comes from the optimizer.

It also shows something stranger. The boundary gets most of the way there within a few hundred steps. After that it creeps, and the creep never ends: after $10^{100}$ steps it is still a quarter of a degree off.

This post is about that phenomenon, called **implicit bias**. Modern networks have far more parameters than training examples, so there are usually infinitely many ways to fit the training set perfectly. The loss can't choose between them. The choice is made by the optimizer, the parameterization and the initialization, and which solution you get decides how the model generalizes. We'll see four classic examples, each reproduced on this machine with the theory drawn on top of the measurements. Then we'll ask the question for the optimizers people actually use today: Adam and Muon.

## 1. Many perfect fits, one answer

Two kinds of problems make "infinitely many zero-loss solutions" concrete:

- **Underdetermined regression.** With $d$ parameters and $n < d$ equations $Xw = y$, the perfect fits form an affine subspace of dimension $d - n$.
- **Separable classification.** With logistic or cross-entropy loss, any separating direction $w$ gives loss $\to 0$ as you scale it up.

In both cases the loss is flat along a huge set of solutions. Whatever training returns must have been selected by *how* we trained. That selection is usually easiest to describe as a hidden optimization problem. Plain gradient descent on these models secretly solves

$$\min_w \ \|w\|_{\text{something}} \quad \text{subject to fitting the data},$$

and the "something" depends on the algorithm and the parameterization. The rest of this post is a catalogue of what fills that slot, and of how long it takes to get there.

## 2. Logistic regression finds the max-margin SVM, eventually <span class="tag lit">literature</span>

Let the data be $(x_n, y_n)$ with labels $y_n \in \{-1, +1\}$, and train a linear classifier (no bias) on the summed logistic loss

$$L(w) = \sum_{n=1}^{N} \log\big(1 + e^{-y_n x_n^\top w}\big).$$

**Why the margin shows up.** Once the data are classified correctly, every term is roughly $e^{-y_n x_n^\top w}$. The negative gradient is then

$$-\nabla L(w) \approx \sum_n e^{-y_n x_n^\top w}\, y_n x_n .$$

Suppose $w$ grows like $\hat w \log t$ for some direction $\hat w$, scaled so the closest points have $y_n x_n^\top \hat w = 1$. A point at margin $m_n$ then contributes $e^{-m_n \log t} = t^{-m_n}$. The closest points contribute $1/t$; everything farther away contributes $t^{-m_n}$ with $m_n > 1$, which is negligible by comparison. So in the long run, the gradient is a *non-negative combination of the closest points only*. Integrating $1/t$ gives $\log t$, and we get

$$\hat w = \sum_{n \in \text{closest}} \alpha_n\, y_n x_n, \qquad \alpha_n \ge 0, \qquad y_n x_n^\top \hat w \ge 1 .$$

These are exactly the optimality (KKT) conditions of the hard-margin SVM, $\min \|w\|_2^2$ subject to $y_n x_n^\top w \ge 1$. The $\alpha_n$ are its dual variables and the closest points are its support vectors.

<div class="keyeq">

**Theorem (Soudry, Hoffer, Nacson, Gunasekar & Srebro, 2018).** For separable data, a loss with an exponential tail and step size $\eta < 2\beta^{-1}\sigma_{\max}^{-2}(X)$, gradient descent satisfies

$$w(t) = \hat w \log t + \rho(t),$$

where $\hat w$ is the $L_2$ max-margin solution. For almost every dataset the residual $\rho(t)$ stays bounded. If the support vectors span the data, $\rho(t) \to \tilde w$, defined by $\eta\, e^{-y_n x_n^\top \tilde w} = \alpha_n$ for every support vector. The direction converges only like $\big\|\tfrac{w(t)}{\|w(t)\|} - \tfrac{\hat w}{\|\hat w\|}\big\| = O(1/\log t)$.

</div>

The $1/\log t$ is the creep in the hero animation. The correction $\rho(t)$ stays bounded while $\|w\|$ grows like $\log t$, so the angle shrinks like $\|\rho\|/\log t$. Going from $10^{10}$ steps to $10^{100}$ steps only divides $\log t$ by ten.

To check this literally, I integrated the gradient flow in *log-time*, $s = \log t$. The substitution makes the ODE well-behaved out to $t = 10^{100}$ in float64. On the 2D hero dataset the two support vectors span the plane, so $\tilde w$ is fully predicted by the SVM duals. Panel (a) below shows the residual $\rho(t)$ landing on that prediction. Panel (b) repeats the experiment on 8 random datasets in $d = 50$ with $n = 40$, where the theorem only promises the rate. Multiplying the angle by $\ln t$ gives a curve that flattens, which is what a $1/\log t$ law looks like.

<figure class="full">
<img src="figures/soudry_crawl.png" alt="Three panels: residual converging to closed-form w tilde; angle times log t flattening; margin-gap rates for GD and normalized GD">
<figcaption><b>(a)</b> 2D hero data. The residual $w(t) - \hat w \ln t$ converges to the $\tilde w$ computed from the SVM dual variables, shown dashed. Dots are discrete GD with $\eta = 0.1$, plotted at flow time $\eta k$. <b>(b)</b> $d = 50$, $n = 40$, random labels, 8 datasets. Solid lines: GD for $10^8$ steps in float64. Dotted lines: the same gradient flow continued in log-time to $10^{100}$. <b>(c)</b> The gap between the current normalized margin and the best achievable margin. Plain GD follows the $1/\ln t$ guide. Normalizing the step (constant step, or $\eta_t \propto 1/\sqrt t$) makes the gap fall polynomially, consistent with the rates of Nacson et al. (2019) and Ji & Telgarsky (2021). Thin lines are individual datasets; thick lines are medians.</figcaption>
</figure>

{{SOUDRY_NUMBERS}}

The fix for the slowness is simple and well known. The creep happens because the gradient shrinks like $1/t$ as the loss vanishes. Normalizing the step, $w \leftarrow w - \eta\, \nabla L / \|\nabla L\|$, keeps the pace constant. Panel (c) shows the payoff: the margin gap falls polynomially instead of logarithmically. The limit doesn't change, only the speed.

## 3. Change the geometry, change the answer <span class="tag lit">literature</span>

Normalized GD has a useful interpretation. It is **steepest descent with respect to the $L_2$ norm**: among all steps of unit $L_2$ length, it takes the one that decreases the loss fastest. Other norms define other "steepest" directions:

- For $L_\infty$, the steepest unit step is $-\operatorname{sign}(\nabla L)$: **sign gradient descent**, the core of Adam without its running averages.
- For $L_1$, the steepest unit step moves only the single coordinate with the largest gradient: **greedy coordinate descent**.

Gunasekar, Lee, Soudry & Srebro (2018) showed that on separable data with exponential loss, steepest descent with respect to a norm $\|\cdot\|$ maximizes the margin measured in **that same norm**:

$$\lim_{t\to\infty}\ \min_n \frac{y_n x_n^\top w(t)}{\|w(t)\|} \;=\; \max_{\|w\| \le 1}\ \min_n\, y_n x_n^\top w .$$

The direction itself converges when that max-margin solution is unique. Their theorem covers unnormalized steepest descent with a small step-size cap; normalized variants with decaying steps were analysed later by Fan, Schmidt & Thrampoulidis (2025). Below, you can race five optimizers live. The data are built so that the $L_2$, $L_\infty$ and $L_1$ max-margin directions are all different (63°, 45° and 90°).

<div class="widget wide" id="geometry-widget"></div>
<script src="widgets/common.js"></script>
<script src="widgets/data_geometry.js"></script>
<script src="widgets/geometry.js"></script>
<p class="wcap" style="max-width:1000px;margin:-1.2rem auto 2rem">
<b>Widget: optimizer geometry.</b> Five optimizers train the same logistic regression in your browser, in float64. Press run; time is on a log scale. <b>Weight space</b> (right) makes the theorem visible. The shaded region is every $w$ with margin at least 1. Grow a circle, a square and a diamond from the origin until each first touches that region. The touching points are the $L_2$, $L_\infty$ and $L_1$ max-margin solutions, and each optimizer's direction heads toward its own. Sign GD locks onto the corner of the square within a few hundred steps. Coordinate descent heads to the diamond's tip. Plain GD crawls toward the circle. Adam starts at the square's corner, then leaves it; set ε = 0 and compare (section 6).</p>

The same experiment at scale ($d = 50$, eight datasets) shows that each method saturates at the best margin *in its own norm*, and at a strictly worse value in the other norm. The dotted lines mark where theory says the "wrong" optimizer should saturate: the margin of the other norm's max-margin solution.

<figure class="wide">
<img src="figures/geometry_margins.png" alt="Normalized margins in L2 and Linf over time for GD, normalized GD and sign GD">
<figcaption>Left: $L_2$-normalized margin as a fraction of the best possible. Right: the same in $L_\infty$. GD and normalized GD approach 1 on the left (GD slowly) and saturate at the dotted level on the right. Sign GD does the opposite. $d = 50$, $n = 40$, 8 datasets; thick lines are medians.</figcaption>
</figure>

## 4. Change the parameterization: diagonal linear networks <span class="tag lit">literature</span>

So far the model was linear in its parameters. Now keep the *function* linear, $f(x) = \langle w, x\rangle$, but change how $w$ is written down. Woodworth et al. (2020) study the simplest "deep" parameterization there is:

$$w = u \odot u - v \odot v, \qquad u(0) = v(0) = \alpha\,\mathbf 1 .$$

This is a two-layer linear network with diagonal weights. The difference of squares lets $w$ take either sign, and $w(0) = 0$ no matter how large the **initialization scale** $\alpha$ is. We train it with gradient flow on the squared loss for an underdetermined regression $Xw = y$, and ask which of the infinitely many interpolants it lands on.

Start with the smallest possible case: $d = 2$ weights and a single data point $x = (1, 0.4)$, $y = 1$. The perfect fits form a line. Two famous points sit on it. The min-$L_2$ solution is where the smallest circle touches the line. The min-$L_1$ solution is where the smallest diamond touches it, and it is sparse: $w = (1, 0)$.

<figure class="wide">
<video autoplay loop muted playsinline poster="figures/diag_2d_still.png" src="figures/diag_2d.mp4"></video>
<figcaption>Gradient-flow trajectories of the diagonal network from $w = 0$, for initialization scales from $\alpha = 10$ (light blue) down to $\alpha = 0.001$ (dark green). Large $\alpha$ walks straight to the min-$L_2$ point. Small $\alpha$ first grows the coordinate with the larger input, $w_1$, and lands on the sparse min-$L_1$ point. The dashed curve is the level set of Woodworth et al.'s potential $Q_\alpha$ that touches the line. It morphs from circle to diamond as $\alpha$ shrinks, and it always touches the line exactly where the trajectory ends. Paths are the closed form $w(c) = 2\alpha^2 \sinh(c\,x)$; LSODA integration of the actual flow agrees with the endpoints to all printed digits.</figcaption>
</figure>

**Where the potential comes from.** Along the flow, $\dot u = -2\,u \odot g$ and $\dot v = +2\,v \odot g$, where $g = \nabla_w L$. Both solve exactly: $u = \alpha\, e^{-2\int g}$ and $v = \alpha\, e^{+2 \int g}$. Since $g = X^\top r$ always lies in the row space of $X$, the flow can only reach points of the form

$$w = u^2 - v^2 = 2\alpha^2 \sinh\!\big(X^\top \nu\big) \quad\text{for some } \nu \in \mathbb{R}^n .$$

That is precisely the stationarity condition of the convex problem

<div class="keyeq">

$$\min_w\ Q_\alpha(w) = \alpha^2 \sum_i q\!\left(\frac{w_i}{\alpha^2}\right) \ \ \text{s.t.}\ Xw = y, \qquad q(z) = 2 - \sqrt{4 + z^2} + z\,\operatorname{arcsinh}(z/2).$$

</div>

This is the **Woodworth et al. (2020)** result. For small $z$, $q(z) \approx z^2/4$: a large $\alpha$ makes every $w_i/\alpha^2$ small, so $Q_\alpha$ is a scaled $L_2$ norm (the "kernel regime"). For large $z$, $q(z) \approx |z|\log|z|$, and $Q_\alpha / \log(1/\alpha^2) \to \|w\|_1$ as $\alpha \to 0$ (the "rich regime"). One knob moves the implicit bias continuously from $L_2$ to $L_1$. The same argument works for depth $D$, $w = u^{D} - v^{D}$, with a different potential that becomes $L_1$-like faster. Try it:

<div class="widget wide" id="diagnet-widget"></div>
<script src="widgets/diagnet.js"></script>
<p class="wcap" style="max-width:1000px;margin:-1.2rem auto 2rem">
<b>Widget: the diagonal network, live.</b> Drag α to move between the kernel and rich regimes. Change the depth D, or move the data point. Everything is computed in your browser: the closed-form flow path, where it lands, and the level set of $Q_\alpha$ (for D ≥ 3, the depth-D potential of Woodworth et al., integrated numerically). <b>Run real gradient descent</b> trains $(u, v)$ step by step (orange dots) so you can check that the dynamics really follow the closed form. The right panel shows that deeper networks reach the sparse answer at much larger α.</p>

The picture is the same in 100 dimensions. Take $d = 100$ weights, $n = 40$ Gaussian measurements and a 5-sparse ground truth, so there is a 60-dimensional space of perfect fits. Sweeping $\alpha$ over seven decades morphs the solution from a dense vector into the sparse truth:

<figure class="wide">
<video autoplay loop muted playsinline poster="figures/diag_stems_still.png" src="figures/diag_stems.mp4"></video>
<figcaption>The interpolant found by the diagonal network, $w_\infty(\alpha)$, as $\alpha$ sweeps from $10^2$ to $10^{-5}$ and back. Rings mark the true 5-sparse $w^\star$. At large $\alpha$ the solution is the dense min-$L_2$ interpolant, with 2.7 units of error. At small $\alpha$ it is basis pursuit, which here recovers $w^\star$ exactly. Frames use the closed-form $Q_\alpha$ minimizer, which matches gradient flow to $10^{-8}$ (next figure).</figcaption>
</figure>

<figure class="wide">
<img src="figures/diag_sweep.png" alt="Distance to basis pursuit and min-L2 vs alpha; recovery error vs n">
<figcaption>Left: dots are the measured end points of gradient flow on $(u, v)$ (LSODA, rtol $10^{-11}$) for 57 values of $\alpha$. Lines are the closed-form $\arg\min Q_\alpha$, solved independently through its convex dual. They agree to $10^{-8}$ everywhere. The distance to basis pursuit shrinks linearly in $\alpha$ (slope 1 on log-log axes); the distance to min-$L_2$ vanishes like $\alpha^{-2}$. Right: relative recovery error against the number of measurements. Ten random problems per point, $d = 100$, 5-sparse truth.</figcaption>
</figure>

A practical note about "small". The distance to basis pursuit shrinks only linearly in $\alpha$, and Woodworth et al. show that reaching the $L_1$ limit in general needs *exponentially* small initialization. Tiny initialization also costs training time: the flow sits near the saddle at $w = 0$ for time $\sim \log(1/\alpha)$ before the coordinates escape, one at a time.

## 5. Depth and low rank: a bias no norm explains <span class="tag lit">literature</span>

The matrix version of the diagonal network is **deep matrix factorization**. We fit an unknown $100 \times 100$ matrix of rank 5, observing 20% of its entries, by writing $W = W_N \cdots W_2 W_1$ and running gradient descent from a tiny random init on the squared error over the observed entries. With $N = 1$ the unobserved entries never receive a gradient, so nothing is learned about them. With $N \ge 2$ something remarkable happens.

<figure class="wide">
<video autoplay loop muted playsinline poster="figures/mc_spectra_still.png" src="figures/mc_spectra.mp4"></video>
<figcaption>Top: the 12 largest singular values of $W$ during training, with ghost ticks at the true singular values. Bottom: $\sigma_1, \ldots, \sigma_8$ over time on a log axis. Depth 1 inflates everything at once and generalizes badly. Depth 2 learns modes in stages. Depth 3 separates the stages sharply, peeling off one singular value at a time, and never grows the spurious ones. Time is gradient-flow time, $\eta \times$ steps. Depth 3 uses a smaller step, $\eta = 4\cdot 10^{-4}$, so that one-step differences track the flow.</figcaption>
</figure>

Arora, Cohen, Hu & Luo (2019) explain the staging with an exact equation. For gradient flow from a balanced initialization, each singular value of the product evolves as

$$\dot\sigma_r = -N\, \big(\sigma_r^2\big)^{1 - 1/N}\ \big\langle \nabla \ell(W),\ u_r v_r^\top \big\rangle .$$

The factor $\sigma_r^{2 - 2/N}$ is a rich-get-richer term. For $N = 1$ it is absent. For $N \ge 2$, singular values that are already large grow faster, and small ones stay frozen near zero until the large ones have fit what they can. Deeper networks make the exponent larger, and the preference for low rank stronger.

<figure class="wide">
<img src="figures/mc_summary.png" alt="Recovery error by depth and nuclear norm; singular value dynamics check">
<figcaption>Left: relative recovery error after training. The convex baseline is the matrix with minimum nuclear norm that agrees with the observations, solved with SCS. {{MC_CAPTION}} Right: the Arora et al. equation checked step by step, for the top 5 singular values at every recorded step. The measured one-step change in $\sigma_r$ is plotted against $\eta$ times the right-hand side.</figcaption>
</figure>

{{MC_NUMBERS}}

A natural guess is that depth secretly minimizes the nuclear norm, a convex proxy for rank. It doesn't. At 20% observed, the depth-3 solution has *larger* nuclear norm than the nuclear-norm minimizer, yet lower error. Razin & Cohen (2020) went further and built a case where the implicit bias can't be *any* norm.

Complete a $2\times 2$ matrix with three observed entries, $W_{12} = W_{21} = 1$ and $W_{22} = 0$. Every completion $\begin{psmallmatrix} w & 1 \\ 1 & 0\end{psmallmatrix}$ has determinant $-1$. A depth-$N \ge 2$ factorization started from balanced factors with $\det W(0) > 0$ can never change the sign of its determinant under gradient flow, so it can never fit the data exactly. As the loss goes to zero it must send the free entry $|w| \to \infty$. Every norm of $W$ then diverges, while $W$ gets ever closer to rank 1: its second singular value is at most $3\sqrt2\sqrt{\ell}$. Norm minimization would pick a finite completion; depth instead lowers the rank and pays with an unbounded norm.

<figure class="full">
<img src="figures/razin.png" alt="Razin-Cohen example: W11 diverges, norms diverge, sigma2 goes to zero">
<figcaption>Gradient flow on the Razin–Cohen $2\times2$ problem from $W(0) = 0.1\,I$, as balanced factors. Left: with $\det W(0) > 0$ the unobserved entry grows without bound. With $\det W(0) < 0$ (one factor's sign flipped) the flow reaches zero loss at a finite, small completion. Middle: nuclear and Frobenius norms grow like $1/\sqrt{\ell}$. Right: $\sigma_2$, the distance to the nearest rank-1 matrix, stays below the Razin–Cohen bound. The × marks where float64 round-off tunnels through $\det W = 0$, something the exact flow can't do. The curves stop there.</figcaption>
</figure>

## 6. Modern optimizers <span class="tag new">new measurements</span>

{{BUILDON}}

## 7. Where it breaks

{{BREAKS}}

## Reproduce it

{{REPRO}}

## References

{{REFS}}
