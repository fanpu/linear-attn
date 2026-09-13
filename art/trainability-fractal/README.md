# Trainability: the fractal edge of gradient descent

*Every pixel is a separate neural network trained for 500 steps. Colour says whether it learned, and how quickly. The line between learning and blowing up is jagged at every scale we could afford to look at.*

<!-- HERO -->

## The phenomenon

Sohl-Dickstein (2024) pointed out that neural-network training has the same structure as a Mandelbrot iteration. You apply one map over and over, here a gradient-descent step, and ask whether the result stays bounded. Fix everything about a tiny network except two hyperparameters and colour a grid of them by the outcome. The boundary between *trains* and *diverges* is then intricate, and he measured it to be fractal over more than ten decades.

The setup reimplemented here is his (github.com/Sohl-Dickstein/fractal, read directly; declared differences are listed below):

$$\hat y(x) = \tfrac{1}{n}\, W_1\, \phi\!\Big(\tfrac{1}{\sqrt n} W_0 x\Big),\qquad \phi(z)=\tanh(\sqrt2\, z)\ \text{ or }\ \sqrt2\,\mathrm{relu}(z),\qquad n=16$$

$$\mathcal L = \tfrac1N\sum_i (\hat y(x_i)-y_i)^2,\qquad W_0 \leftarrow W_0-\eta_0\,\nabla_{W_0}\mathcal L,\quad W_1 \leftarrow W_1-\eta_1\,\nabla_{W_1}\mathcal L$$

There are no biases. $W_0\in\mathbb R^{16\times16}$ and $W_1\in\mathbb R^{1\times16}$ start from $\mathcal N(0,1)$. The dataset has $N = 272$ points, equal to the parameter count, with $x,y\sim\mathcal N(0,1)$. Init and data are shared by every pixel. The two image axes are $\log_{10}\eta_0$ (input layer) and $\log_{10}\eta_1$ (output layer).

**Convergence criterion and colour (his `convergence_measure`, replicated exactly).** Let $v_t = \min(\ell_t/\ell_0,\ 10^6)$, with non-finite losses set to $10^6$ before normalising. A run *converged* if $\operatorname{mean}(v_{T-20..T}) < 1$. The pixel value is $-\sum_t v_t$ if it converged and $+\sum_t 1/v_t$ if it diverged. Small magnitude means fast convergence or fast divergence, and large magnitude means the run sat near the edge for a long time.

**Colouring (his `cdf_img`, replicated exactly).** Each sign is rank-normalised separately: converged values map into $[-1,-0.25]$ and diverged values into $[0.25,1]$, the sign is kept, the result is negated, and it is shown through matplotlib `Spectral` with nearest-neighbour pixels. Converged runs go from pale yellow-green (fast) to deep purple (slow). Diverged runs go from pale orange (fast) to deep red (slow). The two dark ends meet at the boundary. This colour mapping is a declared aesthetic choice; the sign and the rank of the measure are the data.

<!-- BODY -->
