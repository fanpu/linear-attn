# Weight Spectrum: a matrix learning to be less random

*Every eigenvalue of a weight matrix, printed while it trains: it starts as textbook random-matrix noise and grows a tail.*

__HERO__

## The phenomenon

Take a dense layer's weight matrix $W \in \mathbb{R}^{N\times M}$ (oriented so $N \ge M$, aspect ratio $Q = N/M$) and form the
correlation matrix $X = W^\top W / N$. Its eigenvalues $\lambda_i = s_i^2/N$ ($s_i$ are the singular values of $W$) make up the
**empirical spectral density** (ESD).

At initialization the entries are i.i.d. with variance $\sigma^2$. Random matrix theory then predicts the ESD exactly:
the **Marchenko–Pastur (MP) law**

$$\rho_{MP}(\lambda) = \frac{Q}{2\pi\sigma^2\lambda}\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)},\qquad \lambda_\pm = \sigma^2\left(1\pm \tfrac{1}{\sqrt Q}\right)^2 .$$

Nothing goes past $\lambda_+$ except finite-size fluctuations of order $M^{-2/3}$.

Martin & Mahoney's *heavy-tailed self-regularization* (HT-SR) program tracks how SGD pulls the ESD away from this law. First a few
eigenvalues bleed out past $\lambda_+$ ("bleeding out"), then they separate into spikes ("bulk + spikes"), and in well-trained
networks the tail looks like a power law $\rho(\lambda) \sim \lambda^{-\alpha}$ ("heavy-tailed"). They report that smaller batch
sizes push a network further along this sequence, and that smaller $\alpha$ tends to go with better test accuracy.

__PHENOMENON_NUMBERS__

**Two controls are used throughout.**
1. **MP at the measured $\sigma^2$.** The theoretical curve uses the variance of the layer's current entries. Nothing is fitted.
2. **Shuffled-entries null.** At every checkpoint we also compute the ESD of the *same* matrix with its entries randomly permuted.
   This keeps the marginal distribution of the entries (including any heavy-tailed entries) and destroys all row/column
   correlations. Whatever survives the shuffle is not learned structure. "Eigenvalues above the null" counts
   $\lambda_i > \max \lambda^{\text{shuffled}}$.

__GALLERY__
