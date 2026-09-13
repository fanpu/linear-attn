# One Basin

*Two networks trained from different seeds look like two separate valleys. List one network's hidden units in a different order and the mountain between them disappears.*

<img src="gallery/triptych_mnist_spectral.png" width="100%">

*Hero: loss along three paths between the same two MNIST networks, drawn as geological sections. The surface is the measured train loss, and each stratum is one digit class's share of it. The lower strip repeats each section at ×100 vertical exaggeration.*

## 1. The phenomenon

Train two copies of a network, A and B, from different random seeds. Then walk the straight line between them in weight space,

$$\theta(\lambda) = (1-\lambda)\,\theta_A + \lambda\,\theta_B,\qquad \lambda\in[0,1].$$

The loss rises steeply in the middle. The **barrier** is

$$\mathcal{B} = \max_\lambda\; L(\theta(\lambda)) - \big[(1-\lambda)L(\theta_A) + \lambda L(\theta_B)\big].$$

That rise makes it look as though A and B sit in isolated minima. Two findings argue otherwise:

1. **Curved paths.** Garipov et al. and Draxler et al. (2018) showed that low-loss curves connect such minima. We train a quadratic Bézier curve $\theta(t) = (1-t)^2\theta_A + 2t(1-t)\theta_C + t^2\theta_B$, keeping the endpoints fixed and learning the control point C.
2. **Permutations.** Hidden units are interchangeable. Permute the rows of $W_\ell$ and $b_\ell$ and the columns of $W_{\ell+1}$ by the same permutation $\pi_\ell$, and the network computes exactly the same function. Entezari et al. (2021) conjectured that most of the barrier comes from this symmetry. Git Re-Basin (Ainsworth et al., 2022) gave algorithms to remove it. **Weight matching** maximises $\sum_\ell \langle W_\ell^A,\, P_\ell W_\ell^B P_{\ell-1}^\top\rangle$ by coordinate descent, solving one linear assignment problem per layer per sweep. After matching, the straight line from A to $\pi(B)$ is nearly flat.

Two caveats matter and are measured below. Linear mode connectivity after matching **depends on width**: it fails for narrow networks. It is also **emergent during training**: it does not hold at initialisation in any interesting sense. REPAIR (Jordan et al., 2022) resets the per-unit activation statistics of the interpolated network, which removes a further part of the barrier caused by variance collapse.

## 2. Gallery

GALLERY_TBD

## 3. What was computed

COMPUTED_TBD

## 4. Verification and honesty

VERIFY_TBD

## 5. Caveats

- **A plane is a slice.** The 2-D planes show exactly the loss on one affine plane through three trained points. Non-convexity in a slice implies non-convexity overall, but flatness or connectedness in a slice says little about the rest of the space. "One basin" here means that the segment A–π(B) and the sublevel set in *this plane* are connected. It does not mean the full loss surface has a single basin.
- **Euclidean distances in weight space are a choice.** The same function can sit at very different Euclidean distances (for example, permutations, or ReLU rescaling in the style of Dinh et al.). Weight matching itself relies on this: A and π(B) are closer in L2 than A and B, while B and π(B) are the *same function*.
- **"Barrier-free" is measured on a small MLP.** These are 3-hidden-layer MLPs on MNIST and Fashion-MNIST. Git Re-Basin reports that ResNets and VGGs on CIFAR need much more width, and that the barrier on ImageNet does not vanish. Nothing here tests those settings. See the negative results.
- **The Bézier curve's barrier is 0 by the linear-baseline definition** because the curve runs *below* the endpoints' loss for most of its length. The ×100 strip shows that shape. Its midpoint-definition barrier is 0.0012 nats.
- **Weight matching is a heuristic.** It finds a local optimum of a bilinear objective. It is not the permutation that minimises the barrier.
- **Colour.** Every Spectral or split image is rank-normalised per side. Colours encode order, not the size of loss differences, and are not comparable between images.

## 6. References

- Garipov, Izmailov, Podoprikhin, Vetrov & Wilson, *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*, NeurIPS 2018. arXiv:1802.10026
- Draxler, Veschgini, Salmhofer & Hamprecht, *Essentially No Barriers in Neural Network Energy Landscape*, ICML 2018. arXiv:1803.00885
- Frankle, Dziugaite, Roy & Carbin, *Linear Mode Connectivity and the Lottery Ticket Hypothesis*, ICML 2020. arXiv:1912.05671
- Entezari, Sedghi, Saukh & Neyshabur, *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks*, ICLR 2022. arXiv:2110.06296
- Ainsworth, Hayase & Srinivasa, *Git Re-Basin: Merging Models modulo Permutation Symmetries*, ICLR 2023. arXiv:2209.04836
- Jordan, Sedghi, Saukh, Entezari & Neyshabur, *REPAIR: REnormalizing Permuted Activations for Interpolation Repair*, ICLR 2023. arXiv:2211.08403
- Dinh, Pascanu, Bengio & Bengio, *Sharp Minima Can Generalize for Deep Nets*, ICML 2017. arXiv:1703.04933
- Sohl-Dickstein, *The boundary of neural network trainability is fractal*, 2024 (Spectral split colour convention). github.com/Sohl-Dickstein/fractal
