# Art-Codex

An evolving collection of images and moving images made from real machine-learning phenomena. Each work contains a readable account of the experiment, the renderer, the visual output, and the numeric traces that produced it.

## The Collection

| # | Work | Phenomenon | Status |
|---|---|---|---|
| 01 | [The Descent Garden](01-descent-garden/README.md) | Gradient descent and a measured loss slice | Rendered |
| 02 | [Attention as Weather](02-attention-as-weather/README.md) | Causal attention becoming structured through training | Rendered |
| 03 | Diffusion Archaeology | Denoising trajectories | In preparation |
| 04 | The Grammar of a Neuron | Activation maximization | In preparation |
| 05 | Decision Boundary Bloom | Classifier boundary formation | In preparation |
| 06 | Emergent Cartography | Representation organization | In preparation |
| 07 | The Memorization Tide | Signal-before-noise learning | In preparation |
| 08 | Neural Cellular Dreams | Learned growth and repair | In preparation |
| 09 | A Portrait of Uncertainty | Distribution shift and predictive entropy | In preparation |
| 10 | The Adversarial Microscope | Adversarial perturbation | In preparation |
| 11 | The Pruning Chorus | Sparsification and performance | In preparation |
| 12 | Latent Interpolation, Honestly | Learned generative geometry | In preparation |

## Standard of Evidence

Every color, position, and animation state should be derived from a model state, a dataset, or a measured quantity. Source scripts preserve their random seed and write compact data artifacts beside the renders. The objective is a collection that can be experienced as art and inspected as experimental work.

## Rendering Environment

The series uses the project virtual environment at `/home/fzeng/ml/research/.venv`, with PyTorch 2.14.0+cu130 running on an NVIDIA GB10. Renderers are intentionally self-contained and write their deliverables under each project’s `outputs/` directory.
