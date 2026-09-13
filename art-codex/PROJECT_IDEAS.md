# Art-Codex: Project Ideas

This is a studio for art whose form comes from real machine-learning behavior. The goal is not to decorate a technical plot; it is to make an actual computation legible, strange, and worth looking at.

## Studio Principles

- The visual should be generated from real states, gradients, activations, samples, or training traces.
- A viewer should be able to learn something true about the underlying system.
- Aesthetic choices may amplify the signal, but should not invent the phenomenon.
- Save the underlying data and parameters alongside each final work so every piece is reproducible.

## 1. The Descent Garden

**Phenomenon:** Gradient descent traversing a real loss landscape.

Train a small neural network on a two-moons or spiral classification task, then project a local slice of its loss surface into a topographic landscape. Animate the parameter point as it descends, while the decision boundary changes in a companion view. Different optimizers leave visibly distinct trails: SGD jitters, momentum overshoots, Adam cuts fast but can settle differently.

**Output:** Looping GIF/video, still prints of optimizer trajectories.

**First version:** A 2D parameter-plane slice around a trained MLP, rendered as terrain with an animated path.

## 2. Attention as Weather

**Phenomenon:** Transformer attention redistribution across generated text.

Use a compact open-weight language model and record attention maps as it generates a short passage. Interpret each head as a weather system: local attention becomes fog, long-distance recall becomes lightning-like filaments, and attention sinks become pressure centers. The mapping stays deterministic: position, strength, and temporal change derive directly from the attention tensor.

**Output:** Generative video with an accompanying interactive token/head explorer.

**First version:** One prompt, one layer, several heads, animated chord field or particle flow.

## 3. Diffusion Archaeology

**Phenomenon:** Image diffusion denoising from Gaussian noise to coherent structure.

Render the intermediate latent/image states of a diffusion model as an excavation: early noise contains weak directional hints; objects then emerge as unstable fossils before resolving. Instead of a standard contact sheet, calculate frame-to-frame change and expose where the model is still "thinking." 

**Output:** High-resolution animation and layered lenticular-style stills.

**First version:** A 48- to 80-step denoising sequence, with a difference-energy overlay.

## 4. The Grammar of a Neuron

**Phenomenon:** Feature visualization through activation maximization.

Optimize images that maximally activate selected neurons or channels in a vision network. Create a taxonomy that moves from primitive edges to textures, parts, and complex abstractions. The surprising material is the transition zone: units that have a preference but not a human name.

**Output:** Museum-like grid, animated morphs between neighboring units, print series.

**First version:** Feature visualization for a pretrained ResNet or ViT across early, middle, and late layers.

## 5. Decision Boundary Bloom

**Phenomenon:** Classifiers progressively carving a space into categories.

Train classifiers on synthetic data that begins simple and becomes entangled: circles, spirals, XOR, then noisy manifolds. At every checkpoint, render probability as translucent pigments. The boundary does not merely appear; it branches, hardens, tears, and heals during training.

**Output:** A meditative training GIF and a sequence of data-driven prints.

**First version:** Two-dimensional spiral data with a modest MLP, checkpointed every few optimizer steps.

## 6. Emergent Cartography

**Phenomenon:** Representation learning organizing samples in embedding space.

Embed a dataset through successive layers or training epochs, then create a map whose geography comes from neighborhood density, class mixture, and motion. Samples become migrating lights; clusters accrete into continents; ambiguous samples remain at coastlines. UMAP/t-SNE can be used only for display, while neighborhood relations are preserved and documented.

**Output:** Animated map, zoomable web canvas, large-format plot.

**First version:** MNIST or Fashion-MNIST embeddings across a CNN's layers.

## 7. The Memorization Tide

**Phenomenon:** A neural network fitting signal before noise.

Train on an image or label dataset with a controlled fraction of corrupted labels. Track which examples become correctly predicted at which epoch, then turn the ordering into a wave traveling across an image mosaic. Clear structure is learned first; exceptions surface later. This offers a beautiful, truthful view of the memorization effect.

**Output:** Animated mosaic and time-indexed print set.

**First version:** CIFAR-10 subset with 10-30% label noise, highlighting example learning times.

## 8. Neural Cellular Dreams

**Phenomenon:** Neural cellular automata learning to grow and repair a target form.

Train a neural cellular automaton using the real update rule from differentiable programming. Show its development from a seed, then injure or erase a region and let the organism repair itself. The work is both visual and computational: local rules generate global persistence.

**Output:** Rich looping videos, interactive damage-and-recovery studies.

**First version:** Grow a simple textured disk or flower-like target from a single seed.

## 9. A Portrait of Uncertainty

**Phenomenon:** Predictive uncertainty under distribution shift.

Pass an image through a classifier while gradually transforming it away from the training distribution: blur, rotate, tint, occlude, or interpolate with another class. Visualize the changing predictive distribution as a portrait that fractures according to entropy, calibration, and disagreement across an ensemble.

**Output:** Short video and a set of uncertainty portraits.

**First version:** Pretrained image classifier, one source image, controlled occlusion sweep.

## 10. The Adversarial Microscope

**Phenomenon:** Tiny perturbations radically changing model predictions.

Create an apparently stable image that reveals a delicate adversarial pattern only through magnification or alternating frames. A second layer shows the model's confidence changing as perturbations accumulate. The tension comes from the mismatch between what is visible to us and what is decisive to the network.

**Output:** Two-channel video, zoomable still, optical-illusion print.

**First version:** FGSM or PGD attack on a pretrained ImageNet classifier.

## 11. The Pruning Chorus

**Phenomenon:** Network sparsification and the lottery-ticket intuition.

Iteratively prune a trained network and record what remains: weights, activations, accuracy, and topology. Render nodes and connections as a changing score. The composition thins dramatically while behavior may remain almost intact, then reaches a sudden failure point.

**Output:** Network animation with a sonified companion track.

**First version:** Magnitude-prune an MLP trained on Fashion-MNIST and animate its connectivity.

## 12. Latent Interpolation, Honestly

**Phenomenon:** Structure of a learned generative latent space.

Choose two actual latent codes in a trained VAE or diffusion model and interpolate between them. Pair the familiar morph with quantitative traces: reconstruction error, local curvature, and semantic classifier response. The objective is to reveal that a smooth image path can conceal a highly nonuniform internal geometry.

**Output:** Diptych video: generated imagery beside latent-space diagnostics.

**First version:** VAE trained on a small visual dataset, with linear and spherical interpolation compared.

## Suggested First Three

1. **Decision Boundary Bloom**: fast, self-contained, visually immediate, and ideal for establishing a rendering pipeline.
2. **The Descent Garden**: makes optimization physically intuitive and supports many aesthetic variants without losing scientific grounding.
3. **Neural Cellular Dreams**: more computationally involved, but unusually alive and naturally suited to video.

## Common Artifact Structure

Each project should eventually contain:

```text
project-name/
  README.md          # question, method, dependencies, reproduction command
  train.py           # model or experiment
  render.py          # deterministic visual renderer
  configs/           # frozen experiment settings
  data/              # cached derived data; source datasets stay out of git
  outputs/           # renders, checkpoints, and metadata
```

For GPU work, use PyTorch with CUDA, record GPU/model/library versions in render metadata, and make a small CPU-friendly preview mode for iteration.
