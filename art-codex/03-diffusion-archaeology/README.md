# Diffusion Archaeology

![A learned reverse-diffusion sample](outputs/diffusion-archaeology-final.png)

## What you are looking at

The colored points begin as pure Gaussian noise. A small neural network is trained to predict the noise added to examples from a two-moons distribution at every corruption level. During sampling, that prediction is used to take a sequence of reverse steps: noise becomes a suggestion, a suggestion becomes a bend, and the bent cloud resolves into two crescents.

The pale background points are held-out examples from the training distribution. They are not targets drawn into the result; they exist only as a faint reference for the structure the model has learned.

## The ML idea

Diffusion models learn a *denoising direction* rather than memorizing a final image. At a given noise level, the score network estimates how the current sample should move to become less noisy. Repeating those small estimates turns an unstructured field into a distribution with recognizable shape.

## Reading the color

Color encodes the current radial position of each generated point. It is intentionally independent of the target class: the emerging geometry, not a label, is the subject.

## Reproduce

```bash
/home/fzeng/ml/research/.venv/bin/python render.py
```

- [Looping animation](outputs/diffusion-archaeology.gif)
- [Final still](outputs/diffusion-archaeology-final.png)
- [Sample traces and loss](outputs/diffusion-data.npz)

## Production Edition

![The production master: learned structure surfacing from a reverse-diffusion cloud](outputs/master.png)

This edition treats the reverse process as an excavation rather than a collection of snapshots. A score network was trained to predict the noise corrupting points from a two-moons distribution across 72 noise levels. The model then guides 5,200 independent Gaussian samples backwards, one measured denoising step at a time.

- The colored strata are the log density of the current generated sample cloud.
- The fine luminous particles are current samples from that actual cloud.
- The threads are the last 20 reverse-diffusion states for fixed sample identities, so they describe movement rather than decoration.
- Thread color is each particle's final angular position, held constant across the loop to make continuity visible.

At the beginning there is only a diffuse numerical atmosphere. As the score estimates accumulate, the field first folds, then develops two durable arcs. The target distribution is never drawn into the production image; only model samples create its visible structure.

### Production Files

- [2560 x 1440 master still](outputs/master.png)
- [36-frame production loop](outputs/loop-production.gif)
- [Sample states and training loss](outputs/production-data.npz)
- [Exact visual mapping and run metadata](outputs/production-metadata.json)

Render this edition with:

```bash
/home/fzeng/ml/research/.venv/bin/python render_production.py
```
