# The Descent Garden

![A classifier decision field beside a measured loss landscape](outputs/descent-garden-final.png)

## What you are looking at

This work follows a neural network while it learns to separate two intertwined spirals. Every frame is an actual state of the model during training, computed on the GB10 GPU.

The left panel is the classifier's **decision field**. Teal and coral regions indicate which spiral the network predicts; the pale contour is the 50% decision boundary. The small points are the real training data. At first the boundary is almost arbitrary. As training proceeds, it bends and folds into the space between the spirals.

The right panel is not a metaphorical landscape. It is a **measured slice of the loss function**: after training, we choose two normalized directions through the model's weight space, perturb the final weights across a grid, and calculate the cross-entropy loss at every location. The mint line is the path taken by the network, projected into that same plane.

## Why a garden?

Loss is a number, but its geometry governs learning. Low regions act like basins; steep regions make the optimizer move quickly; ridges deflect it. Momentum gives this particular path a soft inertia, visible as a curved approach rather than a direct line. The work pairs the hidden geometry of parameters with the visible geometry of a learned boundary.

## Reading the color

The left panel uses a cool-to-warm probability field, so uncertainty occupies the pale middle range. The terrain uses a separate ember palette: bright areas are low-loss basins and dark areas are high-loss terrain. The two palettes deliberately distinguish prediction from optimization.

## Reproduce

```bash
/home/fzeng/ml/research/.venv/bin/python render.py
```

The renderer writes a looping GIF, a final PNG, compressed numeric traces, and metadata to `outputs/`. Change `SEED`, `STEPS`, or the optimizer in `render.py` to grow a different garden.

## Files

- [Looping animation](outputs/descent-garden.gif)
- [Final still](outputs/descent-garden-final.png)
- [Exact numerical data](outputs/experiment-data.npz)
- [Render metadata](outputs/metadata.json)

## Production Edition

![The production master: a classifier field rendered as a living terrain](outputs/master.png)

This edition replaces the split diagnostic layout with one continuous field. Its visual channels remain tied to measured quantities:

- Hue follows the classifier's probability of the second spiral class.
- Brightness follows certainty, so the ambiguous decision region stays shadowed rather than being falsely resolved.
- Fine contour lines are probability level sets, with the bright central line marking the true 50% decision boundary.
- Root-like streamlines follow the gradient of predicted probability through input space.
- The minute coral and mint points are the actual training examples.
- The small gold route is the parameter trajectory projected with PCA from the network's full weight space.

The result is intended to be read from a distance as a landscape, and at close range as a record of an actual learned function.

### Production Files

- [2560 x 1440 master still](outputs/master.png)
- [32-frame production loop](outputs/loop-production.gif)
- [Probability, uncertainty, gradients, and parameter trace](outputs/production-data.npz)
- [Exact visual mapping and run metadata](outputs/production-metadata.json)

Render this edition with:

```bash
/home/fzeng/ml/research/.venv/bin/python render_production.py
```
