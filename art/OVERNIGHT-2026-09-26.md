# Overnight run — 2026-09-26 (GPU until 21:30 EDT)

New ML-phenomena art, at most two project agents at a time, all GPU work through pasar.
Shared brief: [_shared/OVERNIGHT-BRIEF.md](_shared/OVERNIGHT-BRIEF.md).

## Queue

| wave | project | idea | status |
|---|---|---|---|
| 1 | [ticket-shadow](ticket-shadow/) | lottery-ticket input mask as a perforated sheet that casts where the data lives | done |
| 1 | [one-road](one-road/) | function-space (InPCA) atlas: do all architectures take one road from ignorance to truth? | done |
| 2 | [drainage](drainage/) | greedy decoding's repetition loops as a river basin over the whole vocabulary | running |
| 2 | [palimpsest](palimpsest/) | does a network retrained on page B keep A's fine detail under B's broad strokes? | running |
| 3 | to-scale / super-weight | massive activations at true scale; the one weight that breaks the model | queued |
| 3 | unsayable | under-trained tokens as a type specimen, with the model's failed attempts to say them | queued |

## Results

### ticket-shadow — real, and the optimiser decides the shadow

LeNet-300-100 lottery tickets on MNIST (20 IMP rounds, 5 seeds) are genuine: 98.06% test at 3.5% of
input weights vs 97.52% dense; re-initialised mask 96.3%, random pruning 92.1% at 1.4%. Most of the
shadow is the data (pixel-variance R² 0.75; first-run weight movement R² 0.83). The finding: with raw
pixels, **Adam's ticket draws the outline of every pixel MNIST ever lights** (a flat plateau: pixels lit
in 10–99 images keep as many connections as pixels lit in 20,000+), while **SGD's draws a soft core that
tracks variance** (r 0.88). Per-class digit ghosts are a null. Best: `ticket-shadow/gallery/diptych_adam_vs_sgd_r15.png`.
Likely mechanism (coordinator's reading, untested): Adam normalises each weight's step, so a weight that
receives *any* gradient moves about as far as one that receives a lot. Next: AdamW / large-ε Adam runs to
pin it, then cut the pair as two perforated sheets.

### one-road — one road with lanes; forks on held-out data

64 small runs (logistic, MLPs, CNN, ResNet-8, ViT, GRU) on MNIST, Fashion-MNIST, CIFAR-10 and a+b mod 97,
embedded per task with InPCA (Mao et al. 2024). On training examples every one of 309 cross-architecture
pairs is closer at matched progress than a within-class-shuffled null (median ratio 0.48–0.69), and all
untrained nets start at the uniform point. But architectures sit 1.6–2.5× further apart than seeds: lanes,
not one road. **On held-out CIFAR the road forks into three valleys** (MLPs; CNN/GRU; ResNet/ViT).
Shuffled-label runs leave ignorance in another direction; grokking runs on mod 97 walk away from truth on
held-out pairs and come back. Caveat: the flattest-null picture depends on the metric (Bhattacharyya vs
Hellinger); the numbers barely move. Best: `one-road/gallery/plate_roads_cifar_te.png`. Coordinator's
note: the framed-plot plates read as figures; the CIFAR fork is the image. Next: 5 seeds on held-out CIFAR
to test whether the three valleys are stable.
