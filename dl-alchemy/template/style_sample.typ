// Style sample: every box the course documents use, filled with CS 312's own
// public example problem and quiz questions (verbatim from their site,
// 2026-09-19). Not a unit of this course.
#import "/template/template.typ": *

#show: doc.with(
  unit: none,
  title: "Style sample: the CS 312 example problem and quiz questions",
  kind: "Assignment handout / Prediction quiz / Quiz debrief",
  author: "dl-alchemy (after Stanford CS 312: Deep Learning Alchemy)",
  date: "2026-09-19",
  banner: "content verbatim from deep-learning-alchemy.github.io; layout only",
)

= What a handout looks like <sec-handout>

An assignment scopes out a phenomenon and lists directions to explore. Each direction is an orange Problem box
with a suggested GPU budget. The baseline code the diffs refer to is printed once, in a grey panel.

#problem("ablations", "How much does each component improve loss?", compute: "budget set at calibration")[
  Start from the baseline depth 8 transformer that we pre-train on 614M tokens. This transformer (implementation
  provided below) features multiple components such as prenorm, attention, MLP, residuals, and positional
  embeddings. In this problem, we want to understand how much each component improves loss. Explore the
  following directions.

  #parts(
    [Perform leave-one-out ablations on the role of each component in our current model.],
    [Understand the role of the residual connection. Can you come up with different schemes that improve the
     loss benefit of the residual connection?],
  )

  #deliverable[Plots or tables in your report, with one sentence per result saying what you now believe.]

  #resources[Stated per problem once the baseline run time on the GB10 is measured.]
]

#code[
```python
class Block(nn.Module):
    def __init__(self, width, ...):
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attn = Attention(width, ...)
        self.mlp_norm = RMSNorm(width)
        self.mlp = MLP(width, ...)
    def update(self, x):
        attn = self.attn(self.attn_norm(x))
        mlp = self.mlp(self.mlp_norm(x + attn))
        return attn + mlp
    def forward(self, x):
        return x + self.update(x)
```
]

#lowres("If the GPU window is short")[
  Blue boxes carry tips: a reduced version of a sweep that still answers the question, or a known sharp edge
  of this machine.
]

= What a quiz looks like <sec-quiz>

Each question is one held-out experiment that was actually run, given as a description and a code diff, with the
difficulty in the header. The layout of @sec-handout carries over.

#question(1, "easy", "Rank the ablations", points: 2)[
  We train each of the following depth-8 transformer variants using the same training recipe and
  hyperparameters. Rank the variants from lowest to highest final validation loss. Use $<$, $>$, or $=$ between
  the letters; use $=$ when the losses are within 0.01.

  #grid(columns: (1fr, 1fr, 1fr), row-gutter: 0.7em,
    [*A* Baseline], [*B* No prenorm], [*C* No residual],
    [*D* No MLP], [*E* No attention], [*F* No positional embeddings])

  #runin[No prenorm diff.]
  #code[
```diff
@@ -10,4 +10,4 @@ class Block.update
 def update(self, x):
-    attn = self.attn(self.attn_norm(x))
+    attn = self.attn(x)
-    mlp = self.mlp(self.mlp_norm(x + attn))
+    mlp = self.mlp(x + attn)
     return attn + mlp
```
  ]

  #runin[No residual diff.]
  #code[
```diff
@@ -16,2 +16,2 @@ class Block.forward
 def forward(self, x):
-    return x + self.update(x)
+    return self.update(x)
```
  ]

  #answerline()
]

#question(2, "hard", "Mixed residual", points: 3)[
  We train the default 8 layer transformer as well as a variant where instead of having the residual connection
  come from the previous layer, we take 0.5 contribution from the previous layer and 0.5 contribution from two
  layers before (except for the first layer). For each method, we take the best loss over six learning rates
  ${10^(-4), 3 times 10^(-4), 10^(-3), 3 times 10^(-3), 10^(-2), 3 times 10^(-2)}$. What is your prediction of
  the loss difference between each method, $L_"mixed residual" - L_"default"$?

  #code[
```diff
@@ -34,3 +34,9 @@ Transformer.forward
 x_two_back = None
 for i, block in enumerate(self.blocks):
-    x = block(x)
+    update = block.update(x)
+    residual = (
+        x if i == 0
+        else 0.5 * x + 0.5 * x_two_back
+    )
+    x_two_back, x = x, residual + update
```
  ]

  #answerline(label: [$L_"mixed residual" - L_"default" =$])
]

= What a debrief looks like

After the quiz, each question gets the measured outcome in a black box and the intuition that would have
predicted it in a blue box.

#result("Question 2")[
  #ph[Measured losses of both methods at each learning rate, the best-of-grid difference, the seed-noise band,
  your answer, and the score.]
]

#model("⟨name of the intuition⟩")[
  #ph[Two or three sentences: the picture of the network that makes this outcome the expected one, and the
  nearby experiment where the same picture would predict the opposite.]
]

#figure(
  table(columns: 4,
    [Variant], [LR], [Val loss], [Δ vs baseline],
    [#ph[name]], [#ph[lr]], [#ph[loss]], [#ph[Δ]],
  ),
  caption: [Tables and figures are numbered and captioned below, as in the day-N handouts.],
)
