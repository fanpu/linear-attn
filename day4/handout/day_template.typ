// ============================================================================
// dayN_assignment.typ — skeleton for one day's handout.
// Copy this file, replace every #ph[...] placeholder, delete these comments.
// Structure and order are fixed; only the content changes per day.
// Conventions (never restated in the handout): direction / hypothesis /
// backlog identifiers appear only on the `ids:` line; every quantity is
// defined where first used; nothing the reader must do lives outside a
// Problem; cross-references point only to numbered parts of this handout.
// ============================================================================
#import "template.typ": *

#show: handout.with(
  day: 0,
  question: "⟨the day's question, as one sentence⟩",
  version: "0.0.1",                 // N.0.k: N = day, k = revision of this handout
  author: "⟨author line⟩",
  date: "⟨date⟩",
  ids: "⟨bookkeeping identifiers (direction, hypothesis) — title page only⟩",
)

= Assignment Overview

#ph[One paragraph: the question, why it matters in terms a fresh reader can
follow, and what result closes the day.]

==== What you will implement
+ #ph[Component] (@sec-background)
+ #ph[Component] (@sec-accounting)

==== What you will run
+ #ph[Script or sweep] (@sec-run)

==== What you will write
+ Predictions (@sec-predictions)
+ The experiment log (@sec-observe)
+ The finding in your own words (@sec-wrong)

==== What you can use
#ph[Which libraries and existing modules may be called, with a one-line
description each, and which pieces must be written from scratch. This is
where the you/AI split is stated for the reader.]
- `⟨library⟩`: #ph[one-line description].
- Written from scratch: #ph[the reference recurrence, the metric, the counters].

==== What the code looks like
+ `⟨package⟩/`: #ph[stubs for every you-owned function, typed signature and docstring, raising] `NotImplementedError`.
+ `tests/`: #ph[one test per you-owned Problem, run as] `pytest -k test_<slug>`; `tests/adapters.py` #ph[in the CS336 pattern].
+ `scripts/`: #ph[every script the handout names].
+ `configs/`: #ph[today's values, so no script is edited to set a budget].
+ `log_entry_template.md`: the log skeleton printed in @sec-observe.
+ `README.md`: #ph[setup, running tests and scripts, expected runtimes on the GB10].

==== Before you start
#ph[What must exist before the clock starts, as files and their contents.]

#lowres("If things go slowly")[
  #ph[The reduced result that still closes the day if hour three arrives with
  the main line not working, with the concrete settings to switch to.]
]

==== Who uses this
#ph[Which later experiments depend on today's output, described by what they do.]

= Background <sec-background>

#ph[Open with why the simpler object is not enough. Whole before parts: the
complete object with every tensor's shape stated in words. Then the equation,
with every symbol defined.]

$ "⟨baseline equation⟩" $ <eq-baseline>

#ph[Citations are for depth only, inline and with a section number, as in]
#cite(<yang2025gdn>, supplement: [§3]).

== Remark: conventions
#ph[Notation, memory layout, which axis is which, sign conventions, what
"state size" means today — before the first place they could cause a silent
bug.]

#example("slug", "⟨hand-traceable instance⟩")[
  #ph[A small instance of @eq-baseline the reader can trace by hand, and the
  worked answer.]
]

#ph[Toy before theory: a runnable ten-to-twenty-line snippet showing the
phenomenon, then a one-point Problem that varies one knob.]
```python
>>> ⟨snippet⟩
```

#problem("slug", "⟨Mechanism check⟩", points: 1, owner: "you")[
  #parts(
    [#ph[Vary one knob.]

     #deliverable[A one-to-two sentence response.]],
  )
]

= Accounting <sec-accounting>

#ph[Motivate: why the background section's result is not enough. At least one
pencil-and-paper part per day, later checked against the measured number.
Given values are marked as given, with the reason they are given.]

#setting("⟨label⟩", $"⟨value⟩"$)[#ph[Reason for this value.]]
#setting("⟨label⟩", $"⟨value⟩"$)[#ph[Reason for this value.]]

#figure(
  table(
    columns: 3,
    [⟨Config⟩], [⟨Column⟩], [⟨Column⟩],
    [⟨row⟩], [⟨value⟩], [⟨value⟩],
  ),
  caption: [#ph[Configurations used today.]],
) <tab-configs>

#problem("slug", "⟨Implementation⟩", points: 2, owner: "you")[
  #ph[What to build, what the inputs and outputs are, how the reader knows it
  works, and what it costs in time and memory.]

  #defline[`def ⟨name⟩(⟨args⟩) -> ⟨type⟩`]
  #iolist(
    inputs:  (("⟨arg: type⟩", [#ph[Semantics and shape.]]),),
    outputs: (("⟨out: type⟩", [#ph[Semantics and shape.]]),),
  )

  #parts(
    [#ph[Pencil-and-paper part, for the configurations in @tab-configs.]

     #deliverable[An expression of the form $a dot B + b$ with numerical $a$ and $b$.]],
    [#ph[Implementation part.]

     #testline("slug")

     #deliverable[#ph[Artifact and its size.]]],
  )
]

= Predictions <sec-predictions>

#problem("predictions", "Predictions", points: 1, owner: "you")[
  Log these before proceeding.
  #parts(
    [#prediction(
      what: [#ph[What the quantity is and why today cares about it.]],
      definition: [#ph[Definition, unit, and the script or output field that produces it.]],
      reference: [#ph[Reference value the reader already has, with its provenance in the sentence.]],
    )],
  )
]

= Experiments <sec-run>

#ph[The pitfall, explained before the Problem that meets it.]

#algorithm("⟨title⟩",
  (0, [#kw[Require:] #ph[inputs]]),
  (0, [#kw[for] #ph[loop] #kw[do]]),
  (1, [#ph[body]]),
  (0, [#kw[end for]]),
  (0, [#kw[Return] #ph[output]]),
) <alg-main>

#problem("slug", "⟨Script⟩", points: 1, owner: "AI")[
  #ph[Write the script implementing @alg-main, reading its configuration from
  a file in] `configs/`.

  #deliverable[#ph[The script.]]
]

#debugtip("⟨sharp edge or sanity check⟩")[
  #ph[State the sharp edge in full, with the measured reference number when one
  exists.]
]

#problem("slug", "⟨Main result⟩", points: 3, compute: "⟨x⟩ GB10 hrs", owner: "you")[
  #resources[$<= "⟨t⟩"$ minutes, $<= "⟨m⟩"$ GB GPU memory]

  #parts(
    [Run #ph[the script]. #ph[Hint: expected order of magnitude, and the first
    thing to check when it is missed.]

     #deliverable[#ph[Pipeline health: how many runs failed, and what the failures look like.]]],
    [#deliverable[#ph[Throughput, compared with the reference number given above.]]],
    [#deliverable[#ph[The metric: a table with named columns, or a plot with the seed band shaded.]]],
    [#deliverable[#ph[Ten random failures or the worst cases, and what they have in common.]]],
  )
]

== Ablation 1: ⟨name⟩
#ph[One motivating sentence. Baseline and modified equation side by side.]

$ "⟨baseline equation⟩" $ <eq-abl-base>
$ "⟨modified equation⟩" $ <eq-abl-mod>

#problem("slug", "⟨Ablation⟩", points: 1, compute: "⟨x⟩ GB10 hrs", owner: "AI", optional: true)[
  #resources[$<= "⟨t⟩"$ minutes, $<= "⟨m⟩"$ GB GPU memory]
  #parts(
    [#ph[Run @eq-abl-base against @eq-abl-mod.]

     #deliverable[One learning curve comparing the two, and a few sentences of commentary.]],
  )
]

= What to observe <sec-observe>
#ph[The numbers and plots that matter, what would count as a surprise, and
what would count as a broken pipeline rather than a result.]

#problem("log_entry", "Experiment log", points: 1, owner: "you")[
  Fill in the entry below. It is identical to `log_entry_template.md`.
  #logentry(day: 0, date: "⟨YYYY-MM-DD⟩", question: "⟨the day's question⟩",
            setup: "⟨sizes, tokens, seeds, what changed vs baseline⟩")
  #deliverable[The filled-in entry.]
]

= Things that are easy to get wrong <sec-wrong>
- #ph[One line each.]

#problem("explain_it_back", "Explain it back", points: 1, owner: "you")[
  #parts(
    [State the finding and its mechanism in your own words.

     #deliverable[A few sentences.]],
  )
]

#bibliography("refs.bib", style: "handout.csl", title: "References")
