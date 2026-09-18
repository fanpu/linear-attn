# Daily research handout template

Typst template for the one-result-per-day sprint handouts. Visual style is
measured from the CS336 Spring 2026 assignment PDFs and must not change
between days; only `dayN_assignment.typ` changes.

## Files

- `template.typ` — preamble, title page, and every box/helper. Shared by all days; never edited per day.
- `day_template.typ` — the skeleton of one day's handout. Copy to `dayN_assignment.typ`, replace every `#ph[...]` placeholder, delete the header comments.
- `handout.csl` — citation style: `[Author et al., year, §n]` inline, numbered reference list.
- `refs.bib` — shared bibliography; add entries as days cite them.
- `log_entry_template.md` — the daily log skeleton; must be identical to the text `#logentry(...)` prints in the handout.

## Compile

    typst compile dayN_assignment.typ

Fonts (New Computer Modern, DejaVu Sans Mono) are bundled with Typst; no install needed.

## Helpers in `template.typ`

| Helper | Renders |
|---|---|
| `handout.with(day:, question:, version:, author:, date:, ids:)` | title page: "Day N: question", Version N.0.k, author, date, one grey bookkeeping line |
| `#problem("slug", "Title", points:, compute:, owner:, optional:)` | orange box: `Problem (slug): Title (x GB10 hrs) (N points) [you]`; `optional: true` prefixes OPTIONAL |
| `#example("slug", "Title")[...]` | black box for a hand-traceable instance |
| `#lowres("Title")[...]` / `#debugtip("Title")[...]` | blue boxes "Low-Resource Tip:" / "Debugging Tip:" |
| `#parts([...], [...])` | (a), (b), (c) sub-parts |
| `#deliverable[...]` | "Deliverable: ..." line ending every part |
| `#resources[...]` | "Resource requirements: ..." line on every problem that runs anything |
| `#testline("slug")` | the fixed adapter/pytest sentence closing every [you] implementation part |
| `#defline[...]`, `#arglist(...)`, `#iolist(inputs:, outputs:)` | interface specs in the exemplars' form |
| `#setting("label", value)[reason]` | run-in-label settings entry with its reason |
| `#runin[Label.]` | bold run-in label for fixed slots only |
| `#note[...]` | "Note: ..." paragraph |
| `#prediction(what:, definition:, reference:)` | one prediction part in the fixed order, ending in the log-template answer format |
| `#logentry(day:, date:, question:, setup:)` | the log skeleton, grey mono panel |
| `#algorithm("Title", (depth, [line]), ...)` | numbered Algorithm box; `#kw[for]` for keywords |
| `#prompt("...")` | grey mono panel for prompts, listings, shell sessions |
| `#ph[...]` | grey italic ⟨placeholder⟩; must not survive into a delivered handout |

Cross-reference with `@sec-x`, `@tab-x`, `@eq-x`, `@alg-x` (labels go on headings, figures, equations, algorithms — problem boxes cannot carry labels). Cite with `#cite(<key>, supplement: [§3])`.
