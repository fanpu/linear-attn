// ============================================================================
// Daily research handout template (CS336 Spring 2026 visual style)
//
// Visual specs measured from the compiled CS336 handouts: US letter, 1in
// margins, New Computer Modern 10pt, DejaVu Sans Mono 8pt, 1.5pt coloured
// frames. Do not change anything visual between days; every day's handout
// must look the same. Day-specific content lives in dayN_assignment.typ.
//
// Element inventory (what each helper is for, one line each):
//   handout(...)        document wrapper and title page (day, question, version N.0.k, ids line)
//   problem(...)        orange box; title = "(slug): Title (0.5 GB10 hrs) (N points) [you]"
//   example(...)        black box; hand-traceable instance with the worked answer
//   lowres(...)         blue box "Low-Resource Tip: ..."; the reduced result that still closes the day
//   debugtip(...)       blue box "Debugging Tip: ..."; a GB10 sharp edge or a standard sanity check
//   parts(...)          (a), (b), (c) sub-parts inside a problem
//   deliverable[...]    "Deliverable: ..." line ending every part
//   resources[...]      "Resource requirements: ..." line for any problem that runs anything
//   testline("slug")    the fixed sentence closing every [you] implementation part
//   defline[...]        a `def` line of an interface, in mono
//   arglist(...)        name: type entries with one sentence of semantics each
//   iolist(...)         Input / Output lists for a function interface
//   setting(...)        run-in-label list entry: value plus the reason for it
//   runin[...]          bold run-in label for fixed slots ("Prompting setup.")
//   note[...]           "Note: ..." paragraph separating two things the reader could confuse
//   prediction(...)     one prediction part in the fixed four-step order
//   logentry(...)       the experiment-log skeleton, identical to log_entry_template.md
//   algorithm(...)      numbered Algorithm box
//   prompt(...)         grey mono panel (prompts, file listings, shell sessions)
// ============================================================================

#let blue   = rgb("#0074D9")   // tip boxes
#let orange = rgb("#FF851B")   // problem boxes
#let red    = rgb("#FF4136")   // internal references
#let green  = rgb("#2ECC40")   // citations
#let pink   = rgb("#FF66FF")   // external URLs
#let gray-fill = rgb("#F0F0F0") // prompt blocks
#let bar-gray  = rgb("#AAAAAA") // algorithm indent guides

#let mono = "DejaVu Sans Mono"

// ---------------------------------------------------------------------------
// Document wrapper
//
//   day:      integer, printed as "Day N: " before the question
//   question: the day's question as one sentence; becomes the title
//   version:  "N.0.k" where N is the day and k the revision of the handout
//   author:   author line
//   date:     date line
//   ids:      optional bookkeeping line (direction / hypothesis identifiers).
//             This is the only place such identifiers may appear.
// ---------------------------------------------------------------------------
#let handout(day: none, question: "", version: none, author: "", date: "", ids: none, body) = {
  let title = if day == none { question } else { "Day " + str(day) + ": " + question }
  set document(title: title)
  set page(paper: "us-letter", margin: 1in, numbering: "1", number-align: center)
  set text(font: ("New Computer Modern", "DejaVu Sans"), size: 10pt, lang: "en")
  set par(justify: false, leading: 0.65em, spacing: 0.95em)

  show raw: set text(font: mono, size: 8pt)
  show math.equation: set text(font: "New Computer Modern Math")
  set math.equation(numbering: "(1)")

  set enum(indent: 0pt, spacing: 0.9em)
  set list(indent: 0pt, spacing: 0.9em, marker: [•])

  set footnote.entry(separator: line(length: 140pt, stroke: 0.5pt))
  show footnote.entry: set text(size: 8.5pt)

  // Headings: 14 / 12 / 10 pt bold; level 4 = unnumbered run-in subhead.
  set heading(numbering: "1.1.1")
  show heading: set text(weight: "bold")
  show heading.where(level: 1): set text(size: 14pt)
  show heading.where(level: 2): set text(size: 12pt)
  show heading.where(level: 3): set text(size: 10pt)
  show heading.where(level: 4): set text(size: 10pt)
  show heading.where(level: 4): set heading(numbering: none, outlined: false)
  show heading.where(level: 1): set block(above: 1.6em, below: 1em)
  show heading.where(level: 2): set block(above: 1.5em, below: 0.9em)
  show heading.where(level: 3): set block(above: 1.3em, below: 0.8em)
  show heading.where(level: 4): set block(above: 1.2em, below: 0.7em)

  // Figures / tables: "Figure 1: caption", "Table 1: caption", centred below.
  set figure(gap: 0.9em)
  set figure.caption(separator: [: ])
  show figure: set block(above: 1.4em, below: 1.4em)
  set table(stroke: 1pt + black, inset: (x: 6pt, y: 4.5pt), align: center + horizon)
  show table.cell.where(y: 0): set text(weight: "bold")

  // Boxed links, like hyperref-style PDF link borders.
  show ref: it => {
    if it.element == none { it }   // citations: boxed by the cite rule below
    else { box(stroke: 0.5pt + red, outset: (y: 1.5pt), it) }
  }
  show cite: it => box(stroke: 0.5pt + green, outset: (y: 1.5pt), it)
  show link: it => box(stroke: 0.5pt + pink, outset: (y: 1.5pt), it)

  // Title block
  align(center)[
    #text(size: 17pt, title)
    #v(0.35em)
    #if version != none [#text(size: 12pt)[Version #version]#v(0.35em)]
    #author
    #v(0.35em)
    #date
    #if ids != none [#v(0.35em)#text(size: 8.5pt, fill: luma(110), font: mono, ids)]
  ]
  v(0.6em)
  body
}

// ---------------------------------------------------------------------------
// Framed boxes
// ---------------------------------------------------------------------------
#let _frame(color, width, header, body) = block(
  width: 100%, breakable: true, above: 1.6em, below: 1.6em,
  stroke: width + color, inset: 0pt,
)[
  #block(inset: (x: 8pt, top: 12pt, bottom: 2pt), width: 100%)[
    #text(weight: "bold", header)
    #v(-0.2em)
    #line(length: 100%, stroke: 0.2pt + black)
  ]
  #block(inset: (left: 18pt, right: 8pt, top: 6pt, bottom: 10pt), width: 100%, body)
]

#let _name(n) = text(font: mono, size: 8pt, weight: "bold", n)

/// Orange problem box.
///   points   -> "(2 points)"          weight; points across the handout sum to about 10
///   compute  -> "(0.5 GB10 hrs)"      present only when the problem consumes GPU time
///   owner    -> "[you]" or "[AI]"     who writes the code (silent failure -> you; loud failure -> AI)
///   optional -> prefixes "OPTIONAL: " so the reader knows it can be skipped when behind
#let problem(name, title, points: none, compute: none, owner: none, optional: false, body) = {
  let pts = if points == none { none } else if points == 1 { [1 point] } else { [#points points] }
  let suffix = []
  if compute != none { suffix += [ (#compute)] }
  if pts != none { suffix += [ (#pts)] }
  if owner != none { suffix += [ #text(font: mono, size: 8pt)[[#owner]]] }
  let head = if optional { [OPTIONAL: #title] } else { title }
  _frame(orange, 1.5pt, [Problem (#_name(name)):#h(0.6em)#head#suffix], body)
}

/// Black example box: a hand-traceable instance and its worked answer.
#let example(name, title, body) = _frame(black, 0.5pt, [Example (#_name(name)):#h(0.6em)#title], body)

/// Blue box. `lowres: true` prefixes "Low-Resource Tip: "; `debug: true` prefixes "Debugging Tip: ".
#let tip(title, lowres: true, debug: false, body) = _frame(
  blue, 1.5pt,
  if debug [Debugging Tip: #title] else if lowres [Low-Resource Tip: #title] else [#title],
  body)

/// Shorthands for the two tip kinds every handout uses.
#let lowres(title, body) = tip(title, lowres: true, body)
#let debugtip(title, body) = tip(title, debug: true, body)

/// "Deliverable: ..." — ends every lettered part. Name the artifact and its size.
#let deliverable(body) = [*Deliverable*: #body]

/// "Resource requirements: ..." — on every problem that runs anything. The
/// playtest sums these lines to at most four hours.
#let resources(body) = [*Resource requirements*: #body]

/// Fixed sentence closing every [you] implementation part.
#let testline(slug) = [
  To test your implementation, implement the adapter
  #raw("[adapters.run_" + slug + "]"). Then, run
  #raw("pytest -k test_" + slug) and make sure your implementation passes.
]

/// (a), (b), (c) sub-parts inside problems.
#let parts(..items) = enum(numbering: "(a)", indent: 0pt, body-indent: 0.8em, spacing: 1.1em, ..items)

/// Bold run-in paragraph label: #runin[Evaluation metric.] Text continues...
/// Only for fixed slots ("Prompting setup.", "Evaluation metric.",
/// "Generation hyperparameters.") and for settings lists; never to stack an argument.
#let runin(label) = [*#label* ]

/// "Note: ..." paragraph, used only to separate two things the reader could confuse.
#let note(body) = [*Note*: #body]

// ---------------------------------------------------------------------------
// Interfaces for [you] code, in the exemplars' form.
//   #defline[`def rmsnorm(x: Tensor, eps: float) -> Tensor`]
//   #arglist(("x: Tensor", [Input activations of shape (batch, seq, d_model).]), ...)
//   #iolist(inputs: (("q: Tensor", [...]),), outputs: (("y: Tensor", [...]),))
// ---------------------------------------------------------------------------
#let defline(body) = block(above: 0.9em, below: 0.6em, text(font: mono, size: 8pt, body))

#let arglist(..entries) = {
  let rows = entries.pos().map(((sig, desc)) => [#text(font: mono, size: 8pt, weight: "bold", sig) #desc])
  list(indent: 12pt, spacing: 0.7em, ..rows)
}

#let iolist(inputs: (), outputs: ()) = {
  [*Input*]
  arglist(..inputs)
  [*Output*]
  arglist(..outputs)
}

// ---------------------------------------------------------------------------
// Settings list: each value carries its reason.
//   #setting("d_ff", $1344$)[This is roughly $8/3 d_"model"$ while being a multiple of 64.]
// ---------------------------------------------------------------------------
#let setting(label, value, reason) = block(above: 0.7em, below: 0.7em)[*#label* #value. #reason]

// ---------------------------------------------------------------------------
// Prediction part. Prints the four fixed steps in order, then the answer
// format in the log template's wording. Use one per quantity, inside #parts.
//   #prediction(
//     what:      [...one sentence: what the quantity is and why today cares...],
//     definition:[...unit, and the script or output field that produces it...],
//     reference: [...a value the reader already has, with its provenance...],
//   )
// ---------------------------------------------------------------------------
#let prediction(what: [], definition: [], reference: none) = [
  #what #definition
  #if reference != none [#reference]

  #deliverable[A value, a range, a direction where applicable, and a confidence
  (high / medium / low), in the wording of the log entry's prediction line.]
]

// ---------------------------------------------------------------------------
// Experiment-log skeleton. Prints the RESEARCH_PROGRAM.md daily entry with the
// question and setup filled in and the remaining lines empty. The text here and
// the shipped log_entry_template.md must be identical.
// ---------------------------------------------------------------------------
#let _logtext(day, date, question, setup) = (
  "### Day " + str(day) + ", " + date + "\n" +
  "Question: " + question + "\n" +
  "Prediction (written BEFORE running): what number do I expect, and how confident?\n" +
  "Setup: " + setup + "\n" +
  "Result: (one plot/table, path in repo)\n" +
  "Prediction correct? (yes / no / partially, and what surprised me)\n" +
  "What I now believe:\n" +
  "Spawned question(s):\n" +
  "Hours spent:"
)

// ---------------------------------------------------------------------------
// Prompt block: grey panel, 8pt mono, long lines soft-wrapped with "↪ "
// ---------------------------------------------------------------------------
#let prompt(text-content, width: 94) = {
  let wrap-line(line) = {
    if line.len() <= width { return (line,) }
    let out = ()
    let words = line.split(" ")
    let cur = ""
    let limit = width
    for w in words {
      let candidate = if cur == "" { w } else { cur + " " + w }
      if candidate.clusters().len() > limit and cur != "" {
        out.push(cur)
        cur = w
        limit = width - 2
      } else { cur = candidate }
    }
    out.push(cur)
    out.enumerate().map(((i, l)) => if i == 0 { l } else { "↪ " + l })
  }
  let src = if type(text-content) == str { text-content } else { text-content.text }
  let lines = src.split("\n").map(wrap-line).flatten()
  block(width: 100%, fill: gray-fill, inset: (x: 8pt, y: 8pt), above: 1.2em, below: 1.2em, breakable: true,
    text(font: mono, size: 8pt, par(leading: 0.55em, lines.map(l => raw(l)).join(linebreak()))))
}

#let logentry(day: 1, date: "", question: "", setup: "") = prompt(_logtext(day, date, question, setup))

// ---------------------------------------------------------------------------
// Algorithm box: 2pt top/bottom rules, numbered lines, grey indent guides.
// Usage:  #algorithm("Title", (0, [*Require:* ...]), (1, [body]), ...)
// ---------------------------------------------------------------------------
#let algorithm(title, ..lines) = {
  let lines = lines.pos()
  let step = 10pt
  figure(kind: "algorithm", supplement: [Algorithm], outlined: false, block(width: 80%, breakable: false, above: 1.6em, below: 1.6em, inset: 0pt)[
    #align(center)[#block(width: 100%)[
      #line(length: 100%, stroke: 2pt)
      #v(-0.4em)
      #align(left)[*Algorithm #context counter(figure.where(kind: "algorithm")).display():* #title]
      #v(-0.5em)
      #line(length: 100%, stroke: 1pt)
      #v(-0.3em)
      #grid(
        columns: (14pt, 1fr), row-gutter: 0pt, column-gutter: 6pt,
        ..lines.enumerate().map(((i, entry)) => {
          let (depth, body) = entry
          (
            align(right + horizon, text(size: 8pt)[#(i + 1)]),
            {
              let inner = body
              for d in range(depth) {
                inner = block(stroke: (left: 1pt + bar-gray), inset: (left: 5pt),
                              outset: (y: 3.2pt), width: 100%, inner)
              }
              block(inset: (y: 2.6pt), width: 100%, align(left, inner))
            },
          )
        }).flatten()
      )
      #v(-0.2em)
      #line(length: 100%, stroke: 2pt)
    ]]
  ])
}

// Handy bold keywords for algorithms.
#let kw(s) = text(weight: "bold", s)

/// Placeholder marker for the skeleton: grey italic ⟨text⟩. Must not survive into a delivered handout.
#let ph(body) = text(fill: luma(110), style: "italic")[⟨#body⟩]
