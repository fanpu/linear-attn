// ============================================================================
// training-dynamics course document template.
//
// Visual style copied from ../../day4/handout/template.typ (CS336-style: US
// letter, 1in margins, New Computer Modern 10pt, DejaVu Sans Mono 8pt, 1.5pt
// coloured frames), with the extra helpers of ../../dl-alchemy/template/template.typ.
// Only the look is shared. Do not change anything visual between assignments.
// Documents import it as "/template/template.typ" (build.sh sets the root to the
// course directory).
//
// Element inventory:
//   doc(...)            document wrapper and title block: "A0: <title>" (assignment, title, kind, author, date, banner)
//   problem(...)        orange box: "Problem (slug): Title [Derive + Check, 30 min] [you]"
//   tag(kind)           a [Derive] / [Implement] / [Check] / [Predict] / [Extension] / [Read] tag for a part
//   example(...)        black box; hand-traceable instance with its worked answer
//   result(...)         black box; the expected outcome of a check (solutions)
//   tip(...)            blue box with a free title (hints, gentle to strong)
//   lowres(...)         blue box "Low-Resource Tip: ..."
//   debugtip(...)       blue box "Debugging Tip: ..."
//   model(...)          blue box "Mental model: ..."
//   parts(...)          (a), (b), (c) sub-parts
//   deliverable[...]    "Deliverable: ..." line ending every part
//   hints(...)          "Hints, gentle to strong: (1) ... (2) ... (3) ..." ladder; one on every Derive part
//   primer(sec, what)   "(primer §ODE.4, logistic)": pointer into m0-math-primer/handout.pdf
//   resources[...]      "Resource requirements: ..." line for anything that runs
//   testline(...)       the fixed sentence closing every Implement part
//   defline[...]        a `def` line of an interface, in mono
//   arglist(...)        name: type entries with one sentence of semantics each
//   iolist(...)         Input / Output lists for a function interface
//   setting(...)        run-in-label list entry: value plus the reason for it
//   runin[...]          bold run-in label
//   note[...]           "Note: ..." paragraph
//   code[...]           grey panel around a raw block (keeps syntax highlighting)
//   prompt(...)         grey mono panel with soft-wrapped long lines (shell sessions, listings)
//   algorithm(...)      numbered Algorithm box
//   ph[...]             grey italic placeholder; must not survive into a delivered PDF
//
// Problem-title label convention. `kind` is a list of the problem's types, in the
// order the parts use them, drawn from:
//   "derive"     pen and paper; ends with a deliverable he can type into a stub or compare with a check
//   "implement"  a function in a named file, tested through tests/adapters.py
//   "check"      a short numerical experiment compared against the Derive deliverable
//   "predict"    write the expected outcome in notes.md before running a check
//   "extension"  optional; marked [GPU] when it needs the GPU, with a pasar line
//   "read"       a reading assignment with questions for notes.md
// e.g. #problem("gd-stability", [Stability of GD], kind: ("derive", "implement", "check"), time: "35 min", owner: "you")
//   -> Problem (gd-stability): Stability of GD [Derive + Implement + Check, 35 min] [you]
// ============================================================================

#let blue   = rgb("#0074D9")   // tip boxes
#let orange = rgb("#FF851B")   // problem / question boxes
#let red    = rgb("#FF4136")   // internal references
#let green  = rgb("#2ECC40")   // citations
#let pink   = rgb("#FF66FF")   // external URLs
#let gray-fill = rgb("#F0F0F0") // code panels
#let bar-gray  = rgb("#AAAAAA") // algorithm indent guides

#let mono = "DejaVu Sans Mono"

// ---------------------------------------------------------------------------
// Document wrapper
//
//   assignment: integer or none; printed as "AN: " before the title
//   title:   document title
//   kind:    second line, e.g. "Assignment handout", "Lecture notes", "Prediction quiz", "Quiz debrief"
//   author:  author line
//   date:    date line
//   banner:  optional small grey mono line under the date. Used for provenance,
//            e.g. "Reconstruction: not the official CS 312 handout".
// ---------------------------------------------------------------------------
#let doc(assignment: none, title: "", kind: none, author: "", date: "", banner: none, body) = {
  let full = if assignment == none { title } else { "A" + str(assignment) + ": " + title }
  set document(title: full)
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
  show figure.where(kind: table): set block(breakable: true)   // long tables may span pages
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
    #text(size: 17pt, full)
    #v(0.35em)
    #if kind != none [#text(size: 12pt, kind)#v(0.35em)]
    #author
    #v(0.35em)
    #date
    #if banner != none [#v(0.35em)#text(size: 8.5pt, fill: luma(110), font: mono, banner)]
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
  #block(inset: (x: 8pt, top: 12pt, bottom: 2pt), width: 100%, sticky: true)[
    #text(weight: "bold", header)
    #v(-0.2em)
    #line(length: 100%, stroke: 0.2pt + black)
  ]
  #block(inset: (left: 18pt, right: 8pt, top: 6pt, bottom: 10pt), width: 100%, body)
]

#let _name(n) = text(font: mono, size: 8pt, weight: "bold", n)

/// Problem types for the title label convention (see the header).
#let kinds = (
  derive: "Derive", implement: "Implement", check: "Check",
  predict: "Predict", extension: "Extension", read: "Read",
)

#let _kindname(k) = {
  assert(k in kinds, message: "unknown problem kind: " + k)
  kinds.at(k)
}

/// A small bold tag for one part: #tag("derive") -> [Derive]
#let tag(k) = text(font: mono, size: 8pt, weight: "bold")[[#_kindname(k)]]

/// Orange problem box.
///   kind     -> ("derive", "check", ...) printed as "[Derive + Check, 30 min]"
///   time     -> estimated time for him, e.g. "30 min"
///   owner    -> "you" or "AI": who writes the code / does the work, printed "[you]"
///   gpu      -> e.g. "~1 GB10 hr": marks a GPU extension, printed "[GPU, ~1 GB10 hr]"
///   optional -> prefixes "OPTIONAL: "
#let problem(name, title, kind: (), time: none, owner: none, gpu: none, optional: false, body) = {
  let label = kind.map(_kindname).join(" + ")
  if time != none { label = if label == none or label == "" { time } else { label + ", " + time } }
  let suffix = []
  if label != none and label != "" { suffix += [ #text(weight: "regular")[[#label]]] }
  if gpu != none { suffix += [ #text(font: mono, size: 8pt)[[GPU, #gpu]]] }
  if owner != none { suffix += [ #text(font: mono, size: 8pt)[[#owner]]] }
  let head = if optional { [OPTIONAL: #title] } else { title }
  _frame(orange, 1.5pt, [Problem (#_name(name)):#h(0.6em)#head#suffix], body)
}

/// Black boxes.
#let example(name, title, body) = _frame(black, 0.5pt, [Example (#_name(name)):#h(0.6em)#title], body)
#let result(title, body) = _frame(black, 0.5pt, [Result:#h(0.6em)#title], body)

/// Blue boxes.
#let tip(title, body) = _frame(blue, 1.5pt, title, body)
#let lowres(title, body) = _frame(blue, 1.5pt, [Low-Resource Tip: #title], body)
#let debugtip(title, body) = _frame(blue, 1.5pt, [Debugging Tip: #title], body)
#let model(title, body) = _frame(blue, 1.5pt, [Mental model: #title], body)

#let deliverable(body) = [*Deliverable*: #body]

/// Hint ladder for a Derive part, gentle to strong. The first hint names the technique (and the
/// primer section), the last is the first line of the worked solution.
///   #hints([...], [...], [...])
#let hints(..hs) = block(above: 0.7em, below: 0.2em)[_Hints, gentle to strong:_ #hs.pos().enumerate().map(((i, h)) => [(#(i + 1)) #h]).join(" ")]

/// Pointer into the math primer (m0-math-primer/handout.pdf) at the point a technique is used.
///   #primer("ODE.4", "logistic") -> (primer §ODE.4, logistic)
#let primer(sec, what) = [(primer §#sec, #what)]
#let resources(body) = [*Resource requirements*: #body]

/// (a), (b), (c) sub-parts.
#let parts(..items) = enum(numbering: "(a)", indent: 0pt, body-indent: 0.8em, spacing: 1.1em, ..items)

/// Fixed sentence closing every Implement part. `dir` is the assignment folder.
#let testline(slug, dir: "a0-prereqs") = [
  To test your implementation, run #raw("uv run pytest " + dir + " -k test_" + slug) (the adapter
  #raw("run_" + slug) in #raw("tests/adapters.py") is pre-wired to this signature; edit it only if yours differs).
]

// ---------------------------------------------------------------------------
// Interfaces for [you] code.
//   #defline[`def gd(grad_fn, theta0, eta, steps) -> np.ndarray`]
//   #arglist(("eta: float", [Step size.]), ...)
//   #iolist(inputs: (("x: Tensor", [...]),), outputs: (("y: Tensor", [...]),))
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

#let runin(label) = [*#label* ]
#let note(body) = [*Note*: #body]

/// Settings list: each value carries its reason.
///   #setting("depth", $8$)[Same as the CS 312 baseline.]
#let setting(label, value, reason) = block(above: 0.7em, below: 0.7em)[*#label* #value. #reason]

/// Grey panel around a raw block; keeps syntax highlighting.
///   #code[```diff ... ```]
#let code(body) = block(width: 100%, fill: gray-fill, inset: (x: 8pt, y: 8pt),
  above: 1.2em, below: 1.2em, breakable: true, body)

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

// ---------------------------------------------------------------------------
// Algorithm box: 2pt top/bottom rules, numbered lines, grey indent guides.
// Usage:  #algorithm("Title", (0, [*Require:* ...]), (1, [body]), ...)
// ---------------------------------------------------------------------------
#let algorithm(title, ..lines) = {
  let lines = lines.pos()
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

#let kw(s) = text(weight: "bold", s)

/// Placeholder marker: grey italic ⟨text⟩. Must not survive into a delivered PDF.
#let ph(body) = text(fill: luma(110), style: "italic")[⟨#body⟩]
