// ============================================================================
// dl-alchemy document template.
//
// Visual style copied from ../../day4/handout/template.typ (CS336-style: US
// letter, 1in margins, New Computer Modern 10pt, DejaVu Sans Mono 8pt, 1.5pt
// coloured frames). Only the look is shared; the helpers below are the ones
// this course needs. Do not change anything visual between units.
//
// Element inventory:
//   doc(...)            document wrapper and title block (unit, kind, title, author, date, banner)
//   problem(...)        orange box; one direction of an assignment: "Problem (slug): Title (~3 GB10 hrs)"
//   question(...)       orange box; one quiz question: "Question 2 (hard): Title (2 points)"
//   example(...)        black box; worked instance
//   result(...)         black box; debrief: the true outcome of a held-out experiment
//   tip(...)            blue box with a free title
//   lowres(...)         blue box "Low-Resource Tip: ..."
//   debugtip(...)       blue box "Debugging Tip: ..."
//   model(...)          blue box "Mental model: ..."; debriefs and lecture notes
//   parts(...)          (a), (b), (c) sub-parts
//   deliverable[...]    "Deliverable: ..." line
//   resources[...]      "Resource requirements: ..." line for anything that runs on the GPU
//   setting(...)        run-in-label list entry: value plus the reason for it
//   runin[...]          bold run-in label
//   note[...]           "Note: ..." paragraph
//   code[...]           grey panel around a raw block (keeps syntax highlighting; use ```diff for code diffs)
//   prompt(...)         grey mono panel with soft-wrapped long lines (shell sessions, listings)
//   answerline(...)     ruled blank for a quiz answer
//   algorithm(...)      numbered Algorithm box
//   ph[...]             grey italic placeholder; must not survive into a delivered PDF
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
//   unit:    integer or none; printed as "Unit N: " before the title
//   title:   document title
//   kind:    second line, e.g. "Assignment handout", "Lecture notes", "Prediction quiz", "Quiz debrief"
//   author:  author line
//   date:    date line
//   banner:  optional small grey mono line under the date. Used for provenance,
//            e.g. "Reconstruction: not the official CS 312 handout".
// ---------------------------------------------------------------------------
#let doc(unit: none, title: "", kind: none, author: "", date: "", banner: none, body) = {
  let full = if unit == none { title } else { "Unit " + str(unit) + ": " + title }
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

/// Orange problem box: one direction of an assignment.
///   compute  -> "(~3 GB10 hrs)"   suggested GPU budget for this direction
///   optional -> prefixes "OPTIONAL: "
#let problem(name, title, compute: none, optional: false, body) = {
  let suffix = if compute != none [ (#compute)] else []
  let head = if optional { [OPTIONAL: #title] } else { title }
  _frame(orange, 1.5pt, [Problem (#_name(name)):#h(0.6em)#head#suffix], body)
}

/// Orange quiz-question box.
///   n          -> question number
///   difficulty -> "easy" | "medium" | "hard"
///   points     -> "(2 points)"
#let question(n, difficulty, title, points: none, body) = {
  let pts = if points == none { [] } else if points == 1 { [ (1 point)] } else { [ (#points points)] }
  _frame(orange, 1.5pt, [Question #n (#_name(difficulty)):#h(0.6em)#title#pts], body)
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
#let resources(body) = [*Resource requirements*: #body]

/// (a), (b), (c) sub-parts.
#let parts(..items) = enum(numbering: "(a)", indent: 0pt, body-indent: 0.8em, spacing: 1.1em, ..items)

#let runin(label) = [*#label* ]
#let note(body) = [*Note*: #body]

/// Settings list: each value carries its reason.
///   #setting("depth", $8$)[Same as the CS 312 baseline.]
#let setting(label, value, reason) = block(above: 0.7em, below: 0.7em)[*#label* #value. #reason]

/// Grey panel around a raw block; keeps syntax highlighting.
///   #code[```diff ... ```]
#let code(body) = block(width: 100%, fill: gray-fill, inset: (x: 8pt, y: 8pt),
  above: 1.2em, below: 1.2em, breakable: true, body)

/// Ruled blank for a quiz answer.
#let answerline(label: [Answer:], width: 100%) = block(above: 1.2em, below: 0.6em, width: width)[
  #grid(columns: (auto, 1fr), column-gutter: 6pt, align: bottom,
    [*#label*], line(length: 100%, stroke: 0.4pt + black))
]

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
