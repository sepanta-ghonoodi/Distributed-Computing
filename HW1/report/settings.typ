// =============================================================================
//   SETTINGS.TYP — Global Style, Packages, and Configuration
//   Author: Sepanta Ghonoodi (Translated to Typst)
// =============================================================================

// -----------------------------------------------------------------------------
//   1. COLOR PALETTE
// -----------------------------------------------------------------------------
#let main-color = rgb("#003066")
#let fade-color = rgb("#eaeffb")
#let fade-left = fade-color
#let fade-right = rgb("FFFFFF")

// Helper function to mix colors (simulating LaTeX's fadeleft!50!white)
#let mix-color(c1, p1, c2) = color.mix((c1, p1), (c2, 100% - p1))

// -----------------------------------------------------------------------------
//   2. HELPER COMPONENTS & BOXES
// -----------------------------------------------------------------------------

// Description Box (Equivalent to descriptionbox)
#let description-box(body) = block(
  width: 100%,
  fill: mix-color(fade-left, 40%, white),
  inset: (x: 6pt, y: 4pt),
  spacing: 1em,
  body
)

// Fading Abstract Environment
#let custom-abstract(body) = {
  block(
    width: 100%,
    fill: gradient.linear(fade-left, fade-right),
    inset: (y: 4pt),
    align(center, text(weight: "bold", size: 12pt, "Abstract"))
  )
  pad(x: 2em, body)
}

// Insert Image Shortcut
#let add-img(path, caption: none) = {
  figure(
    image(path, width: 100%),
    caption: caption
  )
}

// References Section Shortcut
#let references-section(bib-file) = {
  pagebreak()
  block(
    width: 100%,
    fill: gradient.linear(fade-left, fade-right),
    inset: (x: 6pt, y: 4pt),
    spacing: 1.5em,
    text(weight: "bold", size: 12pt, "References")
  )
  v(3em)
  bibliography(bib-file, style: "ieee", title: none)
}

// Title Page Generator
// Title Page Generator
#let title-page(
  logo_left: none,
  logo_right: none,
  title: "",
  subtitle: "",
  name: "",
  id: "",
  date: ""
) = {
  // 1. TURN OFF THE HEADER FOR THE TITLE PAGE
  set page(header: none)

  // Pre-calculate images to avoid scoping issues in grid
  let img_l = if logo_left != none { image(logo_left, width: 3cm) } else { [] }
  let img_r = if logo_right != none { image(logo_right, width: 3cm) } else { [] }

  align(center)[

    #grid(
      columns: (1fr, 2fr, 1fr),
      align: (center, center, center),
      img_l,
      text(size: 12pt)[University of Tehran \ College of Engineering \ School of Electrical and Computer Engineering],
      img_r
    )
    #v(1fr)
    #text(size: 16pt, weight: "bold", title) \
    #v(2em)
    #text(size: 14pt, weight: "bold", subtitle) \
    #v(1fr)
    #text(size: 12pt, "Full Name") \
    #text(size: 14pt, name) \
    #v(3em)
    #text(size: 12pt, "Student ID") \
    #text(size: 14pt, id) \
    #v(1fr)
    #text(size: 12pt, date) \
    #v(2em)
  ]

  pagebreak()

  // 2. RESET THE PAGE COUNTER (Optional but recommended)
  // This makes the page directly after the title page become "Page 1"
  counter(page).update(1)
}

#let en(body) = block(width: 100%, dir:ltr )[
  #set text(lang: "en", font: "Times New Roman")
  #body
]
#let fa(body) = block(width: 100%)[
  #set text(lang: "fa", font: "XB Niloofar",dir: rtl)
  #body
]

// -----------------------------------------------------------------------------
//   3. MAIN TEMPLATE FUNCTION
// -----------------------------------------------------------------------------
#let conf(
  title: "",
  body
) = {
  set figure(numbering: "1")
  show figure.caption: set align(center)
  // PAGE LAYOUT, BORDER & HEADER
  set page(
    paper: "a4",
    margin: 1in,
    background: place(
      dx: 0.75cm,
      dy: 0.75cm,
      rect(
        width: 100% - 1.5cm,
        height: 100% - 1.5cm,
        stroke: 0.7pt + main-color.transparentize(20%),
      ),
    ),
    header: context {
      let page-num = counter(page).display()

      // 1. Find all Level 1 headings on the CURRENT page
      let headings = query(
        selector(heading.where(level: 1)).after(here()),
      ).filter(it => it.location().page() == here().page())

      // 2. Decide what to show
      let current-heading = if headings.len() > 0 {
        // If there is a heading on this page, show the FIRST one
        headings.first().body
      } else {
        // If no heading is on this page, look back to the previous section
        let previous = query(
          selector(heading.where(level: 1)).before(here()),
        )

        if previous.len() > 0 { previous.last().body } else { "" }
      }

      grid(
        columns: (1fr, 1fr),
        align: (start, end),
        [#page-num], [#current-heading],
      )
      v(-0.4em)
      block(width: 100%, height: 0.8pt, fill: gradient.linear(main-color.transparentize(20%), white))
    },
  )

  // FONTS, LINKS & LISTS
  set text(
      font: "Times New Roman",
      size: 11pt,
      lang: "en",
      dir: ltr
    )

  // 2. The Language Override!
  // This intercepts any Farsi letters (and half-spaces) and forces them into Vazirmatn
  show regex("[\p{Arabic}\u{200C}]+"): set text(font: "XB Niloofar")
  // Left-to-Right Environment for English Paragraphs

  show link: set text(fill: blue)
  set enum(numbering: "(a)")
  show outline: set outline(depth: 3)
  // TABLE OF CONTENTS STYLING
  show outline.entry.where(level: 1): it => {
    v(12pt, weak: true)
    strong(it)
  }

  // SECTION FORMATTING (Replacing TColorBoxes)
  set heading(numbering: none)

  // H1 (Equivalent to \section -> questionbox)
  show heading.where(level: 1): it => {
    set text(weight: "bold", size: 12pt)
    block(
      width: 100%,
      fill: gradient.linear(fade-left, fade-right),
      inset: (x: 6pt, y: 4pt),
      spacing: 1.5em,
      [#it.body]
    )
  }

  // H2 (Equivalent to \subsection -> qpartbox)
  show heading.where(level: 2): it => {
    set text(weight: "regular", size: 11pt)
    block(
      width: 100%,
      fill: mix-color(fade-left, 50%, white),
      inset: (x: 6pt, y: 3pt),
      spacing: 1em,
      [#strong[#it.body]]
    )
  }



  // H3 (Equivalent to \subsubsection -> Standard Bold Text)
  // Typst handles this correctly out of the box with the default heading rules.

  // CODE HIGHLIGHTING (Equivalent to Minted)
  show raw.where(block: true): it => block(
    width: 100%,
    fill: mix-color(fade-left, 15%, white),
    stroke: 0.5pt + mix-color(fade-left, 50%, main-color),
    inset: 8pt,
    radius: 2pt,
    text(size: 9pt, it)
  )

  // RENDER DOCUMENT BODY
  body
}
