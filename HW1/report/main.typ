#import "settings.typ": *

#show: conf.with(
  title: "My Research Paper",
)
#let today = datetime.today()
#title-page(
  logo_left: "./img/eng-logo.png",
  logo_right: "./img/logo.png",
  title: "Distributed Systems",
  subtitle: "HW1",
  name: "Sepanta Ghonoodi",
  id: "2111143200",
  date: today.display("[month repr:long] [day], [year]"),
)

#outline(title: "Table of Contents")

// #outline(
//   title: [List of Figures],
//   target: figure,
// )
#pagebreak()
= Part 1

= Part 2
#include "part2.typ"
= Part 3
#include "part3.typ"