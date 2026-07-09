#import "@preview/cetz:0.4.2": canvas
#import "@preview/cetz-plot:0.1.3": plot

= Concurrency Benchmark Results

#let data = csv("../part2/results/results.csv").slice(1)

#let benchmark-chart(
  workload-name,
  procs-list,
  x-step,
  y-step,
  y-col,
  col-name,
  plot-size: (4, 3),
  show-legend: false,
) = {
  let datasets = procs-list.map(p => {
    let filtered = data.filter(row => row.at(0) == workload-name and row.at(1) == str(p))
    let points = filtered.map(row => (float(row.at(2)), float(row.at(y-col))))
    (procs: p, points: points)
  })
  set text(size: 8pt)
  canvas({
    plot.plot(
      size: plot-size,
      x-label: "Goroutines (log scale)",
      y-label: col-name,
      ..(if show-legend { (legend-anchor: "north", legend: "south") } else { (:) }),
      x-min: 1,
      x-mode: "log",
      x-base: 2,
      y-min: 0,

      x-tick-step: x-step,
      y-tick-step: y-step,
      y-grid: true,
      y-format: "sci",
      {
        for ds in datasets {
          plot.add(
            ds.points,
            mark: "o",
            // label: "GOMAXPROCS=" + str(ds.procs),
            label: if show-legend { str(ds.procs) + " Cores" } else { none },
          )
        }
      },
    )
  })
}

// #benchmark-chart("CPU-bound", (1, 2, 4, 8), 1, 1000000000,5, "Throughput(task/second)")
// #benchmark-chart("CPU-bound", (1, 2, 4, 8,16), 1, 100,3,"Total Time(ms)")
// #benchmark-chart("CPU-bound", (1, 2, 4, 8), 1, 5,4, "Average Latency(ns)")
// #benchmark-chart("Mutex-bound", (1, 2, 4, 8), 1, 10000000,5, "Throughput(task/second)")
// #benchmark-chart("Mutex-bound", (1, 2, 4, 8), 1, 1000,4, "Average Latency(ns)")
// #benchmark-chart("Mutex-bound", (1, 2, 4, 8,16), 1, 100,3,"Total Time(ms)")
// #benchmark-chart("Sleep-bound", (1, 2, 4, 8), 1, 500000,5, "Throughput(task/second)")
// #benchmark-chart("Sleep-bound", (1, 2, 4, 8), 1, 100000,4, "Average Latency(ns)")
// #benchmark-chart("Sleep-bound", (1, 2, 4, 8,16), 1, 10000,3,"Total Time(ms)")

#let benchmark-table() = {
  let data = csv("../part2/results/results.csv")
  let headers = data.at(0)
  let rows = data.slice(1)

  table(
    columns: 6,

    align: (left, center, center, right, right, right),

    fill: (x, y) => if y == 0 { luma(240) } else { none },

    ..headers.map(h => strong(h)),

    ..rows.flatten()
  )
}
#benchmark-table()
