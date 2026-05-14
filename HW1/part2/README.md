# Concurrency, Scheduling, and Context Switching Analysis

This directory contains a Go application designed to simulate, benchmark, and analyze the effects of concurrency, OS-level scheduling, and context switching across different workload types.

## 1. Features & Architecture

The application runs benchmarks across three distinct workload scenarios to measure the performance impact of goroutine counts and CPU core utilization (`GOMAXPROCS`):

* **CPU-bound:** Pure mathematical calculations simulating heavy processing without blocking.
* **Mutex-bound (Mixed):** Mathematical calculations combined with a shared resource lock (`sync.Mutex`), simulating heavy thread contention.
* **Sleep-bound (Mixed):** Mathematical calculations combined with `time.Sleep`, simulating I/O-bound waits and forced context switches.

The benchmark tracks the following metrics for every combination of `GOMAXPROCS` and Goroutine count:

* Total execution time
* Average task latency
* Overall throughput (Operations per second)

Execution traces are automatically generated using Go's standard `runtime/trace` package for deeper visual analysis.

## 2. Dependencies and Environment

* **Language:** Go 1.22
* **Libraries:** standard library. No external dependencies are required.
* **Environment:** Linux.

## 3. Execution Instructions

**Run the Benchmark Suite:**

```bash
go run main.go
```

*Note: The benchmark will take several minutes to complete as it iterates through all combinations of `GOMAXPROCS` (1, 2, 4, 8, NumCPU) and Goroutine counts (1, 2, 4, 8, 16, 32, 64) for all three workloads.*

**Outputs:**
Upon completion, the application generates the following artifacts in the same directory:

1. `results.csv`: A complete dataset of all calculated metrics (Time, Latency, Throughput) for analysis and graphing.
2. `trace_CPU-bound.out`: The execution trace data for the CPU-bound scenario.
3. `trace_Mutex-bound.out`: The execution trace data for the Mutex-bound scenario.
4. `trace_Sleep-bound.out`: The execution trace data for the Sleep-bound scenario.

## 4. Visualizing Traces

To view and analyze the generated trace files, use the built-in Go trace tool.

Run the following command for the specific workload you wish to inspect (e.g., `CPU-bound`):

```bash
go tool trace trace_CPU-bound.out
```

This command will parse the trace file and automatically open a local web server.

**Note on Trace Viewing:**

* You must have Google Chrome installed to view the trace UI properly.
