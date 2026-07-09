#import "plots.typ": benchmark-chart, benchmark-table
== CPU-Bound task:
In this section, we analyze a purely CPU-bound task that calculates a mathematical sum. The total computational workload remains constant across all tests, but it is divided into equal sub-parts and distributed among the active goroutines. Below, we present the total execution time, throughput, and average latency for each configuration.
#figure(
  grid(
    columns: 3,
    gutter: 4em,
    align: center,

    benchmark-chart("CPU-bound", (1, 2, 4, 8, 16), 1, 1000000000, 5, "Throughput(task/second)"),
    benchmark-chart("CPU-bound", (1, 2, 4, 8, 16), 1, 5, 4, "Average Latency(ns)", show-legend: true),
    benchmark-chart("CPU-bound", (1, 2, 4, 8, 16), 1, 100, 3, "Total Time(ms)"),
  ),
  caption: [Performance benchmarks for CPU-bound tasks],
)
This workload involves neither I/O waits (`sleep`) nor synchronization blocks (`mutex`). Consequently, when we increase the number of goroutines on a single physical core (GOMAXPROCS=1), we observe no improvement in throughput or total execution time. Instead, the average latency per task increases drastically. Because all goroutines are initialized simultaneously but forced to time-slice on a single core, they spend the majority of their lifecycle waiting in the scheduler's queue. While the absolute total execution time remains flat, the active lifespan of each individual worker is stretched, negatively impacting the average latency metric.

Furthermore, we observe that increasing the number of physical cores (GOMAXPROCS > 1) yields no performance boost if the number of goroutines remains lower than the available cores (e.g., 4 cores but 1 goroutine). In this state, the workload is not fully parallelized, leaving the additional physical CPU cores idle.

However, as the plots demonstrate, when multiple cores are available, increasing the number of goroutines creates a massive performance boost in both total time and throughput. This scaling continues optimally until the number of goroutines matches the number of physical cores. Beyond this peak—when goroutines exceed the available physical cores—we no longer see improvements. This proves that for purely computational workloads, adding more goroutines does not equate to better parallelization once hardware resources are fully saturated; it only reintroduces context-switching overhead.

Finally, we can see that even under optimal parallelization (adding both cores and goroutines), the average latency still trends upward. This occurs because parallelization improves overall throughput, but it does not reduce the execution time of an individual worker's batch. Since each worker measures its latency from the start of the application to its respective finish line, adding more concurrent workers inherently increases the average measured turnaround time of the tasks.

#pagebreak()

== Mutex Task
In this section, we examine a task constrained by a Mutex, where multiple goroutines attempt to increment a shared variable within a critical section.

When limited to a single physical core (GOMAXPROCS=1), increasing the number of goroutines does not trigger a catastrophic drop in performance. Because only one operating system thread executes at any given microsecond, the lock is acquired and released sequentially. Consequently, there is no true hardware-level contention for the memory address, keeping the throughput relatively stable.

However, when we increase the number of physical cores alongside the number of goroutines, we observe a severe degradation in performance. In a multi-core environment, parallel CPU cores actively compete to acquire the exact same lock to mutate the shared variable. This introduces massive hardware-level contention, as the cores constantly invalidate each other's caches.

Furthermore, this hardware contention compounds with software-level overhead. As multiple goroutines fail to acquire the lock, the Go scheduler is forced to spend a significant portion of its total execution time managing the traffic—spinning, parking (putting to sleep), and subsequently waking up blocked goroutines.

Therefore, for highly contended workloads, throwing more hardware resources (cores) and software concurrency (goroutines) at the problem does not improve throughput. Instead, the compounding overhead of cache contention and scheduler thrashing leads to a drastic loss of overall performance.
#figure(
  grid(
    columns: 3,
    gutter: 4em,
    align: center,
    benchmark-chart("Mutex-bound", (1, 2, 4, 8, 16), 1, 10000000, 5, "Throughput(task/second)"),
    benchmark-chart("Mutex-bound", (1, 2, 4, 8, 16), 1, 1000, 4, "Average Latency(ns)", show-legend: true),
    benchmark-chart("Mutex-bound", (1, 2, 4, 8, 16), 1, 20, 3, "Total Time(ms)"),
  ),
  caption: [Performance benchmarks for Mutex-bound tasks],
)
#pagebreak()


== Sleep Task

In this final section, we introduce a time.Sleep element to the workload, simulating an I/O-bound task such as waiting for a network response or a database query.

When executing this workload with only a single goroutine, performance is exceptionally poor. Because the execution is entirely sequential, the single worker must initiate a task, enter a sleep state, and wait for the entire duration to pass before moving on to the next operation. The CPU remains completely idle while the thread is blocked, resulting in the longest total execution time and the lowest possible throughput.

However, by increasing the number of goroutines, we observe a massive improvement in performance, even when constrained to a single physical core (GOMAXPROCS=1). This demonstrates the core advantage of Go's concurrency model for I/O-bound tasks: latency hiding. When a goroutine enters a sleep state, it voluntarily yields the CPU. The Go scheduler immediately context-switches, placing a different, ready goroutine onto the core. By rapidly interleaving thousands of sleeping tasks, the scheduler ensures the physical CPU is never left waiting idly.

As demonstrated by the data, assigning more physical cores (e.g., GOMAXPROCS=16) to this highly concurrent workload consistently improves performance. Because the computational work executed between sleep cycles is microscopically small, having additional physical cores allows the Go scheduler to process the rapid waking and sleeping of thousands of goroutines in true parallel. This prevents a single OS thread from becoming a bottleneck when juggling thousands of context switches.
#figure(
  grid(
    columns: 3,
    gutter: 4em,
    align: center,
    benchmark-chart("Sleep-bound", (1, 2, 4, 8, 16), 1, 500000, 5, "Throughput(task/second)"),
    benchmark-chart("Sleep-bound", (1, 2, 4, 8, 16), 1, 100000, 4, "Average Latency(ns)", show-legend: true),
    benchmark-chart("Sleep-bound", (1, 2, 4, 8, 16), 1, 10000, 3, "Total Time(ms)"),
  ),
  caption: "Performance benchmarks for Sleep-bound tasks",
)
== Table of Results
Here is the raw results of the code in teh format of a table for each configuration.
#benchmark-table()
\
== Final Conclusions:

==== \1. What happens to the execution time as the number of goroutines increases?\

CPU-Bound:\ On a single core, increasing goroutines does not decrease execution time; it actually inflates the average latency per task because the scheduler must time-slice the workload, forcing tasks to wait in a queue. On multiple cores, execution time decreases linearly until the number of goroutines equals the number of physical cores.

Mutex-Bound:\ Increasing goroutines drastically increases the execution time. The overhead of context switching, lock thrashing, and parking/waking blocked goroutines dominates the CPU time, slowing down the actual computation.

Sleep-Bound:\ Execution time decreases significantly as goroutines increase, because the Go scheduler can effectively hide the latency of sleeping tasks by continuously swapping in ready goroutines.

==== \2. Does increasing the number of goroutines always increase throughput?\
No, increasing goroutines only increases throughput when there are idle hardware resources or when tasks are waiting on I/O.

In purely computational tasks limited to one core, adding goroutines slightly decreases throughput due to context-switching overhead.

In highly contended scenarios (Mutex locks), adding goroutines destroys throughput, as the system spends more time managing access to the lock than executing the critical section.

==== \3. What is the difference in program behavior between GOMAXPROCS=1 and GOMAXPROCS=NumCPU?\
GOMAXPROCS=1 (Concurrency without Parallelism): Forces the Go runtime to execute all goroutines sequentially on a single operating system thread. This eliminates hardware-level race conditions and cache invalidation issues, making heavily contended tasks (like a Mutex lock) run surprisingly fast, but limits computational scaling.

GOMAXPROCS=NumCPU (True Parallelism): Spreads goroutines across all physical CPU cores. This provides massive speedups for parallelizable, independent computations (CPU-bound) and I/O tasks. However, it exposes shared states to severe hardware-level contention, causing cores to constantly invalidate each other's caches when fighting for the same memory address.

==== \4. In which type of workload are the effects of scheduling and context switching seen the most?
\

Positive Effect (Sleep-Bound): The scheduler's ability to instantly context-switch when a goroutine enters a sleep state is what allows a single core to process thousands of concurrent tasks efficiently.

Negative Effect (Mutex-Bound): The scheduler is forced to rapidly switch, spin, park, and wake goroutines that are fighting over a locked resource, resulting in severe "scheduler thrashing" that degrades performance.

==== \5. From what point does increasing concurrency become useless or even decrease performance?
Increasing concurrency(go routines) degrades performance at two distinct thresholds:

- Hardware Saturation: For purely independent, CPU-bound tasks, concurrency becomes useless the moment the number of goroutines exceeds the number of physical cores (GOMAXPROCS). Any additional goroutines simply add time-slicing overhead without adding processing power.

- State Contention: For tasks sharing memory, concurrency decreases performance the moment multiple active threads attempt to mutate the exact same state simultaneously. As the ratio of "waiting goroutines" to "active locks" increases, the system falls into lock thrashing, drastically reducing overall efficiency.
