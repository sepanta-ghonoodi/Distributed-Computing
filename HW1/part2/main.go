package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/trace"
	"strconv"
	"sync"
	"text/tabwriter"
	"time"
)

var (
	sharedMutex   sync.Mutex
	sharedCounter int
)

type Metrics struct {
	Workload   string
	MaxProcs   int
	Goroutines int
	TotalTime  time.Duration
	AvgLatency time.Duration
	Throughput float64
}

type TaskFunc func(workerID, totalWorkers, limit int, result *int)

func cpuBoundTask(workerID, totalWorkers, limit int, result *int) {
	workPerWorker := limit / totalWorkers
	if workerID == totalWorkers-1 {
		workPerWorker += limit % totalWorkers
	}

	sum := 0
	for i := 0; i < workPerWorker; i++ {
		sum += (i * 3) % 7
	}

	*result = sum
}

func mutexTask(workerID, totalWorkers, limit int, result *int) {
	workPerWorker := limit / totalWorkers
	if workerID == totalWorkers-1 {
		workPerWorker += limit % totalWorkers
	}

	localCount := 0

	for i := 0; i < workPerWorker; i++ {
		sharedMutex.Lock()
		sharedCounter++
		sharedMutex.Unlock()

		localCount++
	}

	*result = localCount
}

func sleepTask(workerID, totalWorkers, limit int, result *int) {
	workPerWorker := limit / totalWorkers
	if workerID == totalWorkers-1 {
		workPerWorker += limit % totalWorkers
	}

	localCount := 0

	for i := 0; i < workPerWorker; i++ {
		time.Sleep(5 * time.Microsecond)

		localCount++
	}

	*result = localCount
}

func setupTracing(filename string) func() {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatalf("failed to create trace output file: %v", err)
	}

	if err := trace.Start(f); err != nil {
		log.Fatalf("failed to start trace: %v", err)
	}

	return func() {
		trace.Stop()
		f.Close()
	}
}

func runBenchmark(name string, task TaskFunc, procs, counts int, globalLimit int) Metrics {
	runtime.GOMAXPROCS(procs)

	var mainWg sync.WaitGroup
	var mu sync.Mutex
	var totalLatency time.Duration

	results := make([]int, counts)

	startTotal := time.Now()

	for i := 0; i < counts; i++ {
		mainWg.Add(1)

		workerStart := time.Now()
		go func(workerID int) {
			defer mainWg.Done()
			ctx := context.Background()
			trace.WithRegion(ctx, fmt.Sprintf("%s-P%d-G%d", name, procs, counts), func() {

				// Execute the specific workload
				task(workerID, counts, globalLimit, &results[workerID])

				latency := time.Since(workerStart)

				mu.Lock()
				totalLatency += latency
				mu.Unlock()
			})
		}(i)
	}

	mainWg.Wait()
	totalTime := time.Since(startTotal)

	totalOperations := float64(globalLimit)

	return Metrics{
		Workload:   name,
		MaxProcs:   procs,
		Goroutines: counts,
		TotalTime:  totalTime,
		AvgLatency: totalLatency / time.Duration(globalLimit),
		Throughput: totalOperations / totalTime.Seconds(),
	}
}

func main() {


	goroutineCounts := []int{1, 2, 4, 8, 16, 32, 64}
	maxProcsValues := []int{1, 2, 4, 8, runtime.NumCPU()}

	scenarios := []struct {
		name  string
		task  TaskFunc
		limit int
	}{
		{"CPU-bound", cpuBoundTask, 500_000_000},
		{"Mutex-bound", mutexTask, 10_000_00},
		{"Sleep-bound", sleepTask, 50_000},
	}

	// 2. Execution Phase
	var allResults []Metrics
	fmt.Println("Running benchmarks... Please wait.")

	for _, s := range scenarios {
		traceFilename := fmt.Sprintf("./results/trace_%s.out", s.name)
		cleanup := setupTracing(traceFilename)
		for _, procs := range maxProcsValues {
			for _, counts := range goroutineCounts {
				m := runBenchmark(s.name, s.task, procs, counts, s.limit)
				allResults = append(allResults, m)

				sharedCounter = 0
				runtime.GC()
				time.Sleep(50 * time.Millisecond)
			}
		}
		cleanup()
		log.Printf("Finished scenario: %s, Trace saved to: %s\n", s.name, traceFilename)
	}

	exportToCSV(allResults, "results/results.csv")
}

func printTerminalReport(results []Metrics) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', tabwriter.AlignRight|tabwriter.Debug)
	defer w.Flush()

	fmt.Fprintln(w, "\nWorkload\tGOMAXPROCS\tGoroutines\tTotal Time\tAvg Latency\tThroughput\t")
	fmt.Fprintln(w, "--------\t----------\t----------\t----------\t-----------\t----------\t")

	for _, m := range results {
		fmt.Fprintf(w, "%s\t%d\t%d\t%v\t%v\t%.2f\t\n",
			m.Workload, m.MaxProcs, m.Goroutines, m.TotalTime, m.AvgLatency, m.Throughput)
	}
}

func exportToCSV(results []Metrics, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatalf("failed to create csv file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Workload", "GOMAXPROCS", "Goroutines", "TotalTime_ms", "AvgLatency_ns", "Throughput"})

	for _, m := range results {
		totalTimeMs := float64(m.TotalTime.Nanoseconds()) / 1e6
		avgLatencyNs := m.AvgLatency.Nanoseconds()

		writer.Write([]string{
			m.Workload,
			strconv.Itoa(m.MaxProcs),
			strconv.Itoa(m.Goroutines),
			fmt.Sprintf("%.3f", totalTimeMs),
			strconv.FormatInt(avgLatencyNs, 10),
			fmt.Sprintf("%.2f", m.Throughput),
		})
	}
	fmt.Printf("\nData successfully exported to '%s'\n", filename)
}