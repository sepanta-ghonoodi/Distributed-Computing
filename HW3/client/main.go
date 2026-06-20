package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

type KeyEntry struct {
	Key       string `json:"key"`
	Value     string `json:"value"`
	Version   int    `json:"version"`
	UpdatedBy string `json:"updated_by"`
	Timestamp int64  `json:"timestamp"`
}

func main() {
	op := flag.String("op", "", "Operation to perform (get, put, run)")
	key := flag.String("key", "", "Key for GET/PUT")
	value := flag.String("value", "", "Value for PUT")
	replica := flag.String("replica", "http://localhost:8001", "Replica URL to send request to")
	consistency := flag.String("consistency", "eventual", "Consistency model (eventual or strong)")
	runScenario := flag.String("run", "", "Scenario to run (scenario1, scenario2, scenario3, scenario4, all)")

	flag.Parse()

	if *runScenario != "" {
		runScenarios(*runScenario)
		return
	}

	switch *op {
	case "get":
		if *key == "" {
			log.Fatal("Key must be specified for GET operation")
		}
		entry, err := getRequest(*replica, *key, *consistency)
		if err != nil {
			log.Fatalf("GET failed: %v", err)
		}
		fmt.Printf("GET Success: Key=%s, Value=%s, Version=%d, UpdatedBy=%s, Timestamp=%d\n",
			entry.Key, entry.Value, entry.Version, entry.UpdatedBy, entry.Timestamp)
	case "put":
		if *key == "" || *value == "" {
			log.Fatal("Key and value must be specified for PUT operation")
		}
		entry, err := putRequest(*replica, *key, *value, *consistency)
		if err != nil {
			log.Fatalf("PUT failed: %v", err)
		}
		fmt.Printf("PUT Success: Key=%s, Value=%s, Version=%d, UpdatedBy=%s, Timestamp=%d\n",
			entry.Key, entry.Value, entry.Version, entry.UpdatedBy, entry.Timestamp)
	default:
		fmt.Println("Usage:")
		fmt.Println("  go run main.go -op get -key <key> [-replica <url>] [-consistency <eventual|strong>]")
		fmt.Println("  go run main.go -op put -key <key> -value <value> [-replica <url>] [-consistency <eventual|strong>]")
		fmt.Println("  go run main.go -run <scenario1|scenario2|scenario3|scenario4|all>")
	}
}


func getRequest(replicaURL, key, consistency string) (KeyEntry, error) {
	url := fmt.Sprintf("%s/get?key=%s&consistency=%s", replicaURL, key, consistency)
	resp, err := http.Get(url)
	if err != nil {
		return KeyEntry{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return KeyEntry{}, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var entry KeyEntry
	if err := json.NewDecoder(resp.Body).Decode(&entry); err != nil {
		return KeyEntry{}, err
	}
	return entry, nil
}

func putRequest(replicaURL, key, value, consistency string) (KeyEntry, error) {
	url := fmt.Sprintf("%s/put", replicaURL)
	reqBody, err := json.Marshal(map[string]string{
		"key":         key,
		"value":       value,
		"consistency": consistency,
	})
	if err != nil {
		return KeyEntry{}, err
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return KeyEntry{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return KeyEntry{}, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var entry KeyEntry
	if err := json.NewDecoder(resp.Body).Decode(&entry); err != nil {
		return KeyEntry{}, err
	}
	return entry, nil
}

func setDelay(replicaURL string, delayMs int) error {
	url := fmt.Sprintf("%s/config", replicaURL)
	reqBody, err := json.Marshal(map[string]int{
		"delay_ms": delayMs,
	})
	if err != nil {
		return err
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("config failed with status %d", resp.StatusCode)
	}
	return nil
}

func getStatus(replicaURL string) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/status", replicaURL)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}
	return status, nil
}


type ManagedReplicas struct {
	cmds []*exec.Cmd
}

func StartReplicas() (*ManagedReplicas, error) {
	files, _ := filepath.Glob("data_replica*.json")
	for _, f := range files {
		os.Remove(f)
	}

	configs := []string{
		"../configs/replica1.json",
		"../configs/replica2.json",
		"../configs/replica3.json",
	}

	var cmds []*exec.Cmd
	for _, cfg := range configs {
		cmd := exec.Command("./replica_bin", "-config", cfg)
		logFile, err := os.Create(fmt.Sprintf("replica_%s.log", filepath.Base(cfg)))
		if err == nil {
			cmd.Stdout = logFile
			cmd.Stderr = logFile
		}
		if err := cmd.Start(); err != nil {
			for _, started := range cmds {
				if started.Process != nil {
					started.Process.Kill()
				}
			}
			return nil, fmt.Errorf("failed to start replica with config %s: %v", cfg, err)
		}
		cmds = append(cmds, cmd)
	}

	time.Sleep(1000 * time.Millisecond)
	return &ManagedReplicas{cmds: cmds}, nil
}

func (mr *ManagedReplicas) Shutdown() {
	for _, cmd := range mr.cmds {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	}
}

func (mr *ManagedReplicas) StartReplica(id int) error {
	cfg := fmt.Sprintf("../configs/replica%d.json", id)
	cmd := exec.Command("./replica_bin", "-config", cfg)
	logFile, err := os.Create(fmt.Sprintf("replica_replica%d.json.log", id))
	if err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	mr.cmds = append(mr.cmds, cmd)
	time.Sleep(500 * time.Millisecond)
	return nil
}

func runScenarios(scenario string) {
	switch scenario {
	case "scenario1":
		runScenario1()
	case "scenario2":
		runScenario2()
	case "scenario3":
		runScenario3()
	case "scenario4":
		runScenario4()
	case "all":
		runScenario1()
		runScenario2()
		runScenario3()
		runScenario4()
	default:
		log.Fatalf("Unknown scenario: %s", scenario)
	}
}

func writeResult(filename, content string) {
	path := filepath.Join("../results", filename)
	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		log.Printf("Failed to write result file %s: %v", path, err)
	}
	fmt.Printf("\n--- Results saved to %s ---\n", path)
}

func runScenario1() {
	fmt.Println("Scenario 1: Temporary Inconsistency")
	mr, err := StartReplicas()
	if err != nil {
		log.Fatalf("Failed to start replicas: %v", err)
	}
	defer mr.Shutdown()

	var output bytes.Buffer
	logOut := func(format string, a ...interface{}) {
		str := fmt.Sprintf(format, a...)
		fmt.Print(str)
		output.WriteString(str)
	}

	logOut("Configuring replication delay on replicas to 2000 ms...\n")
	setDelay("http://localhost:8001", 2000)
	setDelay("http://localhost:8002", 2000)
	setDelay("http://localhost:8003", 2000)

	logOut("Performing PUT request (key='x', value='10', consistency='eventual') to Replica 1...\n")
	startPut := time.Now()
	putEntry, err := putRequest("http://localhost:8001", "x", "10", "eventual")
	putLatency := time.Since(startPut)
	if err != nil {
		logOut("ERROR: PUT failed: %v\n", err)
		return
	}
	logOut("	PUT Succeeded. Latency: %v, Local Version: %d\n", putLatency, putEntry.Version)

	logOut("Immediately performing GET request for key 'x' on Replica 2...\n")
	getEntry, err := getRequest("http://localhost:8002", "x", "eventual")
	if err != nil {
		logOut("   GET on Replica 2 returned error (Expected stale read/Not Found): %v\n", err)
	} else {
		logOut("   GET Succeeded (Unexpected, should be stale): value=%s, version=%d\n", getEntry.Value, getEntry.Version)
	}

	logOut("Waiting for 3 seconds (allowing replication delay of 2 seconds to elapse)...\n")
	time.Sleep(3 * time.Second)

	logOut("Performing GET request for key 'x' on Replica 2 again...\n")
	getEntry, err = getRequest("http://localhost:8002", "x", "eventual")
	if err != nil {
		logOut("   GET on Replica 2 failed: %v\n", err)
	} else {
		logOut("   GET Succeeded (Converged!): value=%s, version=%d, updated_by=%s\n", getEntry.Value, getEntry.Version, getEntry.UpdatedBy)
	}
	writeResult("scenario1.txt", output.String())
}

func runScenario2() {
	fmt.Println("Scenario 2: One Replica Failure")

	mr, err := StartReplicas()
	if err != nil {
		log.Fatalf("Failed to start replicas: %v", err)
	}
	defer mr.Shutdown()

	var output bytes.Buffer
	logOut := func(format string, a ...interface{}) {
		str := fmt.Sprintf(format, a...)
		fmt.Print(str)
		output.WriteString(str)
	}

	logOut("Stopping Replica 3...\n")
	if mr.cmds[2].Process != nil {
		mr.cmds[2].Process.Kill()
	}
	time.Sleep(500 * time.Millisecond)

	logOut("Testing Eventual Consistency PUT (key='y', value='eventual_val') on Replica 1...\n")
	startPut := time.Now()
	putEntry, err := putRequest("http://localhost:8001", "y", "eventual_val", "eventual")
	latency := time.Since(startPut)
	if err != nil {
		logOut("   PUT failed: %v\n", err)
	} else {
		logOut("   PUT Succeeded. Latency: %v, Version: %d\n", latency, putEntry.Version)
	}

	logOut("Testing Strong Consistency PUT (key='z', value='strong_val') on Replica 1 (Majority = 2, Active = 2)...\n")
	startPut = time.Now()
	putEntry, err = putRequest("http://localhost:8001", "z", "strong_val", "strong")
	latency = time.Since(startPut)
	if err != nil {
		logOut("   PUT failed: %v\n", err)
	} else {
		logOut("   PUT Succeeded. Latency: %v, Version: %d\n", latency, putEntry.Version)
	}

	logOut("Reading key 'z' from Replica 2 using Strong consistency...\n")
	getEntry, err := getRequest("http://localhost:8002", "z", "strong")
	if err != nil {
		logOut("   Strong GET failed: %v\n", err)
	} else {
		logOut("   Strong GET Succeeded: value=%s, version=%d\n", getEntry.Value, getEntry.Version)
	}

	logOut("Stopping Replica 2 (leaving only Replica 1 active, which is less than majority)...\n")
	if mr.cmds[1].Process != nil {
		mr.cmds[1].Process.Kill()
	}
	time.Sleep(500 * time.Millisecond)

	logOut("Testing Eventual Consistency PUT (key='w', value='eventual_val2') on Replica 1...\n")
	startPut = time.Now()
	putEntry, err = putRequest("http://localhost:8001", "w", "eventual_val2", "eventual")
	latency = time.Since(startPut)
	if err != nil {
		logOut("   PUT failed: %v\n", err)
	} else {
		logOut("   PUT Succeeded (Expected success since local write is always allowed under eventual consistency). Latency: %v\n", latency)
	}

	logOut("Testing Strong Consistency PUT (key='z2', value='strong_val2') on Replica 1...\n")
	startPut = time.Now()
	_, err = putRequest("http://localhost:8001", "z2", "strong_val2", "strong")
	latency = time.Since(startPut)
	if err != nil {
		logOut("   PUT failed (Expected failure since majority is not active): %v\n", err)
	} else {
		logOut("   PUT Succeeded (Unexpected!): Latency: %v\n", latency)
	}



	writeResult("scenario2.txt", output.String())
}

func runScenario3() {
	fmt.Println("Scenario 3: Concurrent Conflict")

	mr, err := StartReplicas()
	if err != nil {
		log.Fatalf("Failed to start replicas: %v", err)
	}
	defer mr.Shutdown()

	var output bytes.Buffer
	logOut := func(format string, a ...interface{}) {
		str := fmt.Sprintf(format, a...)
		fmt.Print(str)
		output.WriteString(str)
	}

	logOut("Configuring replication delay on replicas to 2000 ms to isolate the concurrent writes...\n")
	setDelay("http://localhost:8001", 2000)
	setDelay("http://localhost:8002", 2000)
	setDelay("http://localhost:8003", 2000)

	logOut("Sending concurrent PUT requests to Replica 1 (c='val_from_r1') and Replica 2 (c='val_from_r2')...\n")
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		_, pErr := putRequest("http://localhost:8001", "c", "val_from_r1", "eventual")
		if pErr != nil {
			logOut("   PUT on Replica 1 failed: %v\n", pErr)
		} else {
			logOut("   PUT on Replica 1 completed.\n")
		}
	}()

	go func() {
		defer wg.Done()
		time.Sleep(50 * time.Millisecond)
		_, pErr := putRequest("http://localhost:8002", "c", "val_from_r2", "eventual")
		if pErr != nil {
			logOut("   PUT on Replica 2 failed: %v\n", pErr)
		} else {
			logOut("   PUT on Replica 2 completed.\n")
		}
	}()

	wg.Wait()

	logOut("Immediately reading value of 'c' from all 3 replicas before replication completes...\n")
	r1Val, _ := getRequest("http://localhost:8001", "c", "eventual")
	r2Val, _ := getRequest("http://localhost:8002", "c", "eventual")
	r3Val, _ := getRequest("http://localhost:8003", "c", "eventual")
	logOut("   Immediate GET results:\n")
	logOut("   Replica 1: value=%s, version=%d, updated_by=%s\n", r1Val.Value, r1Val.Version, r1Val.UpdatedBy)
	logOut("   Replica 2: value=%s, version=%d, updated_by=%s\n", r2Val.Value, r2Val.Version, r2Val.UpdatedBy)
	logOut("   Replica 3: value=%s, version=%d, updated_by=%s (Not replicated yet)\n", r3Val.Value, r3Val.Version, r3Val.UpdatedBy)

	logOut("Waiting for 3 seconds to let replication messages exchange and settle...\n")
	time.Sleep(3 * time.Second)

	logOut("Querying final values from all 3 replicas...\n")
	r1Final, _ := getRequest("http://localhost:8001", "c", "eventual")
	r2Final, _ := getRequest("http://localhost:8002", "c", "eventual")
	r3Final, _ := getRequest("http://localhost:8003", "c", "eventual")
	logOut("   Final GET results:\n")
	logOut("   Replica 1: value=%s, version=%d, updated_by=%s\n", r1Final.Value, r1Final.Version, r1Final.UpdatedBy)
	logOut("   Replica 2: value=%s, version=%d, updated_by=%s\n", r2Final.Value, r2Final.Version, r2Final.UpdatedBy)
	logOut("   Replica 3: value=%s, version=%d, updated_by=%s\n", r3Final.Value, r3Final.Version, r3Final.UpdatedBy)
	writeResult("scenario3.txt", output.String())
}

func runScenario4() {
	fmt.Println("Scenario 4: Network Delay Impact")

	var output bytes.Buffer
	logOut := func(format string, a ...interface{}) {
		str := fmt.Sprintf(format, a...)
		fmt.Print(str)
		output.WriteString(str)
	}

	delays := []int{0, 500, 2000}

	type MetricRow struct {
		Delay           int
		Model           string
		PutLatencyMs    int64
		GetLatencyMs    int64
		ConvergenceMs   int64
		StaleReadsCount int
	}

	var rows []MetricRow

	for _, delay := range delays {
		logOut("Testing Network Delay: %d ms\n", delay)

		mr, err := StartReplicas()
		if err != nil {
			log.Fatalf("Failed to start replicas: %v", err)
		}
		setDelay("http://localhost:8001", delay)
		setDelay("http://localhost:8002", delay)
		setDelay("http://localhost:8003", delay)

		key := fmt.Sprintf("k_eventual_%d", delay)
		startPut := time.Now()
		_, err = putRequest("http://localhost:8001", key, "v", "eventual")
		putLatency := time.Since(startPut).Milliseconds()
		startPoll := time.Now()
		staleReads := 0
		var r2Val, r3Val string

		for {
			e2, _ := getRequest("http://localhost:8002", key, "eventual")
			e3, _ := getRequest("http://localhost:8003", key, "eventual")

			r2Val = e2.Value
			r3Val = e3.Value

			if r2Val == "v" && r3Val == "v" {
				break
			}
			staleReads++
			time.Sleep(50 * time.Millisecond)

			if time.Since(startPoll) > 5*time.Second {
				logOut("   WARNING: Convergence timeout for eventual consistency at %d ms delay\n", delay)
				break
			}
		}
		convergenceTime := time.Since(startPoll).Milliseconds()

		startGet := time.Now()
		_, _ = getRequest("http://localhost:8001", key, "eventual")
		getLatency := time.Since(startGet).Milliseconds()

		rows = append(rows, MetricRow{
			Delay:           delay,
			Model:           "Eventual",
			PutLatencyMs:    putLatency,
			GetLatencyMs:    getLatency,
			ConvergenceMs:   convergenceTime,
			StaleReadsCount: staleReads,
		})

		logOut("   Eventual Model - PUT Latency: %d ms, GET Latency: %d ms, Convergence: %d ms, Stale Reads: %d\n",
			putLatency, getLatency, convergenceTime, staleReads)
		mr.Shutdown()

		mr, err = StartReplicas()
		if err != nil {
			log.Fatalf("Failed to start replicas: %v", err)
		}
		setDelay("http://localhost:8001", delay)
		setDelay("http://localhost:8002", delay)
		setDelay("http://localhost:8003", delay)

		keyStrong := fmt.Sprintf("k_strong_%d", delay)
		startPut = time.Now()
		_, err = putRequest("http://localhost:8001", keyStrong, "v_strong", "strong")
		putLatencyStrong := time.Since(startPut).Milliseconds()

		startGet = time.Now()
		_, err = getRequest("http://localhost:8001", keyStrong, "strong")
		getLatencyStrong := time.Since(startGet).Milliseconds()

		e2, err2 := getRequest("http://localhost:8002", keyStrong, "strong")
		e3, err3 := getRequest("http://localhost:8003", keyStrong, "strong")

		staleReadsStrong := 0
		if err2 != nil || e2.Value != "v_strong" {
			staleReadsStrong++
		}
		if err3 != nil || e3.Value != "v_strong" {
			staleReadsStrong++
		}

		rows = append(rows, MetricRow{
			Delay:           delay,
			Model:           "Strong",
			PutLatencyMs:    putLatencyStrong,
			GetLatencyMs:    getLatencyStrong,
			ConvergenceMs:   0,
			StaleReadsCount: staleReadsStrong,
		})

		logOut("   Strong Model   - PUT Latency: %d ms, GET Latency: %d ms, Convergence: 0 ms, Stale Reads: %d\n",
			putLatencyStrong, getLatencyStrong, staleReadsStrong)
		mr.Shutdown()
	}

	tableStr := "\nMETRICS SUMMARY TABLE\n"
	tableStr += fmt.Sprintf("%-12s | %-12s | %-16s | %-16s | %-18s | %-12s\n", "Scenario/Dly", "Model", "PUT Latency (ms)", "GET Latency (ms)", "Convergence (ms)", "Stale Reads")
	tableStr += "---------------------------------------------------------------------------------------------------------\n"
	for _, r := range rows {
		tableStr += fmt.Sprintf("%-12d | %-12s | %-16d | %-16d | %-18d | %-12d\n",
			r.Delay, r.Model, r.PutLatencyMs, r.GetLatencyMs, r.ConvergenceMs, r.StaleReadsCount)
	}
	logOut(tableStr)



	writeResult("scenario4.txt", output.String())
}
