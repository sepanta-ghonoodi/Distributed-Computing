# Replicated Key-Value Store - Client CLI

This folder contains the Go implementation of the client CLI. The client can run manual commands (GET/PUT) against any running replica or execute automated test scenarios.

## Features
- **GET / PUT Operations**: Command line flags to send standard Key-Value operations to any replica.
- **Support for Consistency Models**: Operations can specify `-consistency eventual` or `-consistency strong`.
- **Automated Scenario Runner**: Runs all homework scenarios, measures metrics (PUT/GET latency, convergence time, stale read counts), and prints summaries.
- **Local Replica Orchestration**: Scenarios dynamically build, start, configure, and shutdown the replicas programmatically to ensure a clean slate and clean measurements.

## Commands

### Automated Scenarios
To run an automated scenario and save the output directly to the `results/` folder:
```bash
# Run Scenario 1 (Temporary Inconsistency)
go run main.go -run scenario1

# Run Scenario 2 (One Replica Failure)
go run main.go -run scenario2

# Run Scenario 3 (Concurrent Conflict & LWW Resolution)
go run main.go -run scenario3

# Run Scenario 4 (Network Delay Analysis & Metrics Table)
go run main.go -run scenario4

# Run all scenarios sequentially
go run main.go -run all
```

### Manual CLI Operations
First, ensure you have running replicas (e.g., using config files). Then run:

- **PUT Operation**:
  ```bash
  go run main.go -op put -key x -value 42 -replica http://localhost:8001 -consistency strong
  ```
- **GET Operation**:
  ```bash
  go run main.go -op get -key x -replica http://localhost:8002 -consistency strong
  ```
