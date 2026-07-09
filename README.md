# Distributed Systems

This repository contains the implementation of homework assignments for the **Distributed Systems** course at the University of Tehran. The projects are implemented in **Go** and explore fundamental concepts of distributed architecture, concurrency, IPC, inter-service communication (gRPC, Pub/Sub), virtualization, containerization, and replication consistency models.

---

## 📂 Project Structure

The repository is organized into three main assignments:

### 1. 01-Concurrency-IPC-and-VMs
*   **Part 1 (IPC via Named Pipes):** An asynchronous worker-client setup communicating via POSIX Linux Named Pipes (FIFOs) for standard arithmetic processing.
*   **Part 2 (Concurrency & Scheduling):** Performance analysis of goroutine scheduling under CPU, Mutex, and Sleep workloads across varying `GOMAXPROCS` values. Generates CSV results and Go execution traces (`runtime/trace`).
*   **Part 3 (VM Service Containerization):** Dockerization of a Go HTTP service run inside a Guest Virtual Machine, configured via VirtualBox NAT port-forwarding to handle Host client requests.

### 2. 02-Distributed-Services-gRPC-PubSub
*   **Multi-VM Setup:** Deployment of services across three different virtual machines.
*   **gRPC Integration:** Implements user authentication and large file streaming services using Protocol Buffers and gRPC.
*   **HTTP Pub/Sub Telemetry:** A decoupled publisher-subscriber system for real-time memory monitoring of the web VM.

### 3. 03-Replicated-Key-Value-Store
*   **Decentralized Replicas:** Three independent replica servers storing data persistently in JSON databases.
*   **Consistency Levels:** Supports both eventual consistency (asynchronous replication) and simplified strong consistency (quorum reads/writes using 2-out-of-3 majority confirmation).
*   **Conflict Resolution:** Uses versioning, timestamping, and deterministic Last-Write-Wins (LWW) to resolve replication conflicts.

---

## 🛠 Prerequisites

*   **Go:** Version 1.22+
*   **Docker:** For containerized assignments
*   **VirtualBox / VM Manager:** For multi-VM networking exercises
*   **Linux / POSIX-compliant Environment:** Required for Named Pipes (`mkfifo`) and scheduling tools.

---

