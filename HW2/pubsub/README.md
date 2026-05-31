# Distributed System Project

This repository contains the implementation of our distributed system. It includes gRPC for authentication and file streaming, and a simple HTTP-based Pub/Sub mechanism for memory monitoring.

## A Quick Note on the `pubsub` Folder Structure
The assignment document suggested putting `publisher.go` and `subscriber.go` right next to each other inside the `pubsub/` directory. However, we ran into a strict rule in Go: you can't have multiple packages in the same folder.

Since `publisher.go` acts as a module for the web server (using `package pubsub`) and `subscriber.go` is a standalone executable (using `package main`), the Go compiler threw an error. To fix this issue without breaking the overall architecture requested in the project, I just moved the subscriber into its own sub-folder (`pubsub/subscriber/`). This keeps the code clean and makes the compiler happy.

---

## 1. How to Run the Services
To run the whole system, open separate terminals (or tmux panes) and run these commands so each service binds to its specific network port:

**Auth Server (VM2):**
```bash
cd auth-vm
go run main.go