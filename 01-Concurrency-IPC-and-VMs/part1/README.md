# Part 1: Inter-Process Communication via Named Pipes

This project implements a robust Inter-Process Communication (IPC) system using Go. It consists of two independent processes (`Interface` and `Worker`) that communicate asynchronously using Linux Named Pipes (FIFO).

## File Structure

*   `worker.go`: The background service (daemon) that continuously listens for incoming requests. It parses the commands, performs the arithmetic calculations, and returns a structured JSON response. It is also responsible for initializing the OS-level named pipes automatically.
*   `interface.go`: The client-facing process. It reads standard input from the user, sends the raw command through the request pipe, and waits to read and display the response from the worker.

## 🛠 Dependencies

*   **Go Environment:** Go 1.18 or higher.
*   **Operating System:** Linux or WSL (Windows Subsystem for Linux). This is strictly required because the project utilizes POSIX-compliant named pipes via the `syscall` package (`mkfifo`), which are not natively supported on Windows filesystems.
*   **External Libraries:** None. The project strictly uses the Go Standard Library (`os`, `syscall`, `encoding/json`, `bufio`, etc.) as per the assignment rules.

## How to Run

You need to run the processes in two separate terminal sessions.

**Step 1: Start the Worker**
Open the first terminal and execute the worker. It will automatically create the required `req_pipe` and `res_pipe` files if they do not exist, and then wait for connections.
```bash
go run worker.go