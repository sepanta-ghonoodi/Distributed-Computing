
== Architecture and Design Overview
In this section, we designed and implemented a robust Inter-Process Communication (IPC) system utilizing two independent Go processes: the `Interface` and the `Worker`. The architecture strictly follows a decoupled Request-Response model. The `Interface` acts as a lightweight, user-facing client responsible for capturing standard input and displaying the final results. Conversely, the `Worker` functions as a resilient background daemon. It continuously listens for incoming byte streams, performs floating-point mathematical operations, and returns structured data without terminating after a single execution.

== IPC Mechanism & Auto-Initialization
The core communication bridge was established using Linux Named Pipes (FIFOs). Instead of relying on manual user setup via terminal commands, we engineered the system to be entirely self-contained by interacting directly with the Linux kernel. Upon initialization, the Worker process utilizes Go’s `syscall.Mkfifo` to automatically generate the `req_pipe` and `res_pipe` files with `0666` (read/write) permissions.

To manage process synchronization efficiently, we leveraged the native blocking behavior of POSIX pipes. By opening the pipes with `os.O_RDONLY`, the Worker naturally suspends its execution (blocks) at the operating system level until the Interface opens the opposite end of the pipe for writing. This eliminates the need for aggressive CPU polling and ensures optimal resource management.

== Communication Protocol & Advanced Serialization
The communication protocol was explicitly defined to maintain consistency and fault tolerance across process boundaries:

*Request Format:* The Interface parses user input and sends raw text streams in a strict `OP A B` format (e.g., `ADD 5.5 4.5`). The Worker intercepts these streams using a buffered I/O scanner (`bufio.NewScanner`) to safely process data line-by-line.

*Response Format:* To ensure a highly structured and scalable design, the Worker serializes the computational results into a JSON payload before returning them through the response pipe.

A significant architectural decision in our implementation was the use of *pointer-based JSON marshalling* (`*float64`). In Go, unassigned numerical values default to zero. By assigning a pointer to the result field combined with the `omitempty` struct tag, our system intelligently differentiates between an actual calculated zero (e.g., `SUB 5 5`), which is successfully serialized as `{"status":"OK","result":0}`, and an error state, where the result field is entirely omitted from the JSON payload.

== Error Handling and Resilience
The system is fortified with comprehensive error handling to prevent unexpected panics and crashes. It natively intercepts computational violations (e.g., Division by Zero), structural inconsistencies (insufficient arguments), and type-casting failures (non-numeric string inputs), returning specific JSON error messages for each scenario. Furthermore, to prevent File Descriptor (FD) leaks during unexpected disconnects, Go’s `defer` statements were strategically implemented to guarantee that all active pipes are safely closed before the function returns.