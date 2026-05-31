# Web Server Service (VM1)

The Web Server serves as the primary gateway and user interface for the distributed system. It is responsible for serving HTML templates, handling incoming HTTP traffic, and orchestrating backend gRPC calls to the Auth and File servers.

Additionally, this service operates as a **Publisher** in the system's Pub/Sub architecture, running a background goroutine to monitor its own memory heap and broadcast alerts if usage exceeds safe thresholds.

## Architecture & Dependencies

This service is entirely stateless and relies on a shared Go workspace (`go.work`) environment to resolve data structures.

* **Frontend:** Go `html/template` (served on port `:8080`).
* **Backend RPC:** gRPC clients configured to dial external VMs for authentication and file streaming.
* **Shared Data:** Imports the `aux` workspace module for the `MemoryEvent` struct contract.
* **Memory Publisher:** Runs concurrently alongside the HTTP server, monitoring `runtime.MemStats`.

### Directory Layout
```text
HW2/
├── aux/               (Required: Shared event structures)
└── web-vm/
    ├── main.go        (HTTP routing, gRPC clients, CLI flag parsing)
    ├── publisher.go   (Background memory monitoring loop)
    ├── templates/     (HTML views)
    └── README.md

```

## 🚀 How to Run

Because the core execution logic is split between `main.go` and `publisher.go` within the `main` package, **you cannot start the server by targeting a single file**.

You must compile the entire directory and pass the network addresses of your Auth (VM2) and File (VM3) servers as command-line arguments.

1. Ensure VM2 and VM3 are actively running and listening on their respective IP addresses.
2. Open your terminal and navigate to the `web-vm` directory:

```bash
cd web-vm

```

1. Run the service by compiling the current directory (`.`) and passing your specific network flags:

```bash
go run . -auth="AUTH_VM_IP:50052" -file="FILE_VM_IP:50053"

```

*(Note: Replace the IP addresses above with the actual assigned addresses of your virtual machines).*

## 🌐 Application Endpoints

### 1. User Interface & Authentication

* **`GET /login`**: Renders the login form.
* **`POST /login`**: Captures user credentials, dials the Auth Service (VM2) via gRPC, and validates the login. On success, sets a secure HTTP-only `session_token` cookie and redirects the user.
* **`GET /dashboard`**: Protected route. Validates the presence of the session cookie before rendering the file dashboard. Unauthenticated requests are rejected and redirected to `/login`.

### 2. File Operations

* **`GET /download?file={filename}`**: Initiates a chunked gRPC download stream from the File Service (VM3). Intercepts the binary chunks over the virtual network and pipes them directly into the user's browser download stream.

### 3. System Testing (Pub/Sub)

* **`GET /consume-memory?mb={amount}`**: A diagnostic endpoint designed to artificially inflate the heap allocation of the web service for testing purposes.
* **Example:** `curl http://localhost:8080/consume-memory?mb=100`
* **Behavior:** Once the total memory allocation exceeds **300 MB**, the background publisher loop (running in `publisher.go`) instantly broadcasts a `HIGH_MEMORY_USAGE` JSON payload to the listening subscriber service over the network.


