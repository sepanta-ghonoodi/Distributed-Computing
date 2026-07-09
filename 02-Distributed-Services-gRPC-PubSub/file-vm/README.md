
# File Server Service (VM3)

The File Server is a dedicated storage microservice responsible for securely serving static assets (files and images) to the front-facing Web Server. To maximize system performance and architectural integrity, this service bypasses standard HTTP file serving in favor of a high-efficiency **gRPC Server-Side Streaming** model.

##  Architecture & Dependencies

This service operates completely isolated from the user's browser, receiving internal network requests exclusively from the Web Server (VM1).

* **Protocol:** gRPC Server (listening on network port `:50053`).
* **Data Transfer:** Implements chunked binary streaming to prevent memory overloads when serving large files across the virtual network.
* **Storage:** Reads local assets directly from the internal `/files` directory.
* **Shared Contracts:** Imports the shared `pb` (Protocol Buffers) workspace module to resolve the file streaming request and response structures.

### Directory Layout
```text
HW2/
├── pb/                (Required: Shared gRPC interfaces)
└── file-vm/
    ├── main.go        (gRPC streaming server initialization and logic)
    ├── files/         (Local directory containing images/downloads)
    └── README.md

```

## 🚀 How to Run

Like the Auth Server, the File Server's logic is fully contained within its `main.go` file and dictates its own listening port.

1. Ensure your terminal is in the root workspace where the `go.work` file can resolve the `pb` module.
2. Navigate to the `file-vm` directory:

```bash
cd file-vm

```

1. Run the service:

```bash
go run main.go

```

Upon successful startup, the terminal will indicate that the File Service is actively listening for incoming gRPC connections on port `:50053`. *(Note: Ensure you have placed sample files or images inside the `files/` directory before requesting them via the Web VM).*

## 📦 gRPC Service Methods

### `FileSystem`

This service implements the streaming RPC contract defined in your `.proto` file.

* **`rpc DownloadFile(FileRequest) returns (stream FileChunk)`**
* **Input:** Receives a payload containing the requested `filename` and validates the incoming authorization metadata (session token).
* **Process:** Locates the file on the local disk, opens it, and reads it into sequential byte chunks (e.g., 4KB blocks).
* **Output:** Utilizes `stream.Send()` to pipe the raw binary chunks continuously across the network to VM1 until the `EOF` (End of File) is reached.

