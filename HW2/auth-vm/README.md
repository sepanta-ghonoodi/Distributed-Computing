

# Auth Server Service (VM2)

The Auth Server is a backend microservice responsible for handling all user authentication logic. To maintain strict security boundaries in the distributed system, it operates completely isolated from the front-facing Web Server. It receives remote procedure calls (RPCs), validates credentials against a local JSON database, and securely returns the authentication status.

## Architecture & Dependencies

This service operates as a dedicated gRPC server and relies on your shared workspace environment to resolve its API contracts.

* **Protocol:** gRPC Server (listening on port `:50052`).
* **Data Storage:** Reads user credentials locally from `users.json`. (The Web VM has no direct access to this file).
* **Shared Contracts:** Imports the shared `pb` (Protocol Buffers) workspace module to understand the `LoginRequest` and `LoginResponse` data structures.

### Directory Layout
```text
HW2/
├── pb/                (Required: Shared gRPC interfaces)
└── auth-vm/
    ├── main.go        (gRPC server initialization and logic)
    ├── users.json     (Local user database)
    └── README.md
```

## 🚀 How to Run

Unlike the Web Server, the Auth Server's logic is fully contained within a single `main.go` file. It does not require any external command-line flags because it dictates its own listening port.

1. Ensure your terminal is in the root workspace where the `go.work` file can resolve the `pb` module.
2. Navigate to the `auth-vm` directory:

```bash
cd auth-vm

```

1. Run the service:

```bash
go run main.go

```

Upon successful startup, the terminal will indicate that the Auth Service is actively listening for incoming network traffic on port `:50052`. *(Note: This service must be running before you attempt to log in via the Web VM).*

## 🔐 gRPC Service Methods

### `AuthService`

This service implements the strict RPC contract defined in your `.proto` file.

* **`rpc Login(LoginRequest) returns (LoginResponse)`**
* **Input:** Receives a payload containing a `username` and `password` string.
* **Process:** Opens `users.json`, searches for a matching username, and verifies the password.
* **Output:** Returns a boolean indicating success or failure, and an optional session token or error message.

