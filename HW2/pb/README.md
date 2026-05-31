# Protocol Buffers Shared Module (`pb`)

This directory acts as the **Single Source of Truth (SSOT)** for all gRPC API contracts in the distributed system. By centralizing the `.proto` definitions and their generated Go code into an independent module, the Web, Auth, and File virtual machines can securely share the exact same data structures without duplicating code.

## Architecture & Usage

This directory is initialized as its own Go module (`module github.com/yourname/myproject/pb` or similar) and is linked to the other microservices locally via the root `go.work` file.

* **AuthService:** Contains the `LoginRequest` and `LoginResponse` contracts.
* **FileService:** Contains the chunked binary streaming contracts for file downloads.

### Directory Layout
```text
HW2/
├── go.work
└── pb/
    ├── go.mod                  (Independent module boundary)
    ├── AuthService/
    │   ├── auth.proto          (Raw contract)
    │   ├── auth.pb.go          (Auto-generated structs)
    │   └── auth_grpc.pb.go     (Auto-generated client/server stubs)
    └── FileService/
        ├── file.proto          (Raw contract)
        ├── file.pb.go          (Auto-generated structs)
        └── file_grpc.pb.go     (Auto-generated client/server stubs)

```

## 🛠 How to Generate Code

If you make any changes to the API contracts (adding new fields to a request, changing a method name, etc.), you must update the `.proto` file and re-compile the Go code.

1. Ensure you have the Protocol Buffers compiler (`protoc`) and the Go gRPC plugins installed on your machine.
2. Open your terminal in the root `HW2/` directory (where the `pb/` folder lives).
3. Run the following commands to generate the updated code in-place:

**For the Auth Service:**

```bash
protoc --go_out=. --go-grpc_out=. pb/AuthService/auth.proto

```

**For the File Service:**

```bash
protoc --go_out=. --go-grpc_out=. pb/FileService/file.proto

```
