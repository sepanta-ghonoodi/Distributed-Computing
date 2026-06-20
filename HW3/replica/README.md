# Replicated Key-Value Store - Replica Server

This folder contains the Go implementation of the replica server.

## Features
- **Independent Services**: Each replica runs as a standalone HTTP server process.
- **Local Database Persistence**: Each replica stores its state in a local JSON database file (e.g. `data_replica1.json`).
- **Eventual Consistency**: PUT operations return immediately after writing locally. Replications to peers are sent asynchronously (with support for artificial replication delays).
- **Simplified Strong Consistency**: PUT operations write to a majority (2 out of 3) of replicas in parallel and succeed only if the majority confirms. GET operations perform a coordinated quorum read across a majority of replicas.
- **Versioning & Last-Write-Wins (LWW)**: Every key stores its version, the ID of the replica that performed the last update, and a nanosecond-precision timestamp. Conflicts are resolved deterministically (version > timestamp > replica ID).

## Config JSON Format
Each replica reads its settings from a config JSON file:
```json
{
  "id": "replica1",
  "port": 8001,
  "db_path": "data_replica1.json",
  "peers": {
    "replica2": "http://localhost:8002",
    "replica3": "http://localhost:8003"
  }
}
```

## How to Run a Replica
To start a replica with a specific configuration:
```bash
go run main.go -config ../configs/replica1.json
```

## API Endpoints
- `GET /get?key=x&consistency=[eventual|strong]`: Read a key's value.
- `GET /local-get?key=x`: Read the key's value locally (used internally for coordination).
- `POST /put`: Save a key-value pair.
  ```json
  {
    "key": "x",
    "value": "10",
    "consistency": "eventual"
  }
  ```
- `POST /replicate`: Replicate a value from a coordinator (used internally).
- `POST /config`: Dynamically configure settings (e.g. replication delay).
  ```json
  {
    "delay_ms": 500
  }
  ```
- `GET /status`: View the state of the store and config.
- `POST /shutdown`: Shut down the replica process.
