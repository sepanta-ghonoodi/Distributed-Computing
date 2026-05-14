
# Containerizing and Running a Service in a VM

This directory contains the implementation for Part 3 of the Distributed Computing Fundamentals assignment. In this section, an HTTP service is written in Go, containerized using Docker, and executed within a Virtual Machine (Guest) while being accessed from the Host machine.

## 1. File Structure

* `main.go`: The main source code of the application, including the web server, endpoints, error handling, and logging middleware.
* `Dockerfile`: The configuration file to build the Docker image, utilizing `golang:1.22-alpine` and the required `runflare` proxy settings.
* `README.md`: The documentation file you are currently reading.

## 2. Dependencies and Environment

* **Language:** Go 1.22 (strictly utilizing standard libraries `net/http` and `encoding/json` without external frameworks).
* **Containerization:** Docker Engine for building and running the service.
* **Proxy Configuration:** Due to network restrictions, the local mirror `https://mirror-go.runflare.com` is configured via the `GOPROXY` environment variable inside the `Dockerfile` to ensure successful builds.

## 3. Docker Image Build Command

To build the Docker image, navigate to this directory inside the VM's terminal and execute the following command:

```bash
sudo docker build -t ds-hw1-part3 .

```

## 4. Container Execution and Port Mapping

The service is configured to listen on **Port 8080**. To run the container in detached mode and map the ports, use this command:

```bash
sudo docker run -d -p 8080:8080 --name calc-service ds-hw1-part3

```

*This command routes traffic from port `8080` on the Virtual Machine directly into port `8080` of the Docker container.*

## 5. Host to VM Connection Method

To successfully establish a connection between the physical Host machine and the Guest VM, the default **NAT** network adapter in **VirtualBox** was utilized.

A **Port Forwarding** rule was defined in the VM's network settings as follows:

* **Host Port:** `8080`
* **Guest Port:** `8080`

With this configuration, all HTTP requests sent from the Host machine to `localhost:8080` are intercepted by the hypervisor and routed directly into the VM and the Docker container.

## 6. Sample Executions (Tested from Host)

### Health Check Test

```bash
curl -i http://localhost:8080/health

```

**Expected Output:** `HTTP/1.1 200 OK` along with the body text `OK`.

### Compute Operation Test (Example: Addition)

```bash
curl -s "http://localhost:8080/compute?op=add&a=5&b=7"

```

**Expected JSON Output:**

```json
{"operation":"add","a":5,"b":7,"result":12}

```

### Error Handling Test (Example: Division by Zero)

```bash
curl --noproxy "*" -s "http://localhost:8080/compute?op=div&a=10&b=0"

```

**Expected JSON Output:**

```json
{"error": "Division by zero"}

```

## 7. Bonus Features (Extra Points)

To differentiate this group's submission and improve code quality, two additional features were implemented:

## Features & Architecture

* **Extended Operations:** Supports standard arithmetic (`add`, `sub`, `mul`, `div`) as well as modulo (`mod`). Includes strict input validation and edge-case handling (e.g., division/modulo by zero protection).
* **Structured Telemetry:** Implements a custom HTTP middleware using Go's native `log/slog`. All incoming traffic generates structured JSON logs containing the HTTP method, request path, client IP, and total processing duration.
  * *To view live logs, run:* `docker logs -f calc-service`
