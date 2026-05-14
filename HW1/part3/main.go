package main

import (
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
	"os"
)

type ComputeResponse struct {
	Operation string `json:"operation"`
	A         int    `json:"a"`
	B         int    `json:"b"`
	Result    int    `json:"result"`
}

func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next(w, r)

		duration := time.Since(start)

		slog.Info("Incoming Request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.String("remote_addr", r.RemoteAddr),
			slog.String("query", r.URL.RawQuery),
			slog.Duration("duration", duration),
		)
	}
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	http.HandleFunc("/health", loggingMiddleware(healthHandler))
	http.HandleFunc("/compute", loggingMiddleware(computeHandler))

	port := ":8080"
	fmt.Printf("Server starting on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func computeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error": "Method Not Allowed!"}`, http.StatusMethodNotAllowed)
		return
	}

	query := r.URL.Query()

	if !query.Has("op") || !query.Has("a") || !query.Has("b") {
		http.Error(w, `{"error": "Required: op, a, b"}`, http.StatusBadRequest)
		return
	}

	op := strings.ToLower(query.Get("op"))
	aStr := query.Get("a")
	bStr := query.Get("b")

	a, errA := strconv.Atoi(aStr)
	b, errB := strconv.Atoi(bStr)
	if errA != nil || errB != nil {
		http.Error(w, `{"error": "Parameters 'a' and 'b' must be integers"}`, http.StatusBadRequest)
		return
	}

	var result int
	switch op {

	case "add":
		result = a + b
	case "sub":
		result = a - b
	case "mul":
		result = a * b
	case "div":
		if b == 0 {
			http.Error(w, `{"error": "Division by zero"}`, http.StatusBadRequest)
			return
		}
		result = a / b
	case "mod":
		if b == 0 {
			http.Error(w, `{"error": "Modulo by zero"}`, http.StatusBadRequest)
			return
		}
		result = a % b
	default:
		http.Error(w, `{"error": "Invalid operation. Supported: add, sub, mul, div, mod"}`, http.StatusBadRequest)
		return
	}

	response := ComputeResponse{
		Operation: op,
		A:         a,
		B:         b,
		Result:    result,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}
