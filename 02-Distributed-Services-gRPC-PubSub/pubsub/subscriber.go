package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)


type MemoryEvent struct {
	EventType   string `json:"event_type"`
	Service     string `json:"service"`
	MemoryMB    uint64 `json:"memory_mb"`
	ThresholdMB uint64 `json:"threshold_mb"`
	Timestamp   string `json:"timestamp"`
}


func alertHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var event MemoryEvent
	err := json.NewDecoder(r.Body).Decode(&event)
	if err != nil {
		http.Error(w, "Invalid JSON data", http.StatusBadRequest)
		return
	}

	fmt.Println("[CRITICAL ALERT] HIGH MEMORY USAGE DETECTED! ")
	fmt.Printf("   Service: %s\n", event.Service)
	fmt.Printf("   Current Memory: %d MB\n", event.MemoryMB)
	fmt.Printf("   Threshold: %d MB\n", event.ThresholdMB)
	fmt.Printf("   Time: %s\n", event.Timestamp)

	w.WriteHeader(http.StatusOK)
}

func main() {
	http.HandleFunc("/alert", alertHandler)

	log.Println("Subscriber is running and waiting for events on port 9090")

	if err := http.ListenAndServe(":9090", nil); err != nil {
		log.Fatalf("Subscriber server failed: %v", err)
	}
}