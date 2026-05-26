package pubsub

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type MemoryEvent struct {
	EventType   string `json:"event_type"`
	Service     string `json:"service"`
	MemoryMB    uint64 `json:"memory_mb"`
	ThresholdMB uint64 `json:"threshold_mb"`
	Timestamp   string `json:"timestamp"`
}


func PublishAlert(memoryMB uint64, thresholdMB uint64, subscriberURL string) error {
	event := MemoryEvent{
		EventType:   "HIGH_MEMORY_USAGE",
		Service:     "web-server",
		MemoryMB:    memoryMB,
		ThresholdMB: thresholdMB,
		Timestamp:   time.Now().Format(time.RFC3339),
	}


	jsonData, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("error creating JSON: %v", err)
	}


	resp, err := http.Post(subscriberURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error connecting to subscriber: %v", err)
	}
	defer resp.Body.Close()

	return nil
}