package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

type KeyEntry struct {
	Key       string `json:"key"`
	Value     string `json:"value"`
	Version   int    `json:"version"`
	UpdatedBy string `json:"updated_by"`
	Timestamp int64  `json:"timestamp"`
}

type Config struct {
	ID     string            `json:"id"`
	Port   int               `json:"port"`
	DBPath string            `json:"db_path"`
	Peers  map[string]string `json:"peers"`
}

type ReplicaServer struct {
	config             Config
	mu                 sync.RWMutex
	store              map[string]KeyEntry
	replicationDelayMs int
	httpClient         *http.Client
}


func NewReplicaServer(cfg Config) *ReplicaServer {
	s := &ReplicaServer{
		config: cfg,
		store:  make(map[string]KeyEntry),
		httpClient: &http.Client{
			Timeout: 500 * time.Millisecond,
		},
	}
	s.loadDB()
	return s
}

func (s *ReplicaServer) loadDB() {
	s.mu.Lock()
	defer s.mu.Unlock()

	file, err := os.Open(s.config.DBPath)
	if err != nil {
		return
	}
	defer file.Close()

	json.NewDecoder(file).Decode(&s.store)
}

func (s *ReplicaServer) saveDB() {
	data, _ := json.MarshalIndent(s.store, "", "  ")
	os.WriteFile(s.config.DBPath, data, 0644)
}

func main() {
	configPath := flag.String("config", "", "Path to the configuration JSON file")
	flag.Parse()

	if *configPath == "" {
		log.Fatal("Config path must be specified via -config flag")
	}

	cfgData, err := os.ReadFile(*configPath)
	if err != nil {
		log.Fatalf("Error reading config file: %v", err)
	}

	var cfg Config
	if err := json.Unmarshal(cfgData, &cfg); err != nil {
		log.Fatalf("Error parsing config file: %v", err)
	}

	server := NewReplicaServer(cfg)
	server.Start()
}

func (s *ReplicaServer) Start() {
	mux := http.NewServeMux()
	mux.HandleFunc("/get", s.handleGet)
	mux.HandleFunc("/local-get", s.handleLocalGet)
	mux.HandleFunc("/put", s.handlePut)
	mux.HandleFunc("/replicate", s.handleReplicate)
	mux.HandleFunc("/config", s.handleConfig)
	mux.HandleFunc("/status", s.handleStatus)

	addr := fmt.Sprintf(":%d", s.config.Port)
	log.Printf("[%s] Starting replica server on %s", s.config.ID, addr)

	srv := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[%s] Server failed to start: %v", s.config.ID, err)
	}
}

func (s *ReplicaServer) handleGet(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	consistency := r.URL.Query().Get("consistency")

	if key == "" {
		http.Error(w, "Missing key parameter", http.StatusBadRequest)
		return
	}

	if consistency == "strong" {
		s.handleStrongGet(w, key)
		return
	}

	s.mu.RLock()
	entry, exists := s.store[key]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, fmt.Sprintf("Key '%s' not found", key), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entry)
}


func (s *ReplicaServer) handleLocalGet(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	if key == "" {
		http.Error(w, "Missing key parameter", http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	entry, exists := s.store[key]
	s.mu.RUnlock()

	if !exists {
		entry = KeyEntry{Key: key, Version: 0}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entry)
}

func (s *ReplicaServer) handleStrongGet(w http.ResponseWriter, key string) {
	log.Printf("[%s] Coordinated GET (strong consistency) for key: %s", s.config.ID, key)

	targets := s.getReplicaAddresses()
	majority := (len(targets) / 2) + 1

	successfulEntries, failedURLs, err := s.fetchQuorumEntries(key, targets, majority)
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}

	latest := successfulEntries[0]
	for _, entry := range successfulEntries[1:] {
		if isNewer(entry, latest) {
			latest = entry
		}
	}

	go s.runReadRepair(latest, failedURLs, successfulEntries)

	if latest.Version == 0 {
		http.Error(w, fmt.Sprintf("Key '%s' not found", key), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(latest)
}

func (s *ReplicaServer) getReplicaAddresses() []string {
	targets := []string{fmt.Sprintf("http://localhost:%d", s.config.Port)}
	for _, peerURL := range s.config.Peers {
		targets = append(targets, peerURL)
	}
	return targets
}

func (s *ReplicaServer) fetchLocalOrRemoteEntry(targetURL, key string) (KeyEntry, error) {
	localURL := fmt.Sprintf("http://localhost:%d", s.config.Port)
	if targetURL == localURL {
		s.mu.RLock()
		entry, exists := s.store[key]
		s.mu.RUnlock()
		if !exists {
			return KeyEntry{Key: key, Version: 0}, nil
		}
		return entry, nil
	}

	resp, err := s.httpClient.Get(fmt.Sprintf("%s/local-get?key=%s", targetURL, key))
	if err != nil {
		return KeyEntry{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return KeyEntry{}, fmt.Errorf("HTTP status %d", resp.StatusCode)
	}

	var entry KeyEntry
	if err := json.NewDecoder(resp.Body).Decode(&entry); err != nil {
		return KeyEntry{}, err
	}
	return entry, nil
}

func (s *ReplicaServer) fetchQuorumEntries(key string, targets []string, majority int) ([]KeyEntry, []string, error) {
	type readResult struct {
		entry KeyEntry
		err   error
		url   string
	}

	ch := make(chan readResult, len(targets))
	var wg sync.WaitGroup

	for _, url := range targets {
		wg.Add(1)
		go func(targetURL string) {
			defer wg.Done()
			entry, err := s.fetchLocalOrRemoteEntry(targetURL, key)
			ch <- readResult{entry: entry, err: err, url: targetURL}
		}(url)
	}

	wg.Wait()
	close(ch)

	var successfulEntries []KeyEntry
	var failedURLs []string

	for res := range ch {
		if res.err == nil {
			successfulEntries = append(successfulEntries, res.entry)
		} else {
			failedURLs = append(failedURLs, res.url)
		}
	}

	if len(successfulEntries) < majority {
		return nil, nil, fmt.Errorf("Quorum read failed: only %d of %d replicas available", len(successfulEntries), majority)
	}

	return successfulEntries, failedURLs, nil
}

func (s *ReplicaServer) runReadRepair(latest KeyEntry, failedURLs []string, successfulEntries []KeyEntry) {
	if latest.Version == 0 {
		return
	}

	for _, url := range failedURLs {
		go s.sendReplicationRequest(url, latest)
	}

	for _, peerURL := range s.config.Peers {
		go s.sendReplicationRequest(peerURL, latest)
	}
}

func (s *ReplicaServer) handlePut(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Key         string `json:"key"`
		Value       string `json:"value"`
		Consistency string `json:"consistency"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	if req.Key == "" {
		http.Error(w, "Missing key parameter", http.StatusBadRequest)
		return
	}

	if req.Consistency == "strong" {
		s.handleStrongPut(w, req.Key, req.Value)
		return
	}

	s.handleEventualPut(w, req.Key, req.Value)
}

func (s *ReplicaServer) handleEventualPut(w http.ResponseWriter, key string, value string) {
	s.mu.Lock()
	current := s.store[key]
	newVersion := current.Version + 1

	entry := KeyEntry{
		Key:       key,
		Value:     value,
		Version:   newVersion,
		UpdatedBy: s.config.ID,
		Timestamp: time.Now().UnixNano(),
	}

	s.store[key] = entry
	s.saveDB()
	s.mu.Unlock()

	log.Printf("[%s] Eventual PUT written locally: key=%s, value=%s, version=%d", s.config.ID, key, value, newVersion)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entry)

	go func() {
		for _, peerURL := range s.config.Peers {
			go s.sendReplicationRequest(peerURL, entry)
		}
	}()
}

func (s *ReplicaServer) handleStrongPut(w http.ResponseWriter, key string, value string) {
	log.Printf("[%s] Coordinated PUT (strong consistency) for key: %s, value: %s", s.config.ID, key, value)

	targets := s.getReplicaAddresses()
	majority := (len(targets) / 2) + 1

	maxVersion, err := s.getQuorumVersion(key, targets, majority)
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}

	entry := KeyEntry{
		Key:       key,
		Value:     value,
		Version:   maxVersion + 1,
		UpdatedBy: s.config.ID,
		Timestamp: time.Now().UnixNano(),
	}

	successfulWrites := s.replicateToQuorum(entry, targets)
	if successfulWrites < majority {
		http.Error(w, fmt.Sprintf("Quorum write failed: only %d of %d replicas confirmed write", successfulWrites, majority), http.StatusServiceUnavailable)
		return
	}

	log.Printf("[%s] Coordinated PUT (strong consistency) succeeded. Quorum size: %d/%d. Version: %d", s.config.ID, successfulWrites, len(targets), entry.Version)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entry)
}

func (s *ReplicaServer) getQuorumVersion(key string, targets []string, majority int) (int, error) {
	type versionResult struct {
		version int
		err     error
	}

	vCh := make(chan versionResult, len(targets))
	var wg sync.WaitGroup

	for _, url := range targets {
		wg.Add(1)
		go func(targetURL string) {
			defer wg.Done()
			entry, err := s.fetchLocalOrRemoteEntry(targetURL, key)
			if err != nil {
				vCh <- versionResult{err: err}
			} else {
				vCh <- versionResult{version: entry.Version}
			}
		}(url)
	}

	wg.Wait()
	close(vCh)

	var successfulVersions []int
	for res := range vCh {
		if res.err == nil {
			successfulVersions = append(successfulVersions, res.version)
		}
	}

	if len(successfulVersions) < majority {
		return 0, fmt.Errorf("Quorum write check failed: only %d of %d replicas available to get version", len(successfulVersions), majority)
	}

	maxVersion := 0
	for _, v := range successfulVersions {
		if v > maxVersion {
			maxVersion = v
		}
	}
	return maxVersion, nil
}

func (s *ReplicaServer) replicateToQuorum(entry KeyEntry, targets []string) int {
	type writeResult struct {
		success bool
	}

	wCh := make(chan writeResult, len(targets))
	var wg sync.WaitGroup

	for _, url := range targets {
		wg.Add(1)
		go func(targetURL string) {
			defer wg.Done()
			var success bool
			localURL := fmt.Sprintf("http://localhost:%d", s.config.Port)
			if targetURL == localURL {
				success = s.applyReplication(entry)
			} else {
				success = s.sendReplicationRequestSync(targetURL, entry)
			}
			wCh <- writeResult{success: success}
		}(url)
	}

	wg.Wait()
	close(wCh)

	successfulWrites := 0
	for res := range wCh {
		if res.success {
			successfulWrites++
		}
	}
	return successfulWrites
}

func (s *ReplicaServer) handleReplicate(w http.ResponseWriter, r *http.Request) {
	var entry KeyEntry
	if err := json.NewDecoder(r.Body).Decode(&entry); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	success := s.applyReplication(entry)
	w.Header().Set("Content-Type", "application/json")
	if success {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "success", "replica": s.config.ID})
	} else {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ignored", "replica": s.config.ID, "reason": "stale version or lost tie-breaker"})
	}
}

func (s *ReplicaServer) applyReplication(entry KeyEntry) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	local, exists := s.store[entry.Key]
	if !exists {
		s.store[entry.Key] = entry
		s.saveDB()
		log.Printf("[%s] Replicated write: new key %s = %s (v%d, by %s)", s.config.ID, entry.Key, entry.Value, entry.Version, entry.UpdatedBy)
		return true
	}

	if entry.Version > local.Version {
		s.store[entry.Key] = entry
		s.saveDB()
		log.Printf("[%s] Replicated write: updated %s = %s (v%d over v%d, by %s)", s.config.ID, entry.Key, entry.Value, entry.Version, local.Version, entry.UpdatedBy)
		return true
	}

	if entry.Version < local.Version {
		log.Printf("[%s] Replicated write ignored: stale version %d < %d for key %s", s.config.ID, entry.Version, local.Version, entry.Key)
		return false
	}

	if isNewer(entry, local) {
		s.store[entry.Key] = entry
		s.saveDB()
		log.Printf("[%s] Conflict resolved: updated %s = %s (same version %d, tie-breaker won by %s over %s)", s.config.ID, entry.Key, entry.Value, entry.Version, entry.UpdatedBy, local.UpdatedBy)
		return true
	}

	log.Printf("[%s] Conflict resolved: kept local key %s = %s (same version %d, tie-breaker lost by %s)", s.config.ID, entry.Key, local.Value, local.Version, entry.UpdatedBy)
	return false
}

func isNewer(a, b KeyEntry) bool {
	if a.Version > b.Version {
		return true
	}
	if a.Version < b.Version {
		return false
	}
	if a.Timestamp > b.Timestamp {
		return true
	}
	if a.Timestamp < b.Timestamp {
		return false
	}
	return a.UpdatedBy > b.UpdatedBy
}

func (s *ReplicaServer) sendReplicationRequest(peerURL string, entry KeyEntry) {
	s.sendReplicationRequestSync(peerURL, entry)
}

// sendReplicationRequestSync sends a sync replication request and returns success status.
func (s *ReplicaServer) sendReplicationRequestSync(peerURL string, entry KeyEntry) bool {
	s.mu.RLock()
	delay := s.replicationDelayMs
	s.mu.RUnlock()
	if delay > 0 {
		time.Sleep(time.Duration(delay) * time.Millisecond)
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return false
	}

	resp, err := s.httpClient.Post(fmt.Sprintf("%s/replicate", peerURL), "application/json", bytes.NewBuffer(data))
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false
	}

	var res struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return false
	}

	return res.Status == "success" || res.Status == "ignored"
}

func (s *ReplicaServer) handleConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		DelayMs int `json:"delay_ms"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	s.replicationDelayMs = req.DelayMs
	s.mu.Unlock()

	log.Printf("[%s] Dynamic config updated: replication delay = %d ms", s.config.ID, req.DelayMs)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "success", "delay_ms": req.DelayMs})
}

func (s *ReplicaServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":                 s.config.ID,
		"port":               s.config.Port,
		"db_path":            s.config.DBPath,
		"replication_delay": s.replicationDelayMs,
		"peers":              s.config.Peers,
		"store":              s.store,
	})
}


