package main

import (
	"context"
	"html/template"
	"log"
	"net/http"

	authpb "app/pb/AuthService"
	filepb "app/pb/FileService"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// --- Global Variables ---
var (
	templates      *template.Template
	authServerAddr string
	fileServerAddr string
)

// --- Struct Layouts ---
type PageData struct {
	ErrorMessage string
}

type MemoryEvent struct {
	EventType   string `json:"event_type"`
	Service     string `json:"service"`
	MemoryMB    uint64 `json:"memory_mb"`
	ThresholdMB uint64 `json:"threshold_mb"`
	Timestamp   string `json:"timestamp"`
}

func main() {
	var err error
	templates, err = template.ParseFiles(filepath.Join("templates", "login.html"))
	if err != nil {
		log.Fatalf("Critical: Failed to compile UI templates: %v", err)
	}

	flag.StringVar(&authServerAddr, "auth", "localhost:50052", "Address of the Auth Server (VM2)")
	flag.StringVar(&fileServerAddr, "file", "localhost:50053", "Address of the File Server (VM3)")

	flag.Parse()

	log.Printf("System Configuration Loaded:")
	log.Printf(" -> Target Auth Endpoint: %s", authServerAddr)
	log.Printf(" -> Target File Endpoint: %s", fileServerAddr)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /login", loginGetHandler)
	mux.HandleFunc("POST /login", loginPostHandler)
	mux.HandleFunc("POST /download", fileRequestHandler)
	mux.HandleFunc("GET /dashboard", dashboardHandler)
	log.Println("Web Server (VM1) initializing on network port :8080...")

	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatalf("Server lifecycle crash: %v", err)
	}
}

func loginGetHandler(w http.ResponseWriter, r *http.Request) {
	err := templates.ExecuteTemplate(w, "login.html", PageData{ErrorMessage: ""})
	if err != nil {
		http.Error(w, "Template execution breakdown", http.StatusInternalServerError)
	}
}

func loginPostHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Form processing error", http.StatusBadRequest)
		return
	}

	username := r.FormValue("username")
	password := r.FormValue("password")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	log.Printf("[Auth] Requesting credential verification for user: %s", username)

	resp, err := authenticateUser(ctx, username, password)
	if err != nil {
		log.Printf("[Auth Error] %v", err)
		renderLoginError(w, "Authentication subsystem initialization failed.")
		return
	}

	if resp.GetToken() == "" || resp.GetRole() == authpb.UserRole_UNDEFINED {
		log.Printf("[Access Denied] VM2 rejected identity for: %s", username)
		renderLoginError(w, "Invalid username or password. Please try again.")
		return
	}

	log.Printf("[Access Granted] Verified by VM2. Role: %v", resp.GetRole())
	http.SetCookie(w, &http.Cookie{
		Name:     "session_token",
		Value:    resp.GetToken(),
		Path:     "/",
		HttpOnly: true,
		Expires:  time.Now().Add(1 * time.Hour),
	})

	http.SetCookie(w, &http.Cookie{Name: "username", Value: username, Path: "/"})
	http.SetCookie(w, &http.Cookie{Name: "user_role", Value: resp.GetRole().String(), Path: "/"})

	http.Redirect(w, r, "/dashboard", http.StatusSeeOther)
}

func authenticateUser(ctx context.Context, username, password string) (*authpb.ResAuth, error) {
	conn, err := grpc.NewClient(authServerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to auth server: %w", err)
	}
	defer conn.Close()

	client := authpb.NewAuthSystemClient(conn)
	return client.Login(ctx, &authpb.ReqAuth{
		Username: username,
		Password: password,
	})
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {

	tokenCookie, err := r.Cookie("session_token")
	if err != nil {

		log.Println("[Security Check] Unauthorized access attempt to /dashboard. Redirecting to login.")
		http.Redirect(w, r, "/login", http.StatusSeeOther)
		return
	}

	usernameCookie, _ := r.Cookie("username")
	roleCookie, _ := r.Cookie("user_role")

	data := struct {
		Username string
		Role     string
		Token    string
	}{
		Username: usernameCookie.Value,
		Role:     roleCookie.Value,
		Token:    tokenCookie.Value,
	}

	tmpl, err := template.ParseFiles("templates/dashboard.html")
	if err != nil {
		http.Error(w, "Unable to load dashboard template", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	_ = tmpl.Execute(w, data)
}

func renderLoginError(w http.ResponseWriter, msg string) {
	w.WriteHeader(http.StatusUnauthorized)
	_ = templates.ExecuteTemplate(w, "login.html", PageData{ErrorMessage: msg})
}

func fileRequestHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Form processing error", http.StatusBadRequest)
		return
	}

	filename := r.FormValue("filename")
	if filename == "" {
		http.Error(w, "Missing filename", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := streamFileToClient(ctx, w, filename, r); err != nil {
		log.Printf("[File Service Error] %v", err)
	}
}

func streamFileToClient(ctx context.Context, w http.ResponseWriter, filename string, r *http.Request) error {
	token, err := extractSessionToken(r)
	if err != nil {
		http.Error(w, "Unauthorized: Session missing or expired", http.StatusUnauthorized)
		return err
	}

	conn, client, err := connectToFileService()
	if err != nil {
		http.Error(w, "File service unavailable", http.StatusInternalServerError)
		return err
	}
	defer conn.Close()

	stream, err := initiateDownloadStream(ctx, client, filename, token)
	if err != nil {
		http.Error(w, "Could not initiate file transfer", http.StatusInternalServerError)
		return err
	}

	w.Header().Set("Content-Type", getMimeType(filename))
	return pipeStreamToHTTP(w, stream)
}

func extractSessionToken(r *http.Request) (string, error) {
	cookie, err := r.Cookie("session_token")
	if err != nil {
		return "", fmt.Errorf("missing session cookie: %w", err)
	}
	return cookie.Value, nil
}

func connectToFileService() (*grpc.ClientConn, filepb.FileSystemClient, error) {
	conn, err := grpc.NewClient(fileServerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, fmt.Errorf("connection to VM3 failed: %w", err)
	}
	return conn, filepb.NewFileSystemClient(conn), nil
}

func initiateDownloadStream(ctx context.Context, client filepb.FileSystemClient, filename, token string) (filepb.FileSystem_DownloadFileClient, error) {
	md := metadata.Pairs("authorization", token)
	ctxWithAuth := metadata.NewOutgoingContext(ctx, md)

	stream, err := client.DownloadFile(ctxWithAuth, &filepb.FileRequest{FileName: filename})
	if err != nil {
		return nil, fmt.Errorf("download initiation failed: %w", err)
	}
	return stream, nil
}

func pipeStreamToHTTP(w http.ResponseWriter, stream filepb.FileSystem_DownloadFileClient) error {
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("stream interrupted: %w", err)
		}

		if _, err := w.Write(chunk.GetChunk()); err != nil {
			return fmt.Errorf("failed to write to http client: %w", err)
		}
	}
}

func getMimeType(filename string) string {
	switch filepath.Ext(filename) {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".txt":
		return "text/plain"
	case ".pdf":
		return "application/pdf"
	default:
		return "application/octet-stream"
	}
}
