package main

import (
	pb "pb/AuthService"
	
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log"
	"net"
	"os"
	"path/filepath"

	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc"
)

type User struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type authServer struct {
	pb.UnimplementedAuthSystemServer
	users []User
}

func generateSecureToken() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "fallback-static-token-123"
	}
	return hex.EncodeToString(b)
}

func loadUserDatabase() ([]User, error) {
	data, err := os.ReadFile(filepath.Join("users.json"))
	if err != nil {
		return nil, err
	}

	var users []User
	if err := json.Unmarshal(data, &users); err != nil {
		return nil, err
	}
	return users, nil
}
func (s *authServer) Login(ctx context.Context, req *pb.ReqAuth) (*pb.ResAuth, error) {
	log.Printf("[gRPC Inbound] Processing validation check for user: %s", req.GetUsername())

	user, found := s.findUserByUsername(req.GetUsername())
	if !found {
		log.Printf("[Rejected] Username not found in database: %s", req.GetUsername())
		return s.denyAccess(), nil
	}

	if !s.isPasswordValid(user.Password, req.GetPassword()) {
		log.Printf("[Rejected] Invalid password attempt for user: %s", req.GetUsername())
		return s.denyAccess(), nil
	}

	log.Printf("[Success] Verified identity matches for user: %s", req.GetUsername())
	return s.grantAccess(user.Username), nil
}

func (s *authServer) findUserByUsername(username string) (*User, bool) {
	for _, user := range s.users {
		if user.Username == username {
			return &user, true
		}
	}
	return nil, false
}

func (s *authServer) isPasswordValid(storedHash, fallbackPlaintext string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(storedHash), []byte(fallbackPlaintext))
	return err == nil
}

func (s *authServer) grantAccess(username string) *pb.ResAuth {
	assignedRole := pb.UserRole_USER
	if username == "admin" {
		assignedRole = pb.UserRole_ADMIN
	}

	return &pb.ResAuth{
		Token: generateSecureToken(),
		Role:  assignedRole,
	}
}

func (s *authServer) denyAccess() *pb.ResAuth {
	return &pb.ResAuth{
		Token: "",
		Role:  pb.UserRole_UNDEFINED,
	}
}

func main() {
	users, err := loadUserDatabase()
	if err != nil {
		log.Fatalf("Critical Error: Unable to read user database: %v", err)
	}
	log.Printf("Successfully registered %d users from users.json", len(users))

	listener, err := net.Listen("tcp", "0.0.0.0:50052")
	if err != nil {
		log.Fatalf("Network Binding Failure on port 50052: %v", err)
	}

	grpcServer := grpc.NewServer()

	serverInstance := &authServer{users: users}
	pb.RegisterAuthSystemServer(grpcServer, serverInstance)

	log.Println("Auth Service (VM2) listening on network port :50052...")

	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Server lifecycle crash: %v", err)
	}
}
