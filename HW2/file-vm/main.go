package main

import (
	"io"
	"log"
	"net"
	"os"
	"path/filepath"

	pb "app/pb/FileService"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type server struct {
	pb.UnimplementedFileSystemServer
}

func (s *server) DownloadFile(req *pb.FileRequest, stream pb.FileSystem_DownloadFileServer) error {
	log.Printf("[gRPC Inbound] Target download request received for asset: %s", req.GetFileName())

	md, ok := metadata.FromIncomingContext(stream.Context())
	if !ok {
		return status.Errorf(codes.Unauthenticated, "missing network transaction metadata")
	}

	//
	tokens := md["authorization"]
	if len(tokens) == 0 || tokens[0] == "" {
		log.Printf("[Security Alert] Blocked unauthenticated download attempt for file: %s", req.GetFileName())
		return status.Errorf(codes.Unauthenticated, "access token omitted from transaction context")
	}

	receivedToken := tokens[0]
	log.Printf("[Security Check] Evaluating download request with passport token: %s", receivedToken)

	targetPath := filepath.Join("files", filepath.Clean(req.GetFileName()))

	file, err := os.Open(targetPath)
	if err != nil {
		log.Printf("[File Error] Unable to find or access target resource: %v", err)
		if os.IsNotExist(err) {
			return status.Errorf(codes.NotFound, "requested asset %s does not exist on storage node", req.GetFileName())
		}
		return status.Errorf(codes.Internal, "internal disk I/O error occurred on file server")
	}
	defer file.Close()

	buffer := make([]byte, 4096)

	for {
		bytesRead, err := file.Read(buffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Printf("[Read Error] System I/O disruption tracking stream: %v", err)
			return status.Errorf(codes.Internal, "file stream read disruption: %v", err)
		}

		err = stream.Send(&pb.FileResponse{
			Chunk: buffer[:bytesRead],
		})
		if err != nil {
			log.Printf("[Stream Error] Network pipe disrupted during dispatch: %v", err)
			return err
		}
	}

	log.Printf("[Success] Stream delivery complete for asset: %s", req.GetFileName())
	return nil
}

func main() {
	listener, err := net.Listen("tcp", "0.0.0.0:50053")
	if err != nil {
		log.Fatalf("Network Binding Failure on port 50053: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterFileSystemServer(grpcServer, &server{})

	log.Println("File Service (VM3) listening on network port :50053...")
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Server lifecycle crash: %v", err)
	}
}
