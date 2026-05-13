package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)
const (
	reqPipe = "req_pipe"
	resPipe = "res_pipe"
)
func checkPipesExist() {
	if _, err := os.Stat(reqPipe); os.IsNotExist(err) {
		fmt.Println("ERR pipes not found")
		os.Exit(1)
	}
}
func runInteractiveLoop(reqFile, resFile *os.File) {
	reader := bufio.NewReader(os.Stdin)
	resScanner := bufio.NewScanner(resFile)

	fmt.Println("Interface ready")
	fmt.Println("Type a command (or 'exit' to quit):")

	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			break
		}
		if input == "" {
			continue
		}

		reqFile.WriteString(input + "\n")

		if resScanner.Scan() {
			fmt.Println(resScanner.Text())
		} else {
			fmt.Println("ERR Lost connection to Worker")
			break
		}
	}
}
func main() {
	checkPipesExist()

	reqFile, err := os.OpenFile(reqPipe, os.O_WRONLY, os.ModeNamedPipe)
	if err != nil {
		fmt.Println("Error connecting to Worker:", err)
		os.Exit(1)
	}
	defer reqFile.Close()

	resFile, err := os.OpenFile(resPipe, os.O_RDONLY, os.ModeNamedPipe)
	if err != nil {
		fmt.Println("Error opening response pipe:", err)
		os.Exit(1)
	}
	defer resFile.Close()

	runInteractiveLoop(reqFile, resFile)
}