package main

import (
	"bufio"
	"fmt"
	"os"
	"syscall"
	"strconv"
	"strings"
	"encoding/json"
)

const (
	reqPipe = "req_pipe"
	resPipe = "res_pipe"
)

type Response struct {
	Status string  `json:"status"`
	Result *float64 `json:"result,omitempty"`
	Error  string  `json:"error,omitempty"`
}

func initPipes() {
	fmt.Println("Starting Worker Initialization...")
	makeNamedPipe(reqPipe)
	makeNamedPipe(resPipe)
}

func makeNamedPipe(name string) {
	if _, err := os.Stat(name); os.IsNotExist(err) {

		err := syscall.Mkfifo(name, 0666)
		if err != nil {
			fmt.Println("Error creating pipe", name, ":", err)
			os.Exit(1)
		}
	}
}

func orderProcessing(request string) string{

	var resObj Response

	parts := strings.Split(request, " ")
	if len(parts) < 3 {
		resObj.Status = "ERR"
		resObj.Error = "Not complete input"
		return toJSON(resObj)

	}
	op := parts[0]
	a, err1 := strconv.ParseFloat(parts[1], 64)
	b, err2 := strconv.ParseFloat(parts[2], 64)

	if err1 != nil || err2 != nil {
		resObj.Status = "ERR"
		resObj.Error = "Invalid input"
		return toJSON(resObj)
	}
	var result float64
	switch op {
		case "ADD":
			result = a + b
		case "SUB":
			result = a - b
		case "MUL":
			result = a * b
		case "DIV":
			if b != 0 {
				result = a / b
			} else {
				resObj.Status = "ERR"
				resObj.Error = "Division by zero error"
				return toJSON(resObj)
			}
		default:
			resObj.Status = "ERR"
			resObj.Error = "Unknown operation"
			return toJSON(resObj)
		}
	resObj.Status = "OK"
	resObj.Result = &result
	return toJSON(resObj)
}

func handleConnection(){
	for{
		reqFile, err := os.OpenFile(reqPipe, os.O_RDONLY, os.ModeNamedPipe)
		if err != nil {
			fmt.Println("Error opening pipe:", err)
			continue
		}
		defer reqFile.Close()

		resFile, err := os.OpenFile(resPipe, os.O_WRONLY, os.ModeNamedPipe)
		if err != nil {
			reqFile.Close()
			continue
		}
		defer resFile.Close()

		scanner := bufio.NewScanner(reqFile)
		for scanner.Scan() {
			request := scanner.Text()
			response := orderProcessing(request)
			resFile.WriteString(response+ "\n")
		}
	}
}

func toJSON(resObj Response) string {
	jsonData, err := json.Marshal(resObj)
	if err != nil {
		return `{"status":"ERR", "error":"JSON encoding failed"}`
	}
	return string(jsonData)
}

func main() {
	initPipes()
	fmt.Println("Worker is ready")
	handleConnection()
}