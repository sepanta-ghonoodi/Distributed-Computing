FROM golang:1.22-alpine

WORKDIR /app

ENV GOPROXY=https://mirror-go.runflare.com

RUN go mod init hw1/part3

COPY main.go .

RUN go build -o server main.go

EXPOSE 8080

CMD ["./server"]