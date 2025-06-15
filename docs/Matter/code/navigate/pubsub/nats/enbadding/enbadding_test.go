package enbadding_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats.go"
	"go.uber.org/goleak"
)

const (
	user     = "testuser"
	password = "testpassword"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestServer(t *testing.T) {
	opts := &server.Options{}
	opts.Host = "127.0.0.1"
	opts.Port = 55555
	ns, err := server.NewServer(opts)

	if err != nil {
		panic(err)
	}

	// Start the server via goroutine
	go ns.Start()

	// Wait for server to be ready for connections
	if !ns.ReadyForConnections(4 * time.Second) {
		panic("not ready for connection")
	}
	t.Logf("Connect url: %s", ns.ClientURL())
	nc, err := nats.Connect(ns.ClientURL(), nats.UserInfo(user, password))
	if err != nil {
		panic(err)
	}

	subject := "my-subject"

	// Subscribe to the subject
	nc.Subscribe(subject, func(msg *nats.Msg) {
		// Print message data
		data := string(msg.Data)
		fmt.Println(data)

		// Shutdown the server (optional)
		ns.Shutdown()
	})

	// Publish data to the subject
	nc.Publish(subject, []byte("Hello embedded NATS!"))

	// Wait for server shutdown
	ns.WaitForShutdown()
}

func TestServerJS(t *testing.T) {
	opts := &server.Options{}
	opts.Port = server.RANDOM_PORT
	embeddedServer, err := server.NewServer(opts)
	if err != nil {
		panic(err)
	}

	serverAddress := embeddedServer.ClientURL()
	connection, err := nats.Connect(serverAddress, nats.RetryOnFailedConnect(true))
	if err != nil {
		t.Fatal(err)
	}

	jetstream, err := connection.JetStream()
	if err != nil {
		t.Fatal(err)
	}

	cfg := nats.StreamConfig{
		Name:         "measure",
		Description:  "测量的数据存储",
		Subjects:     []string{"measure.>"},
		Retention:    nats.WorkQueuePolicy,
		Storage:      nats.FileStorage,
		Discard:      nats.DiscardOld,
		NoAck:        false,
		MaxConsumers: -1,
		MaxMsgs:      -1,
		MaxBytes:     -1,
		MaxAge:       24 * 365 * time.Hour,
		MaxMsgSize:   -1,
		Replicas:     1,
		//MaxBytes: 1024,
	}

	_, err = jetstream.AddStream(&cfg)
	if err != nil {
		t.Logf("jetstream.AddStream err: %+v", err)
	}

	_, err = jetstream.AddStream(&nats.StreamConfig{Name: "test_stream"})
	if err != nil {
		t.Fatal(err)
	}
	embeddedServer.WaitForShutdown()
}
