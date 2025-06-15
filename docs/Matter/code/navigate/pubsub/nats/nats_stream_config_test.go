package nats_test

import (
	"encoding/json"
	"testing"
	"time"

	nats "github.com/nats-io/nats.go"
)

func TestStream(t *testing.T) {
	nc, err := nats.Connect(serverIpPort, nats.UserInfo(user_navl_in, user_navl_in_password))
	if err != nil {
		t.Logf("connect err: %+v", err)
		t.Fail()
	}
	defer func() {
		err := nc.Drain()
		if err != nil {
			t.Logf("nc.Drain() err: %+v", err)
		}
		nc.Close()
	}()

	js, err := nc.JetStream()
	if err != nil {
		t.Logf("js: %+v,err:%+v", js, err)
	}

	cfg := nats.StreamConfig{
		Name:         "navl",
		Description:  "测量的数据存储",
		Subjects:     []string{"rt.gzb.navl.>"},
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

	_, err = js.AddStream(&cfg)
	if err != nil {
		t.Logf("js.AddStream err: %+v", err)
	}

	msg, err := js.Publish("rt.gzb.navl.1.in.2", nil)
	t.Logf("js.Publish: msg: %+v,err : %+v", msg, err)

	{
		ack, err := js.PublishAsync("rt.gzb.navl.1.in.2", nil)
		if err != nil {
			t.Logf("ack err: %+v,%+v", ack, err)
		} else {
			<-ack.Ok()
			t.Logf("ack msg: %+v", ack.Msg())
		}
	}

	select {
	case <-js.PublishAsyncComplete():
		t.Log("published  messages")
	case <-time.After(5 * time.Second):
		t.Logf("publish took too long")
		t.Fail()
	}
	printStreamState(t, js, cfg.Name)

	consumerCfg := nats.ConsumerConfig{
		Durable:       "rt.gzb.navl.1.in.2",
		Name:          "led",
		Description:   "consumer for led",
		DeliverPolicy: nats.DeliverAllPolicy,
		AckPolicy:     nats.AckNonePolicy,
	}
	consumerInfo, err := js.AddConsumer(cfg.Name, &consumerCfg)
	t.Logf("js.AddConsumer: consumerInfo: %+v,err : %+v", consumerInfo, err)
	// // 限制消息大小
	// cfg.MaxBytes = 300
	// js.UpdateStream(&cfg)
	// fmt.Println("set max bytes to 300")

	// printStreamState(t, js, cfg.Name)

	// // 限制消息最大存活时间
	// cfg.MaxAge = time.Second
	// js.UpdateStream(&cfg)
	// fmt.Println("set max age to one second")

	// printStreamState(t, js, cfg.Name)

	// fmt.Println("sleeping one second...")
	// time.Sleep(time.Second)

	// printStreamState(t, js, cfg.Name)
}

func printStreamState(t *testing.T, js nats.JetStreamContext, name string) {
	info, _ := js.StreamInfo(name)
	b, _ := json.MarshalIndent(info.State, "", " ")
	t.Log("inspecting stream info")
	t.Log(string(b))
}
