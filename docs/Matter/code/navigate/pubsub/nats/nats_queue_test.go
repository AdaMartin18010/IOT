package nats_test

import (
	"sync"
	"testing"
	"time"

	nats "github.com/nats-io/nats.go"
)

func TestQueue(t *testing.T) {
	nc, err := CreateNatsClient(serverIpPort, user_navl_in, user_navl_in_password, "iot_connect")
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}
	defer nc.Drain()
	defer nc.Close()

	// 这里最好是明确的 subject or topic name 虽然支持通配表达式
	//queSub := "rt.gzb.navl.1.in.cmp.svr.*"
	queSub := "rt.gzb.navl.1.in.cmp.svr.1"
	// 第一个消息因为没有订阅者 会被nats_server丢弃
	err = nc.Publish("rt.gzb.navl.1.in.cmp.svr.0", []byte("Hello World-00"))
	if err != nil {
		t.Logf("publish err: %+v", err)
		t.Fail()
	}
	// Flush 发布缓冲区
	err = nc.Flush()
	if err != nil {
		t.Logf("publish flush err: %+v", err)
		t.Fail()
	}

	queue := "que-whatever"
	closechan := make(chan struct{}, 3)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		queSub := "rt.gzb.navl.1.in.cmp.svr.*"
		subQue, err := nc.QueueSubscribe(queSub, queue, func(msg *nats.Msg) {
			t.Logf("*. sub:%s,data:%s", msg.Subject, msg.Data)
		})
		//t.Logf("1. %#v", subQue)
		if err != nil {
			t.Logf("1. subscribe queue err: %#v", err)
			t.Fail()
		}
		<-closechan
		subQue.Unsubscribe()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		subQue, err := nc.QueueSubscribe(queSub, queue, func(msg *nats.Msg) {
			t.Logf("2. sub:%s,data:%s", msg.Subject, msg.Data)
		})
		//t.Logf("2. %#v", subQue)
		if err != nil {
			t.Logf("2. subscribe queue err: %#v", err)
			t.Fail()
		}
		<-closechan
		subQue.Unsubscribe()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		subQue, err := nc.QueueSubscribe(queSub, queue, func(msg *nats.Msg) {
			t.Logf("3. sub:%s,data:%s", msg.Subject, msg.Data)
		})
		//t.Logf("3. %#v", subQue)
		if err != nil {
			t.Logf("3. subscribe queue err: %#v", err)
			t.Fail()
		}
		<-closechan
		subQue.Unsubscribe()
	}()

	// 第一个消息因为没有订阅 所以会收不到
	sub, err := nc.QueueSubscribeSync(queSub, queue)
	if err != nil {
		t.Logf("subscribe queue err: %+v", err)
		t.Fail()
		return
	}
	defer sub.Unsubscribe()

	wg.Add(1)
	go func() {
		defer wg.Done()
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-01"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-02"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-03"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-04"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-05"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-06"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-07"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-08"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-09"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-10"))
		// Flush 发布缓冲区
		err = nc.Flush()
		if err != nil {
			t.Logf("subscribe flush err: %+v", err)
			t.Fail()
		}
		<-closechan
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.1", []byte("Hello World-01"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.2", []byte("Hello World-02"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.3", []byte("Hello World-03"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.4", []byte("Hello World-04"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.5", []byte("Hello World-05"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.6", []byte("Hello World-06"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.7", []byte("Hello World-07"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.8", []byte("Hello World-08"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.9", []byte("Hello World-09"))
		nc.Publish("rt.gzb.navl.1.in.cmp.svr.10", []byte("Hello World-10"))
		// Flush 发布缓冲区
		err = nc.Flush()
		if err != nil {
			t.Logf("subscribe flush err: %+v", err)
			t.Fail()
		}
		<-closechan
	}()

	msg, err := sub.NextMsg(300 * time.Millisecond)
	if msg != nil {
		t.Logf("sub:%s,data:%s", msg.Subject, msg.Data)
	} else {
		t.Logf("01,err: %#v,msg:%#v", err, msg)
	}

	msg, err = sub.NextMsg(300 * time.Millisecond)
	if msg != nil {
		t.Logf("sub:%s,data:%s", msg.Subject, msg.Data)
	} else {
		t.Logf("02,err: %#v,msg:%#v", err, msg)
	}

	msg, err = sub.NextMsg(300 * time.Millisecond)
	if msg != nil {
		t.Logf("sub:%s,data:%s", msg.Subject, msg.Data)
	} else {
		t.Logf("03,err: %#v,msg:%#v", err, msg)
	}

	time.Sleep(5 * time.Second)
	close(closechan)
	wg.Wait()
}
