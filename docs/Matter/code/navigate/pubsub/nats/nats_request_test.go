package nats_test

import (
	"sync"
	"testing"
	"time"

	nats "github.com/nats-io/nats.go"
)

func TestRequest(t *testing.T) {
	nc, err := CreateNatsClient(serverIpPort, user_navl_in, user_navl_in_password, "iot_connect")
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}

	defer func() {
		err := nc.Drain()
		if err != nil {
			t.Logf("nc.Drain() err: %+v", err)
		}
		nc.Close()
	}()

	closechan := make(chan struct{}, 3)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		// //err:&errors.errorString{s:"nats: no responders available for request"}
		// //通配表达式 可能造成未定义的交互错误 请求必须明确不可使用通配表达式
		// subject := "rt.gzb.navl.*.in.>"
		subject := "rt.gzb.navl.1.in.01"
		sub, err := nc.Subscribe(subject, func(msg *nats.Msg) {
			name := msg.Subject
			t.Logf("response sub:%s,data:%s", msg.Subject, msg.Data)
			err := msg.Respond([]byte("hello,[" + name + "]"))
			if err != nil {
				t.Logf("Respond err: %#v", err)
			}
		})
		t.Logf("%#v", sub)
		if err != nil {
			t.Logf("subscribe err: %#v", err)
			t.Fail()
		}
		<-closechan
		sub.Unsubscribe()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		rep, err := nc.Request("rt.gzb.navl.1.in.02", []byte("Hello-rt.gzb.navl.1.in.02"), 5*time.Second)
		if err != nil {
			t.Logf("rep: %#v,err:%#v", rep, err)
		} else {
			t.Logf("response  err: %#v,sub:%s,data:%s,Reply:%#v", err, rep.Subject, rep.Data, rep.Reply)
		}
		<-closechan
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		rep, err := nc.Request("rt.gzb.navl.1.in.02", []byte("Hello-rt.gzb.navl.1.in.02"), 5*time.Second)
		if err != nil {
			t.Logf("rep: %#v,err:%#v", rep, err)
		} else {
			t.Logf("response  err: %#v,sub:%s,data:%s,Reply:%#v", err, rep.Subject, rep.Data, rep.Reply)
		}
		<-closechan
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		rep, err := nc.Request("rt.gzb.navl.1.in.02", []byte("Hello-rt.gzb.navl.1.in.02"), 5*time.Second)
		if err != nil {
			t.Logf("rep: %#v,err:%#v", rep, err)
		} else {
			t.Logf("response  err: %#v,sub:%s,data:%s,Reply:%#v", err, rep.Subject, rep.Data, rep.Reply)
		}
		<-closechan
	}()
	time.Sleep(10 * time.Second)
	close(closechan)
	wg.Wait()
}
