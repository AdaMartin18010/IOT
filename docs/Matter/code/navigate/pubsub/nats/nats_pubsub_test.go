package nats_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	nats "github.com/nats-io/nats.go"
)

func CreateNatsClient(url, user, password, connName string) (*nats.Conn, error) {
	opts := nats.GetDefaultOptions()
	opts.Url = url
	opts.User = user
	opts.Password = password
	// Turn on Verbose
	opts.Verbose = true
	opts.Name = connName
	opts.Pedantic = true
	opts.AllowReconnect = true

	// //会与nats服务的权限设置产生冲突 导致结果不一致
	// //[-ERR Permissions Violation for Subscription to "IotInbox.yO1iLwN1HYKdrgpYF30cYG.*"]
	// //Subscription Violation - User "nav_in", Subject "IotInbox.yO1iLwN1HYKdrgpYF30cYG.*"
	// opts.InboxPrefix = "IotInbox"
	return opts.Connect()
}

func TestPublish(t *testing.T) {
	nc, err := CreateNatsClient(serverIpPort, user_navl_in, user_navl_in_password, "iot_connect")
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}
	defer nc.Drain()
	defer nc.Close()

	err = nc.Publish("rt.gzb.navl.1.in.svc", []byte("Hello World"))
	if err != nil {
		t.Logf("publish err: %#v", err)
		t.Fail()
	}
	// Flush 发布缓冲区
	err = nc.Flush()
	if err != nil {
		t.Logf("publish flush err: %+v", err)
		t.Fail()
	}
}

func TestPublishRequest(t *testing.T) {
	nc, err := CreateNatsClient(serverIpPort, user_navl_in, user_navl_in_password, "iot_connect")
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}
	defer nc.Drain()
	defer nc.Close()

	//todo: check that is not working ?
	err = nc.PublishRequest("rt.gzb.navl.1.in.svc", "reply_string", []byte("Hello World"))
	if err != nil {
		t.Logf("publish err: %#v", err)
		t.Fail()
	}
	// Flush 发布缓冲区
	err = nc.Flush()
	if err != nil {
		t.Logf("publish flush err: %+v", err)
		t.Fail()
	}
}

func TestSubscribe(t *testing.T) {
	nc, err := CreateNatsClient(serverIpPort, user_navl_in, user_navl_in_password, "iot_connect")
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}
	defer nc.Drain()
	defer nc.Close()

	// 第一个消息因为没有订阅者 会被nats_server丢弃
	err = nc.Publish("rt.gzb.navl.1.in.svc", []byte("Hello World-00"))
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

	closechan := make(chan struct{}, 3)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		sub, err := nc.Subscribe("rt.gzb.navl.*.in.>", func(msg *nats.Msg) {
			t.Logf("sub:%s,data:%s", msg.Subject, msg.Data)
			err := msg.Ack()
			if err != nil {
				t.Logf("ACK msg: %#v,err:%#v", msg, err)
			}
			// // 必须要配合 请求和回应的方法
			// //Respond err: &errors.errorString{s:"nats: message does not have a reply"}
			// name := msg.Subject
			// err := msg.Respond([]byte("hello,[" + name + "]"))
			// if err != nil {
			// 	t.Logf("Respond err: %#v", err)
			// }
		})
		t.Logf("%#v", sub)
		if err != nil {
			t.Logf("subscribe err: %#v", err)
			t.Fail()
		}
		<-closechan
		sub.Unsubscribe()
	}()

	// 第一个消息因为没有订阅 所以会收不到
	sub, err := nc.SubscribeSync("rt.gzb.navl.1.in.>")
	if err != nil {
		t.Logf("subscribe err: %+v", err)
		t.Fail()
		return
	} else {
		sub.AutoUnsubscribe(10)
	}
	defer sub.Unsubscribe()

	//第一消息会丢弃 因为没有订阅者
	// subscribe msg: (*nats.Msg)(nil),err:&errors.errorString{s:"nats: timeout"}
	msg, err := sub.NextMsg(3 * time.Millisecond)
	if errors.Is(err, nats.ErrTimeout) {
		t.Logf("subscribe msg: %#v,err:%#v", msg, err)
	} else {
		err := msg.Ack()
		if err != nil {
			t.Logf("ACK msg: %#v,err:%#v", msg, err)
		}
	}

	err = nc.Publish("rt.gzb.navl.1.in.svc.1", []byte("Hello World-01"))
	if err != nil {
		t.Logf("Publish err: %+v", err)
	}
	err = nc.Publish("rt.gzb.navl.1.in.svc.2", []byte("Hello World-02"))
	if err != nil {
		t.Logf("Publish err: %+v", err)
	}
	// Flush 发布缓冲区
	err = nc.Flush()
	if err != nil {
		t.Logf("subscribe flush err: %+v", err)
		//t.Fail()
	}

	msg, err = sub.NextMsg(3 * time.Millisecond)
	if msg != nil {
		t.Logf("subscribe msg,err: %+v,sub:%s,data:%s", err, msg.Subject, msg.Data)
	} else {
		err := msg.Ack()
		if err != nil {
			t.Logf("ACK msg: %#v,err:%#v", msg, err)
		}
	}

	msg, err = sub.NextMsg(3 * time.Millisecond)
	if msg != nil {
		t.Logf("subscribe msg,err: %+v,sub:%s,data:%s", err, msg.Subject, msg.Data)
	} else {
		err := msg.Ack()
		if err != nil {
			t.Logf("ACK msg: %#v,err:%#v", msg, err)
		}
	}

	time.Sleep(6 * time.Second)
	close(closechan)
	wg.Wait()
}
