package nats_test

import (
	"testing"
	"time"

	nats "github.com/nats-io/nats.go"
)

const (
	serverIpPort              = "127.0.0.1:4222"
	user_navl_in              = "navl_in"
	user_navl_in_password     = `%$76#@29*&`
	user_navl_in_sub          = `navl_in_sub`
	user_navl_in_sub_password = `%$3&%$sdh*`
)

func TestConnectUserPassword0(t *testing.T) {
	nc, err := nats.Connect(serverIpPort, nats.UserInfo(user_navl_in, user_navl_in_password))
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
	}
	defer nc.Close()
}

func TestConnectUserPassword1(t *testing.T) {
	opts := nats.GetDefaultOptions()
	opts.Url = serverIpPort
	opts.User = user_navl_in
	//opts.User = "awds"
	opts.Password = user_navl_in_password
	// Turn on Verbose
	opts.Verbose = true
	opts.Name = "iot_connect"
	opts.Pedantic = true

	// Turn on Reconnect
	opts.AllowReconnect = true
	// Reconnect forever
	opts.MaxReconnect = -1
	// Reconnect wait time defaults 2s
	opts.ReconnectWait = time.Second

	// 如果两次认证错误 则不再尝试重新连接
	opts.IgnoreAuthErrorAbort = true
	opts.ClosedCB = func(cn *nats.Conn) {
		t.Logf("ClosedCB conn: %#v", cn)
	}

	opts.AsyncErrorCB = func(cn *nats.Conn, sub *nats.Subscription, err error) {
		t.Logf("AsyncErrorCB conn: %#v,sub:%#v,err:%#v", cn, sub, err)
	}

	opts.ConnectedCB = func(cn *nats.Conn) {
		t.Logf("ConnectedCB conn: %#v", cn)
	}

	opts.CustomReconnectDelayCB = func(attempts int) time.Duration {
		if attempts > 0 {
			return time.Duration(time.Duration(attempts+1) * time.Second)
		}
		return time.Second
	}

	opts.DisconnectedCB = func(cn *nats.Conn) {
		t.Logf("DisconnectedCB conn: %#v", cn)
	}

	opts.DisconnectedErrCB = func(cn *nats.Conn, err error) {
		t.Logf("DisconnectedErrCB conn: %#v,err: %#v", cn, err)
	}

	opts.DiscoveredServersCB = func(cn *nats.Conn) {
		t.Logf("DiscoveredServersCB conn: %#v", cn)
	}

	// DrainTimeout sets the timeout for a Drain Operation to complete.
	// Defaults to 30s.
	opts.DrainTimeout = 10 * time.Second
	// FlusherTimeout is the maximum time to wait for write operations
	// to the underlying connection to complete (including the flusher loop).
	// 默认是阻塞写
	opts.FlusherTimeout = 10 * time.Second
	// PingInterval is the period at which the client will be sending ping
	// commands to the server, disabled if 0 or negative.
	// Defaults to 2m.
	// 心跳包的时间间隔 默认为2分钟
	opts.PingInterval = 10 * time.Second
	// MaxPingsOut is the maximum number of pending ping commands that can
	// be awaiting a response before raising an ErrStaleConnection error.
	// Defaults to 2.
	opts.MaxPingsOut = 5

	// ReconnectBufSize is the size of the backing bufio during reconnect.
	// Once this has been exhausted publish operations will return an error.
	// Defaults to 8388608 bytes (8MB).
	//重新连接时的客户端缓存
	opts.ReconnectBufSize = 16 * 1024 * 1024

	// // for request response temporary topic name
	// // 会与nats服务的权限设置产生冲突 导致结果不一致
	// // [-ERR Permissions Violation for Subscription to "IotInbox.yO1iLwN1HYKdrgpYF30cYG.*"]
	// // Subscription Violation - User "nav_in", Subject "IotInbox.yO1iLwN1HYKdrgpYF30cYG.*"
	// opts.InboxPrefix = "IotInbox"

	// // for websocket connection
	// opts.Compression = true
	//opts.ProxyPath = ""
	nc, err := opts.Connect()
	if err != nil {
		t.Logf("connect err: %#v", err)
		t.Fail()
		return
	}
	t.Logf("connect Statistics: %#v", nc.Statistics)
	time.Sleep(20 * time.Second)
	nc.Close()
}

/*
设置心跳时间间隔后的效果
[14120] 2023/03/26 16:39:15.916360 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PING]
[14120] 2023/03/26 16:39:15.916360 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PONG]
[14120] 2023/03/26 16:39:18.217260 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PING]
[14120] 2023/03/26 16:39:18.217260 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PONG]
[14120] 2023/03/26 16:39:23.218059 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PING]
[14120] 2023/03/26 16:39:23.218059 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PONG]
[14120] 2023/03/26 16:39:25.916854 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PING]
[14120] 2023/03/26 16:39:25.916854 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PONG]
[14120] 2023/03/26 16:39:33.220278 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PING]
[14120] 2023/03/26 16:39:33.220278 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PONG]
[14120] 2023/03/26 16:39:35.917553 [TRC] 127.0.0.1:10183 - cid:45 - <<- [PING]
[14120] 2023/03/26 16:39:35.917553 [TRC] 127.0.0.1:10183 - cid:45 - ->> [PONG]
*/
