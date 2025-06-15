package radar

import (
	"context"
	"errors"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	drs "navigate/iot/drivers"
	protocol "navigate/iot/drivers/radar/v220828"

	"go.uber.org/zap"
)

var (
	//Verify RadarTcpClient satisfied interface-Transporter
	_ drs.Transporter = (*RadarTcpClient)(nil)
)

type RadarTcpClient struct {
	// Connect address string
	Address string
	// Connect & Read timeout
	Timeout time.Duration
	// Idle timeout to close the connection
	IdleTimeout time.Duration

	Ctx    context.Context
	Logger *zap.Logger

	*drs.DriverConnectStatus

	mu           sync.Mutex
	conn         net.Conn
	closeTimer   *time.Timer
	lastActivity time.Time
}

// 连接和 关闭-重连
// Connect and Close are exported so that multiple requests can be done with one session
func (rtc *RadarTcpClient) Connect() error {
	rtc.mu.Lock()
	defer rtc.mu.Unlock()

	return rtc.connect()
}

// Caller must hold the mutex before calling this method.
func (rtc *RadarTcpClient) connect() error {
	if rtc.conn == nil {
		dialer := net.Dialer{Timeout: rtc.Timeout}
		rtc.DriverConnectStatus.SetState(drs.Connecting)
		conn, err := dialer.DialContext(rtc.Ctx, "tcp", rtc.Address)
		if err != nil {
			return err
		}
		rtc.conn = conn
		rtc.DriverConnectStatus.SetState(drs.Connected)
		rtc.Logger.Sugar().Debugf("%s connected.", rtc.Address)
	}
	return nil
}

// Caller must hold the mutex before calling this method.
func (rtc *RadarTcpClient) startCloseTimer() {
	if rtc.IdleTimeout <= 0 {
		return
	}
	if rtc.closeTimer == nil {
		rtc.closeTimer = time.AfterFunc(rtc.IdleTimeout, rtc.closeIdleCaller)
	} else {
		rtc.closeTimer.Reset(rtc.IdleTimeout)
	}
}

// Close closes current connection.
func (rtc *RadarTcpClient) Close() error {
	rtc.mu.Lock()
	defer rtc.mu.Unlock()

	return rtc.close()
}

// closeLocked closes current connection.
// Caller must hold the mutex before calling this method.
func (rtc *RadarTcpClient) close() (err error) {
	if rtc.conn != nil {
		err = rtc.conn.Close()
		rtc.DriverConnectStatus.SetState(drs.DoDisConnected)
		rtc.conn = nil
	}
	return
}

// closeIdleCaller closes the connection if last activity is passed behind IdleTimeout.
func (rtc *RadarTcpClient) closeIdleCaller() {
	rtc.mu.Lock()
	defer rtc.mu.Unlock()

	if rtc.IdleTimeout <= 0 {
		return
	}

	idle := time.Since(rtc.lastActivity)
	if idle >= rtc.IdleTimeout {
		rtc.Logger.Sugar().Debugf("tcpclient: closing connection due to idle timeout: %v", idle)
		rtc.close()
	}
}

/*
	  远程主机主动关闭了socket连接,而本地没有关闭且在继续send,就会导致此类问题.
	  windows:
		errno: 10053
			wsasend: An established connection was aborted by the software in your host machine.
			wsasend: An existing connection was forcibly closed by the remote host
	  linux:
	    errno: syscall.EPIPE
			write: broken pipe

	  远程主机主动关闭了socket连接,而本地没有关闭且在继续recv,就会导致此类问题.
	  windows:
	  errno: 10054
	     wsarecv: Connection closed by peer.
	  linux:
		errno: syscall.ECONNRESET
		read: connection reset by peer
	  或者 golang 本身转换为 errno: eof.
*/
func (rtc *RadarTcpClient) IsNetWorkErr(err error) (closed bool) {
	if ne, ok := err.(*net.OpError); ok {
		var se *os.SyscallError
		if errors.As(ne, &se) {
			ss := strings.ToLower(se.Error())
			if strings.Contains(ss, "broken pipe") ||
				strings.Contains(ss, "established connection was aborted") ||
				strings.Contains(ss, "connection was forcibly closed") {
				rtc.Logger.Sugar().Warnf("connetion close,write err:%+v", ss)
				return true
			} else if strings.Contains(ss, "connection reset by peer") ||
				strings.Contains(ss, "connection closed by peer") ||
				strings.Contains(ss, "eof") {
				rtc.Logger.Sugar().Warnf("connetion close,read err:%+v", ss)
				return true
			} else if strings.Contains(ss, "i/o timeout") {
				rtc.Logger.Sugar().Warnf("connetion read timeout err:%+v", ss)
				return false
			} else {
				return false
			}
		} else {
			return false
		}
	} else {
		return false
	}
}

func (rtc *RadarTcpClient) Receive() (rdd []byte, err error) {
	rtc.mu.Lock()
	defer rtc.mu.Unlock()

	if err = rtc.connect(); err != nil {
		return
	}

	rtc.lastActivity = time.Now()
	rtc.startCloseTimer()
	var timeout time.Time
	if rtc.Timeout > 0 {
		timeout = rtc.lastActivity.Add(rtc.Timeout)
	}

	if err = rtc.conn.SetDeadline(timeout); err != nil {
		return
	}

	// Read header
	var data [protocol.ProtocolDataMinLen]byte
	if _, err = io.ReadFull(rtc.conn, data[:protocol.ProtocolDataMinLen]); err != nil {
		return
	}

	if data[4] == 0x00 {
		rdd = append(rdd, data[:]...)
		return
	}

	restLen := 0
	if data[4] >= 0x01 {
		restLen = int(data[4])*10 - 1 + 4 + 1
	}

	restData := make([]byte, restLen)
	if _, err = io.ReadFull(rtc.conn, restData[:restLen]); err != nil {
		return
	}
	rdd = append(rdd, data[:]...)
	rdd = append(rdd, restData[:]...)

	return
}
