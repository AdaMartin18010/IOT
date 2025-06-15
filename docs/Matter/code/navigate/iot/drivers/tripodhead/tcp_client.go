package tripodhead

import (
	"context"
	"errors"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	cmpt "navigate/common/model/component"
	dr "navigate/iot/drivers"
	feiyue "navigate/iot/drivers/tripodhead/feiyue"

	"go.uber.org/zap"
)

const (
	TripodheadKind = cmpt.KindName("TripodheadDriver")
)

var (
	//Verify RadarTcpClient satisfied interface-Transporter
	_ dr.Transporter = (*TripodheadDriver)(nil)
)

type TripodheadDriver struct {
	// Connect address string
	Address string
	// Connect & Read & Write timeout
	Timeout time.Duration
	// Idle timeout to close the connection
	IdleTimeout time.Duration

	Ctx    context.Context
	Logger *zap.Logger

	*dr.DriverConnectStatus

	mu           sync.Mutex
	conn         net.Conn
	closeTimer   *time.Timer
	lastActivity time.Time
}

// 连接和 关闭-重连
// Connect and Close are exported so that multiple requests can be done with one session
func (thtc *TripodheadDriver) Connect() error {
	thtc.mu.Lock()
	defer thtc.mu.Unlock()

	return thtc.connect()
}

// Caller must hold the mutex before calling this method.
func (thtc *TripodheadDriver) connect() error {
	if thtc.conn == nil {
		dialer := net.Dialer{Timeout: thtc.Timeout}
		thtc.DriverConnectStatus.SetState(dr.Connecting)
		conn, err := dialer.DialContext(thtc.Ctx, "tcp", thtc.Address)
		if err != nil {
			return err
		}
		thtc.conn = conn
		thtc.DriverConnectStatus.SetState(dr.Connected)
		thtc.Logger.Sugar().Debugf("%s connected.", thtc.Address)
	}
	return nil
}

// Caller must hold the mutex before calling this method.
func (thtc *TripodheadDriver) startCloseTimer() {
	if thtc.IdleTimeout <= 0 {
		return
	}
	if thtc.closeTimer == nil {
		thtc.closeTimer = time.AfterFunc(thtc.IdleTimeout, thtc.closeIdleCaller)
	} else {
		thtc.closeTimer.Reset(thtc.IdleTimeout)
	}
}

// Close closes current connection.
func (thtc *TripodheadDriver) Close() error {
	thtc.mu.Lock()
	defer thtc.mu.Unlock()

	return thtc.close()
}

// readTcpConn flushes pending data in the connection,
// returns io.EOF if connection is closed.
func (thtc *TripodheadDriver) readTcpConn(b []byte) (err error) {
	if err = thtc.conn.SetReadDeadline(time.Now().Add(thtc.Timeout)); err != nil {
		return
	}
	// Timeout setting will be reset when reading
	if _, err = thtc.conn.Read(b); err != nil {
		// Ignore timeout error
		if netError, ok := err.(net.Error); ok && netError.Timeout() {
			thtc.Logger.Sugar().Debugf("Ignore timeout error %+v", err)
			err = nil
		}
	}
	return
}

// closeLocked closes current connection.
// Caller must hold the mutex before calling this method.
func (thtc *TripodheadDriver) close() (err error) {
	if thtc.conn != nil {
		err = thtc.conn.Close()
		thtc.DriverConnectStatus.SetState(dr.DoDisConnected)
		thtc.conn = nil
	}
	return
}

// closeIdleCaller closes the connection if last activity is passed behind IdleTimeout.
func (thtc *TripodheadDriver) closeIdleCaller() {
	thtc.mu.Lock()
	defer thtc.mu.Unlock()

	if thtc.IdleTimeout <= 0 {
		return
	}

	idle := time.Since(thtc.lastActivity)
	if idle >= thtc.IdleTimeout {
		thtc.Logger.Sugar().Debugf("tcpclient: closing connection due to idle timeout: %v", idle)
		thtc.close()
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
func (thtc *TripodheadDriver) IsNetWorkErr(err error) bool {
	if ne, ok := err.(*net.OpError); ok {
		var se *os.SyscallError
		if errors.As(ne, &se) {
			ss := strings.ToLower(se.Error())
			if strings.Contains(strings.ToLower(ss), "broken pipe") ||
				strings.Contains(strings.ToLower(ss), "established connection was aborted") ||
				strings.Contains(strings.ToLower(ss), "connection was forcibly closed") {
				thtc.Logger.Sugar().Warnf("connetion close,write err:%+v", ss)
				return true
			} else if strings.Contains(strings.ToLower(ss), "connection reset by peer") ||
				strings.Contains(strings.ToLower(ss), "connection closed by peer") ||
				strings.Contains(strings.ToLower(ss), "eof") {
				thtc.Logger.Sugar().Warnf("connetion close,read err:%+v", ss)
				return true
			} else if strings.Contains(strings.ToLower(ss), "i/o timeout") {
				thtc.Logger.Sugar().Warnf("connetion read timeout err:%+v", ss)
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

func (thtc *TripodheadDriver) Send(rdd []byte) (err error) {
	thtc.mu.Lock()
	defer thtc.mu.Unlock()

	if err = thtc.connect(); err != nil {
		return
	}

	thtc.lastActivity = time.Now()
	thtc.startCloseTimer()
	var timeout time.Time
	if thtc.Timeout > 0 {
		timeout = thtc.lastActivity.Add(thtc.Timeout)
	}

	if err = thtc.conn.SetDeadline(timeout); err != nil {
		return
	}

	_, err = thtc.conn.Write(rdd)
	if err != nil {
		if errors.Is(err, os.ErrDeadlineExceeded) {
			// TODO: handle error timeout here
			return
		}
		return
	}

	return nil
}

func (thtc *TripodheadDriver) Receive() (rdd []byte, err error) {
	thtc.mu.Lock()
	defer thtc.mu.Unlock()

	if err = thtc.connect(); err != nil {
		return
	}

	thtc.lastActivity = time.Now()
	thtc.startCloseTimer()
	var timeout time.Time
	if thtc.Timeout > 0 {
		timeout = thtc.lastActivity.Add(thtc.Timeout)
	}

	if err = thtc.conn.SetDeadline(timeout); err != nil {
		return
	}

	// Read header
	var data [feiyue.ProtocolDataLen]byte
	// Timeout setting will be reset when reading
	_, err = io.ReadFull(thtc.conn, data[:feiyue.ProtocolDataLen])
	if err != nil {
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
	if _, err = io.ReadFull(thtc.conn, restData[:restLen]); err != nil {
		return
	}
	rdd = append(rdd, data[:]...)
	rdd = append(rdd, restData[:]...)

	return

}
