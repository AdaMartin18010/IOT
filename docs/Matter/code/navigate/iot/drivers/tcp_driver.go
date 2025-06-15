package drivers

import (
	"errors"
	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	evchs "navigate/common/model/eventchans"
	"net"
	"os"
	"strings"

	"go.uber.org/zap"
)

var (
	//Verify Satisfies interfaces
	_ cmpt.CptRoot     = (*TcpDriver)(nil)
	_ cmpt.Cpt         = (*TcpDriver)(nil)
	_ cm.WorkerRecover = (*TcpDriver)(nil)
)

type TcpDriver struct {
	*cmpt.CptMetaSt
	evchs.EventChans
	cmpt.Cmder

	ConStats *DriverConnectStatus
	Logger   *zap.Logger //日志分支
}

func NewTcpDriver(kn cmpt.KindName, in cmpt.IdName, log *zap.Logger) *TcpDriver {
	return &TcpDriver{
		CptMetaSt: cmpt.NewCptMetaSt(kn, in),
		ConStats:  NewDriverConnectStatus(),
		Logger:    log,
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
func (rmp *TcpDriver) IsNetWorkErr(err error) bool {
	if ne, ok := err.(*net.OpError); ok {
		var se *os.SyscallError
		if errors.As(ne, &se) {
			ss := strings.ToLower(se.Error())
			if strings.Contains(strings.ToLower(ss), "broken pipe") ||
				strings.Contains(strings.ToLower(ss), "established connection was aborted") ||
				strings.Contains(strings.ToLower(ss), "connection was forcibly closed") {
				rmp.Logger.Sugar().Warnf("connetion close,write err:%+v", ss)
				return true
			} else if strings.Contains(strings.ToLower(ss), "connection reset by peer") ||
				strings.Contains(strings.ToLower(ss), "connection closed by peer") ||
				strings.Contains(strings.ToLower(ss), "eof") {
				rmp.Logger.Sugar().Warnf("connetion close,read err:%+v", ss)
				return true
			} else if strings.Contains(strings.ToLower(ss), "i/o timeout") {
				rmp.Logger.Sugar().Warnf("connetion read timeout err:%+v", ss)
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
