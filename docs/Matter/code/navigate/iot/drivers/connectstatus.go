package drivers

import (
	cm "navigate/common/model"
	"sync"
	"time"
)

const (
	Connecting     ConnectStatus = 1
	Connected      ConnectStatus = 2
	DoDisConnected ConnectStatus = 3
	BeDisConnected ConnectStatus = 4
)

var (
	//驱动的连接状态对照map
	ConnectStatusStr = map[ConnectStatus]string{
		0: "Unknown",
		1: "Connecting",
		2: "Connected",
		3: "DoDisConnected",
		4: "BeDisConnected",
	}

	//驱动的连接状态中文对照map
	ConnectStatusStrCN = map[ConnectStatus]string{
		0: "未知",
		1: "连接中",
		2: "连接上",
		3: "主动连接断开",
		4: "被动断开连接",
	}
)

type ConnectStatus uint

func (c ConnectStatus) String() string {
	return ConnectStatusStrCN[c]
}

// 抽离出驱动的连接状态的通用功能封装
type DriverConnectStatus struct {
	ConnectState         ConnectStatus `json:",string"` // 当前连接状态
	ReConnectCount       uint          ``               // 主动重连 重试的次数不包括连接上
	ConnectedCount       uint          ``               // 连接上的次数
	BeDisConnectedCount  uint          ``               // 断开 被动断开的次数
	DoDisConnectedCount  uint          ``               // 断开 主动断开的次数
	DisConnectedLasttime time.Time     `json:""`        // 上次 即 最近一次断开的时间 包括主动断开
	ConnectedLasttime    time.Time     `json:""`        // 上次 即 最近一次建立连接的时间

	rwlocker *sync.RWMutex `json:"-"`
}

func NewDriverConnectStatus() *DriverConnectStatus {
	return &DriverConnectStatus{
		rwlocker: &sync.RWMutex{},
	}
}

func (dcs *DriverConnectStatus) SetState(cs ConnectStatus) {
	dcs.rwlocker.Lock()
	defer dcs.rwlocker.Unlock()

	switch cs {
	case Connecting:
		{
			dcs.ReConnectCount++
			dcs.ConnectState = cs
		}
	case Connected:
		{
			dcs.ConnectedCount++
			dcs.ConnectState = cs
			dcs.ConnectedLasttime = time.Now()
		}
	case DoDisConnected:
		{
			dcs.DoDisConnectedCount++
			dcs.ConnectState = cs
			dcs.DisConnectedLasttime = time.Now()
		}
	case BeDisConnected:
		{
			dcs.BeDisConnectedCount++
			dcs.ConnectState = cs
			dcs.DisConnectedLasttime = time.Now()
		}
	default:
		dcs.ConnectState = 0
	}
}

func (dcs *DriverConnectStatus) GetState() ConnectStatus {
	dcs.rwlocker.RLock()
	defer dcs.rwlocker.RUnlock()
	return dcs.ConnectState
}

func (dcs *DriverConnectStatus) String() string {
	sbuf, _ := cm.Json.Marshal(dcs)
	return string(sbuf)
}
