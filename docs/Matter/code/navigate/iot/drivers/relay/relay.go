package relay

import (
	"context"
	"errors"
	"fmt"
	"time"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	tk "navigate/common/model/token"
	bs "navigate/internal/bytestr"
	drs "navigate/iot/drivers"
	zlan6842 "navigate/iot/drivers/relay/zlan6842"

	"github.com/goburrow/modbus"
	"go.uber.org/zap"
)

const (
	RelayKind = cmpt.KindName("Relay")
)

// Make sure that RelayModbusProcess is Worker interface
var _ cm.WorkerRecover = (*Relay)(nil)

type RelayCommand struct {
	*tk.BaseToken
}

func NewRelayCommond() *RelayCommand {
	return &RelayCommand{
		BaseToken: tk.NewBaseToken(),
	}
}

type Relay struct {
	*drs.TcpDriver
	*RelayConfig
	*drs.DriverConnectStatus

	handler    modbus.Client
	tcphandler *modbus.TCPClientHandler
	safeToExit bool
}

func NewRelay(rlcnf *RelayConfig, in cmpt.IdName, log *zap.Logger) (rmp *Relay) {
	rmp = &Relay{
		TcpDriver:           drs.NewTcpDriver(RelayKind, in, log),
		RelayConfig:         rlcnf,
		DriverConnectStatus: drs.NewDriverConnectStatus(),
		safeToExit:          true,
	}
	return
}

func (rld *Relay) GetDOs() (dosStatus zlan6842.DOitems, err error) {
	// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
	results, err := rld.handler.ReadCoils(zlan6842.ZlanDOaddr, zlan6842.ZlanDOitemsMax)
	if err != nil {
		rld.Logger.Sugar().Errorf("WriteSingleCoil err:%+v", err)
		if rld.IsNetWorkErr(err) {
			rld.SetState(drs.BeDisConnected)
		}
		return nil, err
	}

	dosStatus = make(zlan6842.DOitems, zlan6842.ZlanDOitemsMax)
	var indexOfDO uint16
	// DO id begin with 1+
	for indexOfDO = 1; indexOfDO <= zlan6842.ZlanDOitemsMax; indexOfDO++ {
		dosStatus[indexOfDO-1].Id = indexOfDO
		//屏蔽掉其他位
		bit := 0x01 << (indexOfDO - 1)
		res := results[0] & byte(bit)
		//判断当前位
		res = res & byte(bit)
		if res != 0x00 {
			dosStatus[indexOfDO-1].On = true
		} else {
			dosStatus[indexOfDO-1].On = false
		}
	}
	return
}

func (rld *Relay) SetDOs(dos zlan6842.DOitems) error {
	if len(dos) <= 0 {
		return errors.New("DOitems length <=0")
	}
	var switchOnOff uint16
	noErr := true
	errStr := ""
	for _, do := range dos {
		if do.Id > zlan6842.ZlanDOitemsMax || do.Id < zlan6842.ZlanDOindex {
			return fmt.Errorf("DOitems ID:(no valid) %+v", do)
		}
		iDO := zlan6842.ZlanDOaddr + do.Id - 1
		// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
		// 设置开关:{on:0xFF00,off:0x0000}
		if do.On {
			switchOnOff = zlan6842.ZlanSwitchOn
		} else {
			switchOnOff = zlan6842.ZlanSwitchOff
		}

		//cmd=0x05
		_, err := rld.handler.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			noErr = false
			rld.Logger.Sugar().Errorf("WriteSingleCoil err:%+v", err)
			if rld.IsNetWorkErr(err) {
				rld.SetState(drs.BeDisConnected)
				return err
			} else {
				errStr += fmt.Sprintf("%+v\n", err)
			}
		}
	}

	if noErr {
		return nil
	} else {
		return fmt.Errorf("%s", errStr)
	}

}

func (rld *Relay) SetDOsAll(onOroff bool) error {
	var indexOfDO, switchOnOff uint16
	indexOfDO = 1
	if onOroff {
		switchOnOff = zlan6842.ZlanSwitchOn
	} else {
		switchOnOff = zlan6842.ZlanSwitchOff
	}

	noErr := true
	errStr := ""
	for ; indexOfDO <= zlan6842.ZlanDOitemsMax; indexOfDO++ {
		iDO := 0x0010 + indexOfDO - 1
		// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
		// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
		_, err := rld.handler.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			noErr = false
			rld.Logger.Sugar().Errorf("WriteSingleCoil err:%+v", err)
			if rld.IsNetWorkErr(err) {
				rld.SetState(drs.BeDisConnected)
				return err
			} else {
				errStr += fmt.Sprintf("%+v\n", err)
			}
		}
	}

	if noErr {
		return nil
	} else {
		return fmt.Errorf("%s", errStr)
	}

}

func (rld *Relay) Recover() {
	if rc := recover(); rc != nil {
		rld.Logger.Sugar().Warnf("RelayDriver recover: %v", rc)
	}
	rld.Logger.Sugar().Info("RelayDriver stopped")
}

func (rld *Relay) Work() error {
	if rld.Logger == nil {
		panic(fmt.Errorf("logger is nil"))
	}

	if rld.RelayConfig == nil {
		rld.Logger.Sugar().Fatal("RelayDriver RelayConfig is nil! Exit.")
	}
	rld.Logger.Sugar().Debugf(`"RelayDriver Starting autotest:%t"`, rld.EnableAutoTest)

	//根据 配置文件初始化一次
	iPport := fmt.Sprintf("%s:%d", rld.RelayConfig.Ipaddress, rld.RelayConfig.Port)
	rld.tcphandler = modbus.NewTCPClientHandler(iPport)
	rld.tcphandler.Timeout = time.Duration(rld.RelayConfig.ConnectReadTimeout) * time.Millisecond
	rld.tcphandler.IdleTimeout = time.Duration(rld.RelayConfig.IdleTimeout) * time.Second
	rld.tcphandler.SlaveId = byte(rld.RelayConfig.SlaveId)

	//使用接口 创建一次
	rld.handler = modbus.NewClient(rld.tcphandler)

	//主程序通知退出
	NotifyToExit := false
	//一旦有开关量设置 就设置为false, 所有都关闭的情况下设置为true.
	rld.safeToExit = true
	for {
		if NotifyToExit {
			if rld.safeToExit {
				if rld.GetState() == drs.Connected {
					rld.CheckSafeExit(true)
				}

				return nil
			} else {
				rld.Logger.Sugar().Errorf("Is NoSafe To Exit: SafeToExit:%t - Wait to Close all switchs.", rld.safeToExit)
				if rld.GetState() == drs.Connected {
					rld.CheckSafeExit(true)
				}
				continue
			}
		}

		select {
		case <-rld.Ctrl().Context().Done():
			{
				//退出通知
				NotifyToExit = true
				if err := rld.Ctrl().Context().Err(); err != context.Canceled {
					rld.Logger.Sugar().Warnf("err : %+v\n", err)
				}
				continue
			}
		default:
			{
				//初始化或者断开后的连接
				if rld.GetState() != drs.Connected {
					rld.SetState(drs.Connecting)
					err := rld.tcphandler.Connect()
					if err != nil {
						rld.Logger.Sugar().Warnf("Connect failed: %s , %+v", rld.tcphandler.Address, err)
					} else {
						rld.SetState(drs.Connected)
						rld.Logger.Sugar().Infof("%s -Connected.", rld.tcphandler.Address)
					}
				} else if rld.GetState() == drs.Connected {
					// todo: 测试和检查所有的开关通讯和设置
					//rmp.TestSwitchOnOff()
					//rmp.ScheduleOnNavlock()
					//检查是否可以安全退出
					rld.safeToExit = rld.CheckSafeExit(false)
				}
			}
		}
		time.Sleep(time.Duration(rld.ReConnectRetryTimeInterval) * time.Second)
	}
}

type RelayInfo struct {
	DriveName  string
	RelayArray string
}

func (r *RelayInfo) String() string {
	buf, _ := cm.Json.Marshal(r)
	return bs.BytesToString(buf)
}

func (rld *Relay) TestSwitchOnOff() {
	//连接状态下 执行相应的操作
	//1. 先获取所有状态入库
	_, err := rld.GetDOs()
	if err != nil {
		rld.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
	}

	//2. 再打开所有的开关状态
	err = rld.SetDOsAll(true)
	if err != nil {
		rld.Logger.Sugar().Errorf("Set Dos all true,err:%+v", err)
	} else {
		rld.safeToExit = false
	}

	//3. 再获取所有状态入库
	_, err = rld.GetDOs()
	if err != nil {
		rld.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
	}

	//4. 再关闭所有的开关状态
	err = rld.SetDOsAll(false)
	if err != nil {
		rld.Logger.Sugar().Errorf("Set Dos all true,err:%+v", err)
	} else {
		rld.safeToExit = true
	}

	//5. 再获取所有状态入库
	_, err = rld.GetDOs()
	if err != nil {
		rld.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
	}

}

func (rld *Relay) CheckSafeExit(ifTurnOffAll bool) bool {
	if ifTurnOffAll {
		err := rld.SetDOsAll(false)
		if err != nil {
			rld.Logger.Sugar().Errorf("Set Dos all false,err:%+v", err)
			return false
		} else {
			rld.safeToExit = true
			return true
		}
	}

	//获取所有状态判断
	dos, err := rld.GetDOs()
	if err != nil {
		//todo:
		rld.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
		return false
	} else {
		//如果全部关闭
		if dos.IsAll(false) {
			return true
		} else {
			return false
		}
	}
}

// 开启
func (rld *Relay) Switch(id uint16, onOrOff bool) bool {
	dos := zlan6842.DOitems{
		zlan6842.DOitem{Id: id, On: onOrOff},
	}

	err := rld.SetDOs(dos)
	if err != nil {
		rld.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
		return false
	}
	return true
}
