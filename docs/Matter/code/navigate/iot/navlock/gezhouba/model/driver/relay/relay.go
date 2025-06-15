package relay

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	cmd_token "navigate/common/model/token"
	iot_cf "navigate/config/iot/drivers"
	bs "navigate/internal/bytestr"
	drs "navigate/iot/drivers"
	zlan6842 "navigate/iot/drivers/relay/zlan6842"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
	db "navigate/store/iot/navlock/model"

	"github.com/martin-ly/modbus"
	"gorm.io/gorm"
)

const (
	RelayDriverKind = cmpt.KindName("relay")

	cmdbufsize       = 10240
	cmdWaitInterval  = 1000 * time.Millisecond
	checkCmdInterval = 1000 * time.Millisecond
)

var (
	//Verify Satisfies interfaces
	_ mdl.WorkerRecover = (*Relay)(nil)
	_ cmpt.Cpt          = (*Relay)(nil)
	_ cmpt.CptsOperator = (*Relay)(nil)
	_ cmpt.CptComposite = (*Relay)(nil)
	_ cmpt.CptRoot      = (*Relay)(nil)
)

type RelayCmd struct {
	*cmd_token.BaseToken
	SetRadarON      bool
	SetTripodheadON bool
}

func NewRelayCmd() *RelayCmd {
	return &RelayCmd{
		BaseToken:       cmd_token.NewBaseToken(),
		SetRadarON:      false,
		SetTripodheadON: false,
	}
}

type RelayInfo struct {
	DriveName  string
	RelayArray string
}

func (r *RelayInfo) JsonData() []byte {
	buf, _ := mdl.Json.Marshal(r)
	return buf
}

func (r *RelayInfo) String() string {
	buf, _ := mdl.Json.Marshal(r)
	return bs.BytesToString(buf)
}

type Relay struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	*drs.DriverConnectStatus
	Cnf     *iot_cf.RelayConfig
	cmdChan chan cmd_token.TokenCompletor

	radarIsSetOnOrOff   bool
	radarNeedSetCmd     bool
	radarCurrentOnOrOff bool
	stateMux            *sync.Mutex

	wkd        *drs_wd.Work
	client     modbus.Client
	tcphandler *modbus.TCPClientHandler
	safeToExit bool
}

func NewRelayDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Relay {
	tmpMnd := &Relay{
		BaseCptCtx:          dr,
		TrackingSt:          trs,
		DriverConnectStatus: drs.NewDriverConnectStatus(),
		radarIsSetOnOrOff:   false,
		radarCurrentOnOrOff: false,
		radarNeedSetCmd:     false,
		stateMux:            &sync.Mutex{},
		safeToExit:          true,
	}
	tmpMnd.cmdChan = make(chan cmd_token.TokenCompletor, cmdbufsize)
	tmpMnd.WorkerRecover = tmpMnd
	return tmpMnd
}

func (rl *Relay) Start() (err error) {
	if rl.CptSt == nil || rl.ModelSysSt == nil {
		return fmt.Errorf("is not initialized.%s", rl.CmptInfo())
	} else {
		if err = rl.CptSt.Validate(); err != nil {
			return err
		}
	}

	if rl.State.Load().(bool) {
		return fmt.Errorf("is already Started.%s", rl.Info())
	}

	if rl.Cnf.EnableDebug {
		rl.Logger.Sugar().Debugln("--- Starting ---")
	}

	//------------- worker -----------------
	// 只有RadarDevice初始化完成后 才能取到合适的参数 所以在Start()里面实现
	cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
		cmpt.IdName("Relay-work-dvr"),
		0,
		rl.Ctrl().ForkCtxWg())

	dr := navl_md.NewBaseCptCtx(cs,
		navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
			rl.TrackingSt.NavlockIdentifier),
			"worker",
			"",
			"Relay-work-dvr"))

	workdr := drs_wd.NewWorkDvr(dr, rl.TrackingSt)
	workdr.Fn = rl.WorkDriverFn
	rl.wkd = workdr
	rl.wkd.EnableDebug = rl.Cnf.EnableDebug
	rl.Cpts.AddCpts(rl.wkd)
	// --------------------------------------------

	//根据 配置文件初始化一次
	iPport := fmt.Sprintf("%s:%d", rl.Cnf.Ipaddress, rl.Cnf.Port)
	rl.tcphandler = modbus.NewTCPClientHandler(iPport)
	rl.tcphandler.Timeout = time.Duration(rl.Cnf.ConnectReadTimeout) * time.Millisecond
	rl.tcphandler.IdleTimeout = time.Duration(rl.Cnf.IdleTimeout) * time.Second
	rl.tcphandler.SlaveId = byte(rl.Cnf.SlaveId)
	rl.tcphandler.WithCtx(rl.Ctrl().Context())

	//使用接口 创建一次
	rl.client = modbus.NewClient(rl.tcphandler)

	rl.Ctrl().WaitGroup().StartingWait(rl.WorkerRecover)
	if rl.Cpts != nil && rl.Cpts.Len() > 0 {
		err = rl.Cpts.Start()
		if err != nil {
			rl.Logger.Sugar().Errorf("Start() error : %+v,%s", err, rl.Info())
		}
	}

	rl.State.Store(true)
	if rl.Cnf.EnableDebug {
		rl.Logger.Sugar().Debugf("Ctrl_info: %s ", rl.Ctrl().DebugInfo())
		rl.Logger.Sugar().Debugf("--- Started ---.%s", rl.Info())
	}
	return err
}

func (rl *Relay) Stop() (err error) {
	if !rl.State.Load().(bool) {
		return fmt.Errorf("is already Stopped.%s", rl.Info())
	}

	rl.Ctrl().Cancel()
	if rl.Cnf.EnableDebug {
		rl.Logger.Sugar().Debugf("--- Stopping ---.%s", rl.Info())
		rl.Logger.Sugar().Debugf("Ctrl_info: %s ", rl.Ctrl().DebugInfo())
		defer rl.Logger.Sugar().Debugf("--- Stopped ---.%s", rl.Info())
	}

	<-rl.Ctrl().Context().Done()
	if rl.Cpts != nil && rl.Cpts.Len() > 0 {
		err = rl.Cpts.Stop()
		if err != nil {
			rl.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	rl.State.Store(false)
	return
}

func (rl *Relay) RadarCurrentOn() (connectd, radarCurrentOn bool) {
	rl.stateMux.Lock()
	defer rl.stateMux.Unlock()
	return (rl.GetState() == drs.Connected), rl.radarCurrentOnOrOff
}

// 外部的输入的命令
func (rl *Relay) Commander(cmd cmd_token.TokenCompletor) bool {
	if !rl.State.Load().(bool) {
		cmd.SetErr(fmt.Errorf("is exited.%s", rl.Info()))
		return false
	}

	timer := mdl.TimerPool.Get(cmdWaitInterval)
	defer mdl.TimerPool.Put(timer)
	for {
		select {
		case <-rl.Ctrl().Context().Done():
			return false
		case rl.cmdChan <- cmd:
			if rl.Cnf.EnableDebug {
				rl.Logger.Sugar().Debugf("get cmd_token - ok.cmd chan cap:%d.%s", cap(rl.cmdChan), rl.Info())
			}
			return true
		case <-timer.C:
			if rl.Cnf.EnableDebug {
				rl.Logger.Sugar().Debugf("get cmd_token - time out.cmd chan cap:%d.%s", cap(rl.cmdChan), rl.Info())
			}
			return false
		}
	}
}

func (rl *Relay) Recover() {
	if rc := recover(); rc != nil {
		var buf [4096]byte
		n := runtime.Stack(buf[:], false)
		rl.Logger.Sugar().Warnf(`Work recover :%+v,stack trace: %s`, rc, buf[:n])
	}
}

func (rl *Relay) Work() error {
	if rl.Cnf.EnableDebug {
		rl.Logger.Sugar().Debugf(`"Work Starting autotest:%t.%s"`,
			rl.Cnf.EnableAutoTest, rl.Info())
	}

	if rl.wkd == nil {
		rl.Logger.Sugar().Fatalf("relay Worker is nil.%s", rl.Info())
	}
	rl.wkd.Start()

	tickerCheckCmd := time.NewTicker(checkCmdInterval)
	defer tickerCheckCmd.Stop()
	tickerCheckCmd.Stop()
	select {
	case <-tickerCheckCmd.C:
	default:
	}
	tickerCheckCmdOn := false

	reConnectedInterval := time.Duration(rl.Cnf.ReConnectRetryTimeInterval) * time.Second
	tickerConnected := time.NewTicker(reConnectedInterval)
	defer tickerConnected.Stop()
	//连接上以后就启动检查命令 否则就取消检查
	if err := rl.checkRelayConnected(); err != nil {
		//用户取消的情况下
		if errors.Is(err, context.Canceled) {
			return nil
		}
		rl.Logger.Sugar().Errorf("connecting err %+v.%s", err.Error(), rl.Info())
	}

	for {
		select {
		case <-rl.Ctrl().Context().Done():
			{
				if err := rl.Ctrl().Context().Err(); err != context.Canceled {
					rl.Logger.Sugar().Warnf("err : %+v\n", err)
				}

				if !rl.safeToExit {
					if rl.GetState() == drs.Connected {
						rl.Logger.Sugar().Infof("Checking Relay state is Safety to exit.%s", rl.Info())
						rl.safeToExit = rl.checkSafeExit(true)
						if rl.Cnf.EnableAutoTest && rl.LDB != nil {
							rl.saveToLDB()
						}
						rl.tcphandler.Close()
						rl.SetState(drs.DoDisConnected)
					}

					if !rl.safeToExit {
						rl.Logger.Sugar().Warnln("*****Relay state is No-Safe to exit,Check Out this!*****")
					}
				}
				return nil
			}
		case <-tickerCheckCmd.C:
			{
				// drain time event
				select {
				case <-tickerCheckCmd.C:
				default:
				}

				//只有连接成功的情况下 才会执行的流程
				rl.stateMux.Lock()
				rl.radarNeedSetCmd = (rl.radarCurrentOnOrOff != rl.radarIsSetOnOrOff)
				radarNeedSetCmd := rl.radarNeedSetCmd
				rl.stateMux.Unlock()

				if !radarNeedSetCmd {
					continue
				}

				rl.stateMux.Lock()
				radarSetOn := rl.radarIsSetOnOrOff
				rl.stateMux.Unlock()

				if err := rl.setRelayRadar(radarSetOn); err != nil {
					//用户取消的情况下
					if errors.Is(err, context.Canceled) {
						tickerConnected.Stop()
						select {
						case <-tickerConnected.C:
						default:
						}

						if tickerCheckCmdOn {
							tickerCheckCmd.Stop()
							select {
							case <-tickerCheckCmd.C:
							default:
							}
							tickerCheckCmdOn = false
						}

						continue
					}

					rl.Logger.Sugar().Errorf("cmd Set Radar: %t, err: %+v.%s", radarSetOn, err, rl.Info())
				} else {
					if rl.Cnf.EnableDebug {
						rl.Logger.Sugar().Debugf("cmd Set Radar: %t,Ok .%s", radarSetOn, rl.Info())
					}
				}
			}
		case <-tickerConnected.C:
			{
				// drain time event
				select {
				case <-tickerConnected.C:
				default:
				}

				//连接上以后就启动检查命令 否则就取消检查
				if err := rl.checkRelayConnected(); err != nil {
					//用户取消的情况下
					if errors.Is(err, context.Canceled) {
						tickerConnected.Stop()
						select {
						case <-tickerConnected.C:
						default:
						}

						if tickerCheckCmdOn {
							tickerCheckCmd.Stop()
							select {
							case <-tickerCheckCmd.C:
							default:
							}
							tickerCheckCmdOn = false
						}

						continue
					}
				}

				if rl.GetState() == drs.Connected {
					//连接的情况下 如果未启动cmd检查 就启动
					if !tickerCheckCmdOn {
						tickerCheckCmd.Reset(checkCmdInterval)
						tickerCheckCmdOn = true
						if rl.Cnf.EnableDebug {
							rl.Logger.Sugar().Debugf("Reset check cmd Ticker.%s", rl.Info())
						}
					}
				} else {
					//未连接的情况下 如果启动了cmd检查 就先关闭
					if tickerCheckCmdOn {
						tickerCheckCmd.Stop()
						select {
						case <-tickerCheckCmd.C:
						default:
						}
						tickerCheckCmdOn = false
						if rl.Cnf.EnableDebug {
							rl.Logger.Sugar().Debugf("Stop check cmd Ticker.%s", rl.Info())
						}
					}
					continue
				}
			}
		}
	}

}

func (rl *Relay) WorkDriverFn(ctx context.Context) {
	if rl.Cnf.EnableDebug {
		rl.Logger.Sugar().Debugf("WorkDriverFn Started : ---- OK ----.%s", rl.Info())
	}
	for {
		runtime.Gosched()
		select {
		case <-rl.Ctrl().Context().Done():
			{
				return
			}
		case cmd, ok := <-rl.cmdChan:
			{
				if !ok {
					rl.Logger.Sugar().Warnf("rld.getCmdChan() chan closed ")
					continue
				}

				rlcmd, transformOK := cmd.(*RelayCmd)
				if !transformOK {
					rl.Logger.Sugar().Warnf(" TokenCompletor transform to RelayCmd false")
					continue
				}

				rl.stateMux.Lock()
				rl.radarIsSetOnOrOff = rlcmd.SetRadarON
				rl.radarNeedSetCmd = (rl.radarCurrentOnOrOff != rl.radarIsSetOnOrOff)
				rl.stateMux.Unlock()
				rlcmd.Completed()
				rlcmd = nil
			}
		}
	}
}

func (rl *Relay) checkRelayConnected() error {
	if rl.GetState() != drs.Connected {
		rl.SetState(drs.Connecting)
		rl.tcphandler.WithCtx(rl.Ctrl().Context())
		err := rl.tcphandler.Connect()
		if err != nil {
			rl.Logger.Sugar().Errorf("Connect failed: %s , %+v.%s", rl.tcphandler.Address, err, rl.Info())
			rl.SetState(drs.BeDisConnected)
			return err
		} else {
			rl.SetState(drs.Connected)
			rl.Logger.Sugar().Infof("%s -Connected. %s", rl.tcphandler.Address, rl.Info())
			dos, err := rl.getDOs()
			if err != nil {
				return err
			} else {
				//todo: check 这里配置出现错误的情况下会出问题
				radar_id := uint16(rl.Cnf.RadarId)
				//与实际的relay相关
				if radar_id > 8 {
					rl.Logger.Sugar().Warnf("radar Id config err: %d.%s", rl.Cnf.RadarId, rl.Info())
				} else {
					rl.stateMux.Lock()
					rl.radarCurrentOnOrOff = dos.IsOn(radar_id)
					rl.safeToExit = !rl.radarCurrentOnOrOff
					rl.stateMux.Unlock()
				}

				rl.WriteRelayInfoToLDB(dos)

				if rl.Cnf.EnableDebug {
					rl.Logger.Sugar().Debugf("getDOs: %s; err: %+v.%s", dos.String(), err, rl.Info())
				}
				return nil
			}
		}
	} else {
		return nil
	}

}

func (rl *Relay) setRelayRadar(onOrOff bool) error {
	err := rl.switchOnOff(uint16(rl.Cnf.RadarId), onOrOff)
	if err != nil {
		rl.Logger.Sugar().Errorf("setRadar %t, err: %+v.%s", onOrOff, err, rl.Info())
		return err
	} else {
		//设置成功的情况下 更新当前雷达开关状态
		rl.stateMux.Lock()
		rl.radarCurrentOnOrOff = onOrOff
		rl.safeToExit = !rl.radarCurrentOnOrOff
		rl.stateMux.Unlock()
		dos, err := rl.getDOs()
		if err == nil {
			if rl.LDB != nil {
				rl.WriteRelayInfoToLDB(dos)
			}
			if rl.Cnf.EnableDebug {
				rl.Logger.Sugar().Debugf("getDOs OK: %s .%s", dos.String(), rl.Info())
			}
		} else {
			if rl.Cnf.EnableDebug {
				rl.Logger.Sugar().Debugf("getDOs err: %+v .%s", err, rl.Info())
			}
			return err
		}
		return nil
	}
}

func (rl *Relay) getDOs() (dosStatus zlan6842.DOitems, err error) {
	// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
	results, err := rl.client.ReadCoils(zlan6842.ZlanDOaddr, zlan6842.ZlanDOitemsMax)
	if err != nil {

		if rl.IsNetWorkErr(err) {
			rl.SetState(drs.BeDisConnected)
			rl.tcphandler.Close()
			rl.Logger.Sugar().Errorf("ReadCoils err:%+v --close.", err)
			return nil, err
		} else if rl.IsModbusErr(err) {
			rl.SetState(drs.DoDisConnected)
			rl.tcphandler.Close()
			rl.Logger.Sugar().Errorf("ReadCoils err:%+v --close.", err)
			return nil, err
		}
		rl.Logger.Sugar().Errorf("ReadCoils err:%+v", err)
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

func (rl *Relay) setDOsAll(onOroff bool) error {
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
		_, err := rl.client.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			noErr = false
			if rl.IsNetWorkErr(err) {
				rl.SetState(drs.BeDisConnected)
				rl.tcphandler.Close()
				rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v  --- close.", err)
				return err
			} else if rl.IsModbusErr(err) {
				rl.SetState(drs.DoDisConnected)
				rl.tcphandler.Close()
				rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v  --- close.", err)
				return err
			} else if errors.Is(err, context.Canceled) {
				rl.SetState(drs.DoDisConnected)
				rl.tcphandler.Close()
				rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v  --- close.", err)
				return err
			} else {
				errStr += fmt.Sprintf("%+v\n", err)
			}
			rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v", err)
		}
	}

	if noErr {
		return nil
	} else {
		return fmt.Errorf("%s", errStr)
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
func (rl *Relay) IsNetWorkErr(err error) (closed bool) {
	if ne, ok := err.(*net.OpError); ok {
		var se *os.SyscallError
		if errors.As(ne, &se) {
			ss := strings.ToLower(se.Error())
			if strings.Contains(ss, "broken pipe") ||
				strings.Contains(ss, "established connection was aborted") ||
				strings.Contains(ss, "connection was forcibly closed") {
				rl.Logger.Sugar().Warnf("connetion close,write err:%+v", ss)
				return true
			} else if strings.Contains(ss, "connection reset by peer") ||
				strings.Contains(ss, "connection closed by peer") ||
				strings.Contains(ss, "eof") {
				rl.Logger.Sugar().Warnf("connetion close,read err:%+v", ss)
				return true
			} else if strings.Contains(ss, "i/o timeout") {
				rl.Logger.Sugar().Warnf("connetion read timeout err:%+v", ss)
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

func (rl *Relay) IsModbusErr(err error) bool {
	ss := strings.ToLower(err.Error())
	if strings.Contains(ss, "modbus") {
		return true
	} else {
		return false
	}
}

func (rl *Relay) WriteRelayInfoToLDB(dois zlan6842.DOitems) {
	var info db.IotNavlDevicesStatus

	ScheduleId := rl.ScheduleId.GetScheduleId()
	navlState, navs := rl.PredicateSchedule.GetScheduleStatus()
	info.NavlockId = fmt.Sprintf("%d", rl.TrackingSt.NavlockIdentifier)
	info.ScheduleId = ScheduleId
	info.NavlockStatus = navlState.String()
	info.NavStatus = navs.String()
	info.DeviceTag = rl.Cnf.DeviceTag
	info.FormatVersion = db.FormatVersionTest + 0

	ralayinfo := RelayInfo{
		DriveName:  rl.Cnf.DriveName,
		RelayArray: dois.String(),
	}
	info.Info = ralayinfo.JsonData()

	if rl.Cnf.EnableAutoTest && rl.LDB != nil {
		rl.writeNavlockRuntimeToDB(rl.LDB, &info)
	}
}

func (rl *Relay) writeNavlockRuntimeToDB(dab *gorm.DB, info *db.IotNavlDevicesStatus) {
	if err := dab.Create(info).Error; err != nil {
		rl.Logger.Sugar().Errorf(`ScheduleId-[%s]:"insert to db err: %+v"`, info.ScheduleId, err)
	}
}

// 快捷的开启单个通道
func (rl *Relay) switchOnOff(id uint16, onOrOff bool) error {
	dos := zlan6842.DOitems{
		zlan6842.DOitem{Id: id, On: onOrOff},
	}

	err := rl.setDOs(dos)
	if err != nil {
		rl.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
		return err
	}
	return nil
}

func (rl *Relay) setDOs(dos zlan6842.DOitems) error {
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
		_, err := rl.client.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			noErr = false
			if rl.IsNetWorkErr(err) {
				rl.SetState(drs.BeDisConnected)
				rl.tcphandler.Close()
				rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v ----close", err)
				return err
			} else if rl.IsModbusErr(err) {
				rl.SetState(drs.DoDisConnected)
				rl.tcphandler.Close()
				rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v ----close", err)
				return err
			} else {
				errStr += fmt.Sprintf("%+v\n", err)
			}
			rl.Logger.Sugar().Errorf("WriteSingleCoil err:%+v", err)
		}
	}

	if noErr {
		return nil
	} else {
		return fmt.Errorf("%s", errStr)
	}
}

func (rl *Relay) checkSafeExit(ifTurnOffAll bool) bool {
	if ifTurnOffAll {
		err := rl.setDOsAll(false)
		if err != nil {
			rl.Logger.Sugar().Warnf("Set Dos all false,err:%+v", err)
			return false
		} else {
			rl.safeToExit = true
			return true
		}
	}
	return false
}

func (rl *Relay) saveToLDB() {
	dos, err := rl.getDOs()
	if err != nil {
		rl.Logger.Sugar().Errorf("GetDOs,err:%+v", err)
		return
	}
	rl.WriteRelayInfoToLDB(dos)
	rl.writeRuntimeHistorytoLDB(dos.JsonData())
}

func (rl *Relay) writeRuntimeHistorytoLDB(bs []byte) {
	if !rl.Cnf.EnableAutoTest || rl.LDB == nil {
		return
	}
	var info db.IotNavlCpntState
	info.NavlockId = fmt.Sprintf("%d", rl.NavlockIdentifier)
	info.ScheduleId = rl.ScheduleId.GetScheduleId()
	navls, navs := rl.PredicateSchedule.GetScheduleStatus()
	info.NavStatus = navs.String()
	info.NavlockStatus = navls.String()

	info.ServerName = rl.ServerName
	info.SystemName = rl.SystemName
	info.DriverName = rl.DriverName + rl.Cnf.DriveName

	info.FormatVersion = db.FormatVersionTest + 0
	info.InfoLevel = uint(db.LevelInfo)
	info.Info = bs

	dbd := rl.LDB.Debug()
	tx := dbd.Begin()
	if err := tx.Create(&info).Error; err != nil {
		rl.Logger.Sugar().Errorf(`ScheduleId-[%s]:"insert to db err: %+v"`, rl.ScheduleId.GetScheduleId(), err)
		tx.Rollback()
	}
	tx.Commit()
}
