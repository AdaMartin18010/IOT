package radar

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"sync"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	cmd_token "navigate/common/model/token"
	iot_cf "navigate/config/iot/drivers"
	drs "navigate/iot/drivers"
	radar "navigate/iot/drivers/radar"
	v0826 "navigate/iot/drivers/radar/v220828"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
	gzb_db "navigate/store/iot/navlock/gezhouba"
)

const (
	RadarKind       = cmpt.KindName("radar")
	radarMsgLen     = 10240
	cmdbufsize      = 10240
	cmdWaitInterval = 1000 * time.Millisecond
)

var (
	//Verify Satisfies interfaces
	_ mdl.WorkerRecover = (*Radar)(nil)
	_ cmpt.Cpt          = (*Radar)(nil)
	_ cmpt.CptsOperator = (*Radar)(nil)
	_ cmpt.CptComposite = (*Radar)(nil)
	_ cmpt.CptRoot      = (*Radar)(nil)
)

type RadarCmd struct {
	*cmd_token.BaseToken
	SetConnect bool
}

func NewRadarCmd() *RadarCmd {
	return &RadarCmd{
		BaseToken:  cmd_token.NewBaseToken(),
		SetConnect: false,
	}
}

type Radar struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	msgChan chan any
	cmdChan chan cmd_token.TokenCompletor

	needSetCmd     bool
	currentConnect bool
	isSetConnected bool
	stateMux       *sync.Mutex
	//sdu            *v0826.ShipDataUnit
	wkd *drs_wd.Work

	Cnf        *iot_cf.ModbusConfig
	tcphandler *radar.RadarTcpClient
	//lastSpeed  float32
	currentSID string
}

func NewRadarDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Radar {
	tmpMnd := &Radar{
		BaseCptCtx: dr,
		TrackingSt: trs,
		msgChan:    make(chan any, radarMsgLen),
		stateMux:   &sync.Mutex{},
	}
	tmpMnd.cmdChan = make(chan cmd_token.TokenCompletor, cmdbufsize)
	tmpMnd.WorkerRecover = tmpMnd
	return tmpMnd
}

func (rdd *Radar) Recover() {
	if rc := recover(); rc != nil {
		var buf [4096]byte
		n := runtime.Stack(buf[:], false)
		rdd.Logger.Sugar().Warnf(`Work recover :%+v,stack trace: %s`, rc, buf[:n])
	}
}

func (rdd *Radar) Start() error {
	if rdd.State.Load().(bool) {
		return fmt.Errorf("is already Started.%s", rdd.Info())
	}

	//------------- worker -----------------
	// 只有RadarDevice初始化完成后 才能取到合适的参数 所以在Start()里面实现
	cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
		cmpt.IdName("RadarMsgReceive"),
		0,
		rdd.Ctrl().ForkCtxWg())

	dr := navl_md.NewBaseCptCtx(cs,
		navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
			rdd.TrackingSt.NavlockIdentifier),
			"worker",
			"",
			"RadarDriver-work-dr"))

	workdr := drs_wd.NewWorkDvr(dr, rdd.TrackingSt)
	workdr.Fn = rdd.WorkDriverFn
	rdd.wkd = workdr
	rdd.wkd.EnableDebug = rdd.Cnf.EnableDebug
	// --------------------------------------------
	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf("Starting: ip-%s.%s", rdd.Cnf.Ipaddress, rdd.Info())
		rdd.Logger.Sugar().Debugf("config : %+v.%s", rdd.Cnf, rdd.Info())
	}

	rdd.tcphandler = &radar.RadarTcpClient{
		Address:     fmt.Sprintf("%s:%d", rdd.Cnf.Ipaddress, rdd.Cnf.Port),
		Timeout:     time.Duration(rdd.Cnf.ConnectReadTimeout) * time.Millisecond,
		IdleTimeout: time.Duration(rdd.Cnf.IdleTimeout) * time.Second,
	}

	rdd.tcphandler.Ctx = rdd.Ctrl().Context()
	rdd.tcphandler.Logger = rdd.Logger
	rdd.tcphandler.DriverConnectStatus = drs.NewDriverConnectStatus()

	rdd.Ctrl().WaitGroup().StartingWait(rdd.WorkerRecover)
	rdd.State.Store(true)
	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf("Started: ip-%s.%s", rdd.Cnf.Ipaddress, rdd.Info())
	}

	return nil
}

func (rdd *Radar) Stop() (err error) {
	if !rdd.State.Load().(bool) {
		return fmt.Errorf("iss already Stopped.%s", rdd.Info())
	}

	rdd.Ctrl().Cancel()

	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf("--- Stopping ---.%s", rdd.Info())
		rdd.Logger.Sugar().Debugf("Ctrl_info: %s .%s", rdd.Ctrl().DebugInfo(), rdd.Info())
	}
	<-rdd.Ctrl().Context().Done()

	if rdd.Cpts != nil && rdd.Cpts.Len() > 0 {
		err = rdd.Cpts.Stop()
		if err != nil {
			rdd.Logger.Sugar().Errorf("Stop() error : %+v.%s", err, rdd.Info())
		}
	}

	//rdd.lastSpeed = 0.0
	rdd.State.Store(false)

	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf("--- Stopped ---.%s", rdd.Info())
	}
	return
}

// 外部使用  来发送外部的数据
func (rdd *Radar) MsgChan() <-chan any {
	return rdd.msgChan
}

// 外部的输入的命令
func (rdd *Radar) Commander(cmd cmd_token.TokenCompletor) bool {
	if !rdd.State.Load().(bool) {
		cmd.SetErr(fmt.Errorf("Exited.%s", rdd.Info()))
		return false
	}

	timer := mdl.TimerPool.Get(cmdWaitInterval)
	defer mdl.TimerPool.Put(timer)
	for {
		select {
		case <-rdd.Ctrl().Context().Done():
			return false
		case rdd.cmdChan <- cmd:
			if rdd.Cnf.EnableDebug {
				rdd.Logger.Sugar().Debugf("get cmd_token - ok.cmd chan cap:%d.%s", cap(rdd.cmdChan), rdd.Info())
			}
			return true
		case <-timer.C:
			if rdd.Cnf.EnableDebug {
				rdd.Logger.Sugar().Debugf("get cmd_token - time out.cmd chan cap:%d.%s", cap(rdd.cmdChan), rdd.Info())
			}
			return false
		}
	}
}

func (rdd *Radar) Work() error {

	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf(`Work Starting autotest:%t.%s`, rdd.Cnf.EnableAutoTest, rdd.Info())
	}

	if rdd.wkd == nil {
		rdd.Logger.Sugar().Fatalf("Radar Worker is nil.%s", rdd.Info())
	}
	rdd.wkd.Start()
	for {
		runtime.Gosched()
		select {
		case <-rdd.Ctrl().Context().Done():
			{
				rdd.tcphandler.Close()
				return nil
			}
		case cmd, ok := <-rdd.cmdChan:
			{
				if !ok {
					rdd.Logger.Sugar().Warnf("rld.getCmdChan() chan closed.%s", rdd.Info())
					continue
				}

				rdCmd, transformOK := cmd.(*RadarCmd)
				if !transformOK {
					rdd.Logger.Sugar().Warnf("TokenCompletor transform to RadarCmd false.%s", rdd.Info())
					continue
				}

				rdd.stateMux.Lock()
				rdd.isSetConnected = rdCmd.SetConnect
				rdd.stateMux.Unlock()
				rdCmd.Completed()
				rdCmd = nil

				rdd.stateMux.Lock()
				rdd.needSetCmd = (rdd.currentConnect != rdd.isSetConnected)
				if rdd.Cnf.EnableDebug {
					rdd.Logger.Sugar().Debugf("isSetCon:%t,currentCon:%t,needSetCmd:%t.%s",
						rdd.isSetConnected, rdd.currentConnect, rdd.needSetCmd, rdd.Info())
				}
				rdd.stateMux.Unlock()
				continue
			}
		}

	}

}

func (rdd *Radar) WorkDriverFn(ctx context.Context) {
	if rdd.Cnf.EnableDebug {
		rdd.Logger.Sugar().Debugf("WorkDriverFn Started : ---- OK ----.%s", rdd.Info())
	}

	reConnectedInterval := time.Duration(1) * time.Second
	tickerConnected := time.NewTicker(reConnectedInterval)
	defer tickerConnected.Stop()

	receiveDataInterval := time.Duration(rdd.Cnf.ReadTimeInterval) * time.Millisecond
	tickerReceiveData := time.NewTicker(receiveDataInterval)
	defer tickerReceiveData.Stop()
	tickerReceiveData.Stop()
	select {
	case <-tickerReceiveData.C:
	default:
	}
	tickerReceiveDataSet := false

	for {
		runtime.Gosched()
		select {
		case <-rdd.Ctrl().Context().Done():
			{
				err := rdd.Ctrl().Context().Err()
				if rdd.TrackingSt.EnableDebug {
					rdd.Logger.Sugar().Debugf("WorkDriverFn Exit : err : %+v  ---------- OK ------.%s", err, rdd.Info())
				}
				return
			}
		case <-tickerConnected.C:
			{
				// drain time event
				select {
				case <-tickerConnected.C:
				default:
				}

				checkConnected := false

				rdd.stateMux.Lock()
				if rdd.needSetCmd {
					checkConnected = rdd.isSetConnected
				} else {
					checkConnected = rdd.currentConnect
				}
				rdd.stateMux.Unlock()

				if err := rdd.VerifyConnect(checkConnected); err != nil {
					//用户取消的情况下
					if errors.Is(err, context.Canceled) {
						continue
					}
					rdd.Logger.Sugar().Errorf("connecting err %+v.%s", err.Error(), rdd.Info())
				}

				if rdd.tcphandler.GetState() == drs.Connected {
					if !tickerReceiveDataSet {
						tickerReceiveData.Reset(receiveDataInterval)
						tickerReceiveDataSet = true
						//rdd.lastSpeed = 0.0
						rdd.currentSID = rdd.TrackingSt.ScheduleId.GetScheduleId()
						//rdd.sdu = nil
						if rdd.Cnf.EnableDebug {
							rdd.Logger.Sugar().Debugf("Reset Receive-data Ticker.%s", rdd.Info())
						}
					}
				} else {
					if tickerReceiveDataSet {
						tickerReceiveDataSet = false
						tickerReceiveData.Stop()
						select {
						case <-tickerReceiveData.C:
						default:
						}
						//rdd.lastSpeed = 0.0
						rdd.currentSID = ""
						//rdd.sdu = nil
					}
				}
			}
		case <-tickerReceiveData.C:
			{
				//快速反应命令 设置的状态
				rdd.stateMux.Lock()
				StopRecive := !rdd.isSetConnected
				rdd.stateMux.Unlock()
				if StopRecive {
					tickerReceiveData.Stop()
					select {
					case <-tickerReceiveData.C:
					default:
					}
					tickerReceiveDataSet = false
					continue
				}

				rdd.ReceiveMsg()
			}
		}
	}
}

func (rdd *Radar) VerifyConnect(connected bool) error {
	defer func() {
		rdd.stateMux.Lock()
		rdd.currentConnect = (rdd.tcphandler.GetState() == drs.Connected)
		rdd.needSetCmd = (rdd.currentConnect != rdd.isSetConnected)
		rdd.stateMux.Unlock()
	}()

	if connected {
		//初始化或者断开后的连接
		if rdd.tcphandler.GetState() != drs.Connected {
			rdd.tcphandler.Ctx = rdd.Ctrl().Context()
			err := rdd.tcphandler.Connect()
			if err != nil {
				rdd.Logger.Sugar().Warnf("Connect failed: %s, %+v.%s", rdd.tcphandler.Address, err, rdd.Info())
				return err
			} else {
				rdd.Logger.Sugar().Infof("Connected: %s.%s", rdd.tcphandler.Address, rdd.Info())
				return nil
			}
		}
		return nil
	} else {
		if rdd.tcphandler.GetState() == drs.Connected {
			err := rdd.tcphandler.Close()
			if err != nil {
				rdd.Logger.Sugar().Warnf("Close failed: %s, %+v.%s", rdd.tcphandler.Address, err, rdd.Info())
			}
			return err
		}
		return nil
	}
}

func (rdd *Radar) GetCurrentConnected() bool {
	rdd.stateMux.Lock()
	defer rdd.stateMux.Unlock()
	return rdd.currentConnect
}

func (rdd *Radar) ReceiveMsg() bool {
	defer func() {
		rdd.stateMux.Lock()
		rdd.currentConnect = (rdd.tcphandler.GetState() == drs.Connected)
		rdd.needSetCmd = (rdd.currentConnect != rdd.isSetConnected)
		rdd.stateMux.Unlock()
	}()

	rsp, err := rdd.tcphandler.Receive()
	if err != nil {
		//无数据可读的情况下
		if errors.Is(err, io.EOF) {
			rdd.Logger.Sugar().Errorf("tcphandler.Receive() data:%+v  err: %+v.%s", rsp, err, rdd.Info())
		}

		if rdd.tcphandler.IsNetWorkErr(err) {
			rdd.Logger.Sugar().Errorf("%s - read err: %+v.%s", rdd.tcphandler.Address, err, rdd.Info())
			rdd.tcphandler.Close()
			rdd.Logger.Sugar().Infof("%s -Close.", rdd.tcphandler.Address)
		}

		if rdd.TrackingSt.EnableDebug {
			rdd.Logger.Sugar().Debugf("tcphandler.Receive() data:%+v  err: %+v.%s", rsp, err, rdd.Info())
		}

		return false
	}

	var rdu v0826.RadarDataUnit
	rdu.Reset()
	rddu, err := rdu.Decode(rsp)
	if err != nil {
		rdd.Logger.Sugar().Errorf("rdu.Decode(rsp) err: %+v.%s", err, rdd.Info())
		return false
	}

	err = rddu.Verify()
	if err != nil {
		rdd.Logger.Sugar().Errorf("rdup.Verify() err: %+v.%s", err, rdd.Info())
		return false
	}

	if rdd.TrackingSt.EnableDebug {
		rdd.Logger.Sugar().Debugf("receive package:%s,%s", rddu.String(), rdd.Info())
	}

	if len(rddu.Ships) >= 1 {
		//判断通航状态是否是运营有效的 并获取
		scheduleId := rdd.TrackingSt.ScheduleId.GetScheduleId()
		navs, navsValid := rdd.PredicateSchedule.GetIsValidNavState()
		if !navsValid {
			rdd.Logger.Sugar().Errorf("navs,navsValid:%s,%t return.%s", navs.String(), navsValid, rdd.Info())
			return false
		}

		//判断船闸状态是否是 船舶测速系统有效的 并获取
		navls, navlsValid := rdd.PredicateSchedule.GetIsValidShipSpeedNavlState()
		if !navlsValid {
			rdd.Logger.Sugar().Errorf("navls,navlsShipSpeedValid:%s,%t return.%s", navls.String(), navlsValid, rdd.Info())
			return false
		}

		// 限定 本闸次内的速度获取
		if rdd.currentSID != rdd.TrackingSt.ScheduleId.GetScheduleId() {
			rdd.Logger.Sugar().Errorf("currentSID : %s,ScheduleId : %s is not same,%s", rdd.currentSID, rdd.TrackingSt.ScheduleId.GetScheduleId(), rdd.Info())
			return false
		}

		//限定雷达的探测距离
		if (rdu.Ships[0].Distance - rdd.TrackingSt.DetectionDistance) > 0.01 {
			rdd.Logger.Sugar().Errorf("radar package Distance > DetectionDistance:%+v,%s", rddu.String(), rdd.Info())
			return false
		}

		//限定雷达探测的目标最大速度
		{
			speed := rddu.Ships[0].Speed
			if speed < 0.0 {
				speed = -1 * speed
			}

			if (speed - rdd.TrackingSt.DetectionSpeedMax) > 0.01 {
				rdd.Logger.Sugar().Errorf("radar package Speed > DetectionSpeedMax:%+v,%s", rddu.String(), rdd.Info())
				return false
			}
		}

		//  限定加速度的情况 只要出现一次速度不符合加速度的定义 后续的速度都会被过滤掉 暂时注释掉
		// //限定雷达探测速度的加速度
		// if rdd.lastSpeed == 0.0 {
		// 	if rddu.Ships[0].Speed < 0.0 {
		// 		rdd.lastSpeed = -1 * rddu.Ships[0].Speed
		// 	}
		// } else {
		// 	speed := rddu.Ships[0].Speed
		// 	if speed < 0.0 {
		// 		speed = -1 * speed
		// 	}

		// 	if (speed-rdd.lastSpeed*rdd.TrackingSt.TargetAccelerationMax) > 0.01 ||
		// 		(rdd.lastSpeed*rdd.TrackingSt.TargetAccelerationMax-speed) > 0.01 {
		// 		rdd.Logger.Sugar().Debugf("radar package speed acceleration: %.2f,%.2f > %.2f,%s", rdd.lastSpeed, speed, rdd.TrackingSt.TargetAccelerationMax, rdd.Info())
		// 		return false
		// 	} else {
		// 		rdd.lastSpeed = speed
		// 	}
		// }

		// 注释掉 雷达数据的过滤
		// // 限定雷达速度距离的过滤 如果连续并且在精度范围内相同就过滤掉
		// if rdd.sdu == nil {
		// 	//1. 如果为初始化 则先赋值
		// 	rdd.sdu = &rddu.Ships[0]
		// } else {
		// 	//2. 对比看是否相同 相同则过滤掉
		// 	if rdd.sdu.CompareDistanceSpeed(&rddu.Ships[0]) {
		// 		rdd.Logger.Sugar().Errorf("radar package Distance Speed same filter:%+v,%s", rddu.String(), rdd.Info())
		// 		return false
		// 	}
		// 	//3. 不同则缓存为当前值
		// 	rdd.sdu = &rddu.Ships[0]
		// }

		rdd.msgChan <- rddu

		if rdd.Cnf.EnableDebug {
			rdd.Logger.Sugar().Debugf("msgchan cap:%d, push package:%+v.%s", cap(rdd.msgChan), rddu.String(), rdd.Info())
		}

		{
			// 同步给平台的表
			if rdd.LDB != nil {
				var syncSpeed gzb_db.NavlGzbSpeedLimit
				syncSpeed.NavlockId = fmt.Sprintf("葛洲坝%d#", rdd.TrackingSt.NavlockIdentifier)
				syncSpeed.ScheduleId = scheduleId
				syncSpeed.DeviceTag = rdd.Cnf.DeviceTag
				syncSpeed.ScheduleStatus = navls.String()
				syncSpeed.Time = time.Now()
				speed := rddu.Ships[0].Speed

				//速度数据处理
				if speed < 0 {
					speed = -1 * speed
				}
				syncSpeed.Speed = speed

				{
					speed64, err := strconv.ParseFloat(fmt.Sprintf("%.2f", speed), 32)
					if err != nil {
						rdd.Logger.Sugar().Errorf("strconv.ParseFloat err:%+v,%s", err, rdd.Info())
					}
					syncSpeed.Speed = float32(speed64)
				}

				//超速的判断需要从数据库获取
				if rdd.TrackingSt.Ref.SpeedSetupWrap.CheckOverSpeed(float64(rdu.Ships[0].Distance), float64(syncSpeed.Speed)) {
					syncSpeed.Warn = "超速"
				} else {
					syncSpeed.Warn = ""
				}

				err := syncSpeed.SaveToDB(rdd.LDB, rdd.Logger)
				if err != nil {
					rdd.Logger.Sugar().Errorf("syncSpeed.SaveToDB ldb err:%+v,%s", err, rdd.Info())
				}

				// 同时也写入到生产数据库
				err = syncSpeed.SaveToDB(rdd.PDB, rdd.Logger)
				if err != nil {
					rdd.Logger.Sugar().Errorf("syncSpeed.SaveToDB pdb err:%+v,%s", err, rdd.Info())
				}
			}

		}
	}
	return true
}
