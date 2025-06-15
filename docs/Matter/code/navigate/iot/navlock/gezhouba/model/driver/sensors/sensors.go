package sensors

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	cmd_token "navigate/common/model/token"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	drs_radar "navigate/iot/navlock/gezhouba/model/driver/radar"
	drs_relay "navigate/iot/navlock/gezhouba/model/driver/relay"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

const (
	SensorsUnitKind     = cmpt.KindName("Sensors")
	SensorsUnitId_Left  = cmpt.IdName("Left")
	SensorsUnitId_Right = cmpt.IdName("Right")

	cmdbufsize      = 10240
	cmdWaitInterval = 1000 * time.Millisecond
)

var (
	//Verify Satisfies interfaces
	_ mdl.WorkerRecover = (*Sensors)(nil)
	_ cmpt.Cpt          = (*Sensors)(nil)
	_ cmpt.CptsOperator = (*Sensors)(nil)
	_ cmpt.CptComposite = (*Sensors)(nil)
	_ cmpt.CptRoot      = (*Sensors)(nil)
)

type SensorsCmd struct {
	*cmd_token.BaseToken
	SetRelayONOrOFF bool
	SetRadarConnect bool
}

func NewSensorsCmd() *SensorsCmd {
	return &SensorsCmd{
		BaseToken:       cmd_token.NewBaseToken(),
		SetRelayONOrOFF: false,
		SetRadarConnect: false,
	}
}

type cmdSetInfo struct {
	ConnectOk  bool // 是否是处于连接状态
	CurrentOn  bool // 当前的状态是否打开
	CmdSetOn   bool // 命令设置的状态
	NeedCmdSet bool // 是否需要执行命令
}

// Sensors 包含雷达 云台 relay等实现一个独立的控制和测速功能的驱动集合
// 实现基本的测速单元控制逻辑
type Sensors struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	cmdChan chan cmd_token.TokenCompletor
	Cnf     *gzb_cf.SensorsUnitSetup

	relayCmdInfo *cmdSetInfo
	Relay        *drs_relay.Relay
	radarCmdInfo *cmdSetInfo
	Radar        *drs_radar.Radar
}

func NewSensorsDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Sensors {
	tmp := &Sensors{
		BaseCptCtx:   dr,
		TrackingSt:   trs,
		relayCmdInfo: &cmdSetInfo{},
		radarCmdInfo: &cmdSetInfo{},
	}
	tmp.cmdChan = make(chan cmd_token.TokenCompletor, cmdbufsize)
	tmp.WorkerRecover = tmp
	return tmp
}

// 外部的输入的命令
func (ss *Sensors) Commander(cmd cmd_token.TokenCompletor) bool {
	if !ss.State.Load().(bool) {
		cmd.SetErr(fmt.Errorf("%s is exited", ss.Info()))
		return false
	}

	timer := mdl.TimerPool.Get(cmdWaitInterval)
	defer mdl.TimerPool.Put(timer)
	for {
		select {
		case <-ss.Ctrl().Context().Done():
			return false
		case ss.cmdChan <- cmd:
			if ss.TrackingSt.EnableDebug {
				ss.Logger.Sugar().Debugf("get cmd_token - ok.cmd chan cap:%d.%s", cap(ss.cmdChan), ss.Info())
			}

			return true
		case <-timer.C:
			if ss.TrackingSt.EnableDebug {
				ss.Logger.Sugar().Debugf("get cmd_token - time out.cmd chan cap:%d.%s", cap(ss.cmdChan), ss.Info())
			}
			return false
		}
	}
}

// todo: 打开开关 和 接收雷达数据等逻辑 细腻的控制在这实现 当前没有单独打开和关闭的需求
func (ss *Sensors) Work() error {
	//创建上下游开关检查
	timeIntervalCheck := time.Duration(500) * time.Millisecond
	tickerChecker := time.NewTicker(timeIntervalCheck)
	tickerChecker.Stop()
	select {
	case <-tickerChecker.C:
	default:
	}
	tickercheckedEnabled := false
	defer tickerChecker.Stop()

	for {
		runtime.Gosched()
		select {
		case <-ss.Ctrl().Context().Done():
			{
				if err := ss.Ctrl().Context().Err(); err != nil {
					if !errors.Is(err, context.Canceled) {
						return err
					}
				}
				ss.Logger.Sugar().Infof("Sensors Exit : ------ OK ------,%s", ss.Info())
				return nil
			}
		case <-tickerChecker.C:
			{
				// drain time event
				select {
				case <-tickerChecker.C:
				default:
				}

				if ss.TrackingSt.EnableDebug {
					ss.Logger.Sugar().Debugf("Check cmd - relayCmdInfo:%+v,radarCmdInfo:%+v,%s.",
						ss.relayCmdInfo, ss.radarCmdInfo, ss.Info())
				}

				ss.CmdSensors()
				ss.CheckingSensorState()
				if ss.relayCmdInfo.ConnectOk {
					if !(ss.relayCmdInfo.NeedCmdSet || ss.radarCmdInfo.NeedCmdSet) {
						tickerChecker.Stop()
						// drain time event
						select {
						case <-tickerChecker.C:
						default:
						}
						tickercheckedEnabled = false
						if ss.TrackingSt.EnableDebug {
							ss.Logger.Sugar().Debugf("Stop Checking Cmd: relayCmdInfo:%+v,radarCmdInfo:%+v,%s.",
								ss.relayCmdInfo, ss.radarCmdInfo, ss.Info())
						}
					}
				}
			}
		case cmd, ok := <-ss.cmdChan:
			{
				if !ok {
					ss.Logger.Sugar().Warnf("%s: cmdChan receive err, cmdChan Closed.", ss.Info())
					continue
				}

				ssuCmd, transformOK := cmd.(*SensorsCmd)
				if !transformOK {
					ss.Logger.Sugar().Warnf("%s: TokenCompletor transform to SensorsCmd false", ss.Info())
					continue
				}
				// 只是简单的复制命令参数 判断是否需要设置命令
				ss.relayCmdInfo.CmdSetOn = ssuCmd.SetRelayONOrOFF
				ss.radarCmdInfo.CmdSetOn = ssuCmd.SetRadarConnect
				ssuCmd.Completed()
				ssuCmd = nil

				ss.CheckingSensorState()
				ss.CmdSensors()
				ss.CheckingSensorState()

				if ss.relayCmdInfo.ConnectOk {
					if ss.relayCmdInfo.NeedCmdSet || ss.radarCmdInfo.NeedCmdSet {
						if !tickercheckedEnabled {
							tickerChecker.Reset(timeIntervalCheck)
							tickercheckedEnabled = true

							if ss.TrackingSt.EnableDebug {
								ss.Logger.Sugar().Debugf(`Start Checking Cmd: relayCmdInfo:%+v,radarCmdInfo:%+v,%s.`,
									ss.relayCmdInfo, ss.radarCmdInfo, ss.Info())
							}
						}
					}
				} else {
					if !tickercheckedEnabled {
						tickerChecker.Reset(timeIntervalCheck)
						tickercheckedEnabled = true

						if ss.TrackingSt.EnableDebug {
							ss.Logger.Sugar().Debugf(`Start Checking Cmd: relayCmdInfo:%+v,radarCmdInfo:%+v,%s.`,
								ss.relayCmdInfo, ss.radarCmdInfo, ss.Info())
						}
					}
				}
			}
		}
	}
}

// 获取当前所有传感器的状态参数
func (ss *Sensors) CheckingSensorState() {
	connect, radarCurrentOn := ss.Relay.RadarCurrentOn()
	ss.relayCmdInfo.ConnectOk = connect
	ss.relayCmdInfo.CurrentOn = radarCurrentOn
	ss.relayCmdInfo.NeedCmdSet = !(ss.relayCmdInfo.CurrentOn == ss.relayCmdInfo.CmdSetOn)

	ss.radarCmdInfo.ConnectOk = ss.Radar.GetCurrentConnected()
	ss.radarCmdInfo.CurrentOn = ss.radarCmdInfo.ConnectOk
	ss.radarCmdInfo.NeedCmdSet = !(ss.radarCmdInfo.CurrentOn == ss.radarCmdInfo.CmdSetOn)
}

// 针对需要的传感器发送命令
func (ss *Sensors) CmdSensors() {
	if ss.relayCmdInfo.ConnectOk {
		if ss.relayCmdInfo.NeedCmdSet {
			ss.controlRelay(ss.relayCmdInfo.CmdSetOn)
		}
	}

	if ss.radarCmdInfo.NeedCmdSet {
		ss.controlRadar(ss.radarCmdInfo.CmdSetOn)
	}
}

func (ss *Sensors) controlRelay(setOnOrOff bool) {
	if ss.Relay == nil {
		ss.Logger.Sugar().Warnf(`[%s]:" Relay is nil!"`, ss.Info())
		return
	}

	cmd := drs_relay.NewRelayCmd()
	cmd.SetRadarON = setOnOrOff
	Ok1 := ss.Relay.Commander(cmd)
	if Ok1 {
		ok2 := cmd.WaitTimeout(cmdWaitInterval)
		if ok2 {
			if ss.TrackingSt.EnableDebug {
				//忽略云台
				ss.Logger.Sugar().Debugf("SetRelay-Radar device: relayCmdInfo:%+v, %s.", ss.relayCmdInfo, ss.Info())
			}
		} else {
			ss.Logger.Sugar().Errorf("SetRelay-Radar device: relayCmdInfo:%+v,err:%+v, %s.", ss.relayCmdInfo, cmd.Err(), ss.Info())
		}
	} else {
		ss.Logger.Sugar().Errorf("SetRelay-Radar device: relayCmdInfo:%+v,false, %s.", ss.relayCmdInfo, ss.Info())
	}
}

func (ss *Sensors) controlRadar(setConnect bool) {
	if ss.Radar == nil {
		ss.Logger.Sugar().Warnf(`[%s]:" Radar is nil!"`, ss.Info())
		return
	}

	cmd := drs_radar.NewRadarCmd()
	cmd.SetConnect = setConnect
	Ok := ss.Radar.Commander(cmd)
	if Ok {
		ok := cmd.WaitTimeout(cmdWaitInterval)
		if ok {

			if ss.TrackingSt.EnableDebug {
				ss.Logger.Sugar().Debugf("SetRadar: RadarCmdInfo:%+v, %s.", ss.radarCmdInfo, ss.Info())
			}
		} else {
			ss.Logger.Sugar().Errorf("SetRadar: RadarCmdInfo:%+v,err: %+v, %s.", ss.radarCmdInfo, cmd.Err(), ss.Info())
		}
	} else {
		ss.Logger.Sugar().Errorf("SetRadar device: RadarCmdInfo:%+v,false, %s.", ss.radarCmdInfo, ss.Info())
	}
}

func (ss *Sensors) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		ss.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (ss *Sensors) Start() (err error) {
	if ss.CptSt == nil || ss.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", ss.Info())
	} else {
		if err := ss.CptSt.Validate(); err != nil {
			return err
		}
	}

	if ss.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", ss.Info())
	}

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("---- Starting --- %s", ss.Info())
	}

	ss.Ctrl().WaitGroup().StartingWait(ss.WorkerRecover)
	if ss.Cpts != nil && ss.Cpts.Len() > 0 {
		err = ss.Cpts.Start()
		if err != nil {
			ss.Logger.Sugar().Errorf("%s Components Start() error : %#v", ss.Info(), err)
		}
	}
	ss.State.Store(true)

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("---- Started --- %s", ss.Info())
	}
	return err
}

func (ss *Sensors) Stop() (err error) {
	if !ss.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", ss.Info())
	}

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("---- Stopping --- %s", ss.Info())
	}

	ss.Ctrl().Cancel()
	<-ss.Ctrl().Context().Done()
	if ss.Cpts != nil && ss.Cpts.Len() > 0 {
		err = ss.Cpts.Stop()
		if err != nil {
			ss.Logger.Sugar().Errorf("%s  Stop() error : %+v", ss.Info(), err)
		}
	}
	ss.State.Store(false)

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("---- Stopped --- %s", ss.Info())
	}
	return nil
}

// todo: 读取配置文件 创建组件
func (ss *Sensors) InitComponentsFromConfig(conf *gzb_cf.SensorsUnitSetup) error {
	//------------- 初始化WorkDriver -----------------

	// ----------------------------------------------
	navlockId := fmt.Sprintf("#%d", ss.TrackingSt.NavlockIdentifier)
	//------------- 初始化Relay -----------------
	if ss.Cpts.Cpt(cmpt.IdName("Relay")) == nil {
		mcs := navl_md.NewCptSt(drs_relay.RelayDriverKind,
			cmpt.IdName("Relay"),
			0,
			ss.Ctrl().ForkCtxWg())
		mss := navl_md.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"",
			"",
			"Relay")

		tmpdr := navl_md.NewBaseCptCtx(mcs, mss)
		ss.Relay = drs_relay.NewRelayDvr(tmpdr, ss.TrackingSt)
		ss.Relay.Cnf = conf.RelaySetup
		ss.Cpts.AddCpts(ss.Relay)
	}
	// --------------------------------------------

	//------------- 初始化Radar -----------------
	if ss.Cpts.Cpt(cmpt.IdName("Radar")) == nil {
		mcs := navl_md.NewCptSt(drs_radar.RadarKind,
			cmpt.IdName("Radar"),
			0,
			ss.Ctrl().ForkCtxWg())
		mss := navl_md.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"",
			"",
			"Radar")

		tmpdr := navl_md.NewBaseCptCtx(mcs, mss)

		ss.Radar = drs_radar.NewRadarDvr(tmpdr, ss.TrackingSt)
		ss.Radar.Cnf = conf.RadarSetup
		ss.Cpts.AddCpts(ss.Radar)
	}
	// --------------------------------------------

	return nil
}
