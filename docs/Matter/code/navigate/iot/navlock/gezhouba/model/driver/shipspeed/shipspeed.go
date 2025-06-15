package shipspeed

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	cmd_token "navigate/common/model/token"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	v0826 "navigate/iot/drivers/radar/v220828"
	gzb_md_sens "navigate/iot/navlock/gezhouba/model/driver/sensors"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	gzb_schedule "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"
)

const (
	Kind             = cmpt.KindName("ShipSpeed")
	ShipSpeedId_Up   = cmpt.IdName("Up")
	ShipSpeedId_Down = cmpt.IdName("Down")

	cmdbufsize      = 10240
	cmdWaitInterval = 1000 * time.Millisecond
)

var (
	//Verify Satisfies interfaces
	_ mdl.WorkerRecover = (*ShipSpeed)(nil)
	_ cmpt.Cpt          = (*ShipSpeed)(nil)
	_ cmpt.CptsOperator = (*ShipSpeed)(nil)
	_ cmpt.CptComposite = (*ShipSpeed)(nil)
	_ cmpt.CptRoot      = (*ShipSpeed)(nil)
)

type ShipSpeedCmd struct {
	*cmd_token.BaseToken
	SetONOrOFF bool
}

func NewShipSpeedCmd() *ShipSpeedCmd {
	return &ShipSpeedCmd{
		BaseToken:  cmd_token.NewBaseToken(),
		SetONOrOFF: false,
	}
}

// ShipSpeed 包含左右游测速单元 可以只包含一个 组合成一个逻辑上的测速单元
// 实现基本的船闸上下游测速单元控制逻辑
type ShipSpeed struct {
	*navl_md.BaseCptCtx
	*gzb_schedule.TrackingSt

	msgChanShipSpeed <-chan any

	cmdChan         chan cmd_token.TokenCompletor
	Cnf             *gzb_cf.SpeedMeasureUnitSetup
	leftwkd         *drs_wd.Work //左边收数据
	rightwkd        *drs_wd.Work //右边收数据
	LeftSensUnit    *gzb_md_sens.Sensors
	setLeftOnOrOff  bool
	RightSensUint   *gzb_md_sens.Sensors
	setRightOnOrOff bool

	predicateUnit *gzb_md_sens.Sensors
}

func NewShipSpeedDvr(dr *navl_md.BaseCptCtx, trs *gzb_schedule.TrackingSt) *ShipSpeed {
	tmp := &ShipSpeed{
		BaseCptCtx:    dr,
		TrackingSt:    trs,
		leftwkd:       nil,
		rightwkd:      nil,
		LeftSensUnit:  nil,
		RightSensUint: nil,
		predicateUnit: nil,
	}
	tmp.cmdChan = make(chan cmd_token.TokenCompletor, cmdbufsize)
	tmp.WorkerRecover = tmp
	return tmp
}

// 获取左右是否开启的状态 综合所有情况
func (ss *ShipSpeed) IsConnectedOn() (connect, on bool) {
	// 上下游 左右单边部署的情况下
	if ss.predicateUnit != nil {
		relayConnected, radarOnorOff := ss.predicateUnit.Relay.RadarCurrentOn()
		radarConnected := ss.predicateUnit.Radar.GetCurrentConnected()
		if relayConnected {
			//雷达打开的情况下 同时雷达也连接上
			if radarOnorOff {
				on = radarOnorOff && radarConnected
			} else {
				// 雷达关闭的情况下 同时雷达也断开连接
				on = radarOnorOff || radarConnected
			}
		}
		connect = relayConnected
		return
	}

	// 上下游 左右双边部署的情况下
	if (ss.LeftSensUnit != nil) && (ss.RightSensUint != nil) {
		leftConnect := false
		leftOnOrOff := false
		rightConnect := false
		rightOnOrOff := false

		//检查左边的情况
		{
			relayConnected, radarOnorOff := ss.LeftSensUnit.Relay.RadarCurrentOn()
			radarConnected := ss.LeftSensUnit.Radar.GetCurrentConnected()

			if relayConnected {
				//雷达打开的情况下 同时雷达也连接上
				if radarOnorOff {
					leftOnOrOff = radarOnorOff && radarConnected
				} else {
					// 雷达关闭的情况下  同时雷达也断开连接
					leftOnOrOff = radarOnorOff || radarConnected
				}
			}
			leftConnect = relayConnected
		}

		//检查右边的情况
		{
			relayConnected, radarOnorOff := ss.RightSensUint.Relay.RadarCurrentOn()
			radarConnected := ss.RightSensUint.Radar.GetCurrentConnected()

			if relayConnected {
				//雷达打开的情况下 同时雷达也连接上
				if radarOnorOff {
					rightOnOrOff = radarOnorOff && radarConnected
				} else {
					// 雷达关闭的情况下 同时雷达也断开连接
					rightOnOrOff = radarOnorOff || radarConnected
				}
			}
			rightConnect = relayConnected
		}

		connect = leftConnect && rightConnect
		on = leftOnOrOff && rightOnOrOff
		return
	}

	return
}

// 外部的输入的命令
func (ss *ShipSpeed) Commander(cmd cmd_token.TokenCompletor) bool {
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
				ss.Logger.Sugar().Debugf("get cmd_token - time out: cmd chan cap:%d.%s", cap(ss.cmdChan), ss.Info())
			}
			return false
		}
	}
}

func (ss *ShipSpeed) cmdSensorsUnitLeft(SetONOrOFF bool) bool {
	if ss.setLeftOnOrOff == SetONOrOFF {
		return true
	}

	upCmd := gzb_md_sens.NewSensorsCmd()
	upCmd.SetRelayONOrOFF = SetONOrOFF
	upCmd.SetRadarConnect = SetONOrOFF
	if !ss.LeftSensUnit.Commander(upCmd) {
		ss.Logger.Sugar().Warnf(`LeftUnit Commander: SetONOrOFF = %t , UpUnit.Commander() return false.%s`, SetONOrOFF, ss.Info())
		return false
	}

	ok := upCmd.WaitTimeout(cmdWaitInterval)
	if ok {
		ss.setLeftOnOrOff = SetONOrOFF
		if ss.TrackingSt.EnableDebug {
			ss.Logger.Sugar().Debugf("Set LeftUnit SetONOrOFF:%t,OK.%s", SetONOrOFF, ss.Info())
		}
		return true
	} else {
		ss.Logger.Sugar().Errorf("Set LeftUnit SetONOrOFF:%t,Failed, err : %+v.%s", SetONOrOFF, upCmd.Err(), ss.Info())
		return false
	}
}

func (ss *ShipSpeed) cmdSensorsUnitRight(SetONOrOFF bool) bool {
	if ss.setRightOnOrOff == SetONOrOFF {
		return true
	}

	downCmd := gzb_md_sens.NewSensorsCmd()
	downCmd.SetRelayONOrOFF = SetONOrOFF
	downCmd.SetRadarConnect = SetONOrOFF
	if !ss.RightSensUint.Commander(downCmd) {
		ss.Logger.Sugar().Warnf(`RightUnit Commander: SetONOrOFF = %t , UpUnit.Commander() return false.%s`, SetONOrOFF, ss.Info())
		return false
	}

	ok := downCmd.WaitTimeout(cmdWaitInterval)
	if ok {
		ss.setRightOnOrOff = SetONOrOFF
		if ss.TrackingSt.EnableDebug {
			ss.Logger.Sugar().Debugf("Set RightUnit SetONOrOFF:%t,OK.%s", SetONOrOFF, ss.Info())
		}
		return true
	} else {
		ss.Logger.Sugar().Errorf("Set RightUnit SetONOrOFF:%t,Failed, err : %+v.%s", SetONOrOFF, downCmd.Err(), ss.Info())
		return false
	}
}

// todo: implement this
// 需要实现基础的打开关闭情况下的测试 和 控制
func (ss *ShipSpeed) Work() error {
	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Infof("ShipSpeed Started : ---- OK ----.%s", ss.Info())
	}
	// 各个子组件创建成功的情况下 启动接收雷达数据
	if ss.leftwkd == nil && ss.rightwkd == nil {
		ss.Logger.Sugar().Fatalf("ShipSpeed Worker is nil .%s", ss.Info())
	}

	// 检查是否符合现场的部署环境
	if (ss.LeftSensUnit != nil) && (ss.RightSensUint != nil) {
		// if ss.TrackingSt.SensorsLayout != 2 {
		// 	ss.Logger.Sugar().Fatalf("ShipSpeed both left and right worker is not nil And config SensorsLayout is not 2. %s", ss.Info())
		// }
		ss.predicateUnit = nil
	} else {
		// if ss.TrackingSt.SensorsLayout != 1 {
		// 	ss.Logger.Sugar().Fatalf("ShipSpeed one of  left and right worker is  not nil And config SensorsLayout is not 1. %s", ss.Info())
		// }

		if ss.RightSensUint != nil {
			ss.predicateUnit = ss.RightSensUint
		}

		if ss.LeftSensUnit != nil {
			ss.predicateUnit = ss.LeftSensUnit
		}
	}

	if ss.leftwkd != nil {
		// // 1:  上下游 左右单边部署 2: 上下游 左右双边部署
		// if ss.TrackingSt.SensorsLayout == 1 {
		// 	ss.predicateUnit = ss.LeftSensUnit
		// }

		ss.leftwkd.Start()
	}

	if ss.rightwkd != nil {
		// // 1:  上下游 左右单边部署 2: 上下游 左右双边部署
		// if ss.TrackingSt.SensorsLayout == 1 {
		// 	ss.predicateUnit = ss.RightSensUint
		// }

		ss.rightwkd.Start()
	}

	// if ss.TrackingSt.SensorsLayout == 1 {
	// 	if ss.predicateUnit == nil {
	// 		ss.Logger.Sugar().Fatalf("ShipSpeed both left and right sensors is  not created. %s", ss.Info())
	// 	}
	// }

	//发送给播放屏的速度消息
	ss.msgChanShipSpeed = ss.EventChans.Subscribe(gzb_schedule.MsgTopic_PlayScreen_ShipSpeeds)
	defer func() {
		ss.EventChans.UnSubscribe(gzb_schedule.MsgTopic_PlayScreen_ShipSpeeds, ss.msgChanShipSpeed)
	}()

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
				if ss.TrackingSt.EnableDebug {
					ss.Logger.Sugar().Debugf("ShipSpeed Exit : ---------- OK ------.%s", ss.Info())
				}
				return nil
			}
		case cmd, ok := <-ss.cmdChan:
			{
				if !ok {
					ss.Logger.Sugar().Warnf("cmdChan() chan closed ")
					continue
				}

				muCmd, transformOK := cmd.(*ShipSpeedCmd)
				if !transformOK {
					ss.Logger.Sugar().Warnf("TokenCompletor transform to RelayCmd false")
					continue
				}

				if ss.TrackingSt.EnableDebug {
					ss.Logger.Sugar().Debugf("get cmd form chan cap:%d,value:%+v.%s", cap(ss.cmdChan), muCmd, ss.Info())
				}
				// 只是简单的透传 命令 不做状态维护和更新
				leftOk := false
				rightOk := false
				if ss.LeftSensUnit != nil {
					leftOk = ss.cmdSensorsUnitLeft(muCmd.SetONOrOFF)
				} else {
					leftOk = true
				}

				if ss.RightSensUint != nil {
					rightOk = ss.cmdSensorsUnitRight(muCmd.SetONOrOFF)
				} else {
					rightOk = true
				}

				if leftOk && rightOk {
					muCmd.Completed()
				} else {
					muCmd.SetErr(errors.New("Failed"))
				}
			}
		}
	}
}

// //区分开雷达测速的数据 很难支持同时过滤 暂时保留该类代码
// func (ss *ShipSpeed) WorkDriverFn(ctx context.Context) {
// 	var leftMsgChan, rightMsgChan <-chan any
// 	leftMsgChan = nil
// 	rightMsgChan = nil
// 	if ss.LeftSensUnit != nil {
// 		leftMsgChan = ss.LeftSensUnit.Radar.MsgChan()
// 	}
// 	if ss.RightSensUint != nil {
// 		rightMsgChan = ss.RightSensUint.Radar.MsgChan()
// 	}
// 	ss.Logger.Sugar().Debugf("WorkDriverFn Started : ---- OK ----,%s", ss.Info())
// 	for {
// 		runtime.Gosched()
// 		select {
// 		case <-ss.Ctrl().Context().Done():
// 			{
// 				err := ss.Ctrl().Context().Err()
// 				ss.Logger.Sugar().Debugf("WorkDriverFn Exit : err : %+v  ---------- OK ------,%s", err, ss.Info())
// 				return
// 			}
// 		case leftMsg, ok := <-leftMsgChan:
// 			{
// 				if !ok {
// 					ss.Logger.Sugar().Warnf("Left-Radar-msg-chan receive err, leftMsgChan Closed.%s", ss.Info())
// 					continue
// 				}
// 				ss.receiveRadarDataLeft(leftMsg)
// 			}
// 		case rightMsg, ok := <-rightMsgChan:
// 			{
// 				if !ok {
// 					ss.Logger.Sugar().Warnf("Right-Radar-msg-chan receive err, rightMsgChan Closed. %s", ss.Info())
// 					continue
// 				}
// 				ss.receiveRadarDataRight(rightMsg)
// 			}
// 		}
// 	}
// }

func (ss *ShipSpeed) LeftWorkDriverFn(ctx context.Context) {
	var leftMsgChan <-chan any
	leftMsgChan = nil
	if ss.LeftSensUnit != nil {
		leftMsgChan = ss.LeftSensUnit.Radar.MsgChan()
	}
	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("LeftWorkDriverFn Started : ---- OK ----.%s", ss.Info())
	}
	for {
		runtime.Gosched()
		select {
		case <-ss.Ctrl().Context().Done():
			{
				err := ss.Ctrl().Context().Err()
				if ss.TrackingSt.EnableDebug {
					ss.Logger.Sugar().Debugf("LeftWorkDriverFn Exit : err : %+v  ---------- OK ------.%s", err, ss.Info())
				}
				return
			}
		case leftMsg, ok := <-leftMsgChan:
			{
				if !ok {
					ss.Logger.Sugar().Warnf("Left-Radar-msg-chan receive err, leftMsgChan Closed.%s", ss.Info())
					continue
				}
				ss.receiveRadarDataLeft(leftMsg)
			}
		}
	}
}

func (ss *ShipSpeed) RightWorkDriverFn(ctx context.Context) {
	var rightMsgChan <-chan any
	rightMsgChan = nil
	if ss.RightSensUint != nil {
		rightMsgChan = ss.RightSensUint.Radar.MsgChan()
	}

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("RightWorkDriverFn Started : ---- OK ----.%s", ss.Info())
	}
	for {
		runtime.Gosched()
		select {
		case <-ss.Ctrl().Context().Done():
			{
				err := ss.Ctrl().Context().Err()
				if ss.TrackingSt.EnableDebug {
					ss.Logger.Sugar().Debugf("RightWorkDriverFn Exit : err : %+v  ---------- OK ------.%s", err, ss.Info())
				}
				return
			}
		case rightMsg, ok := <-rightMsgChan:
			{
				if !ok {
					ss.Logger.Sugar().Warnf("Right-Radar-msg-chan receive err, rightMsgChan Closed.%s", ss.Info())
					continue
				}
				ss.receiveRadarDataRight(rightMsg)
			}
		}
	}
}

func (ss *ShipSpeed) receiveRadarDataLeft(leftMsg any) (result bool) {
	rdu, ok := leftMsg.(*v0826.RadarDataUnit)
	if !ok {
		ss.Logger.Sugar().Warnf("Left Radar msg transform err:%+v.%s", rdu, ss.Info())
	}

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("Left Radar receive:%+v.%s", rdu, ss.Info())
	}

	//判断通航状态是否是运营有效的 并获取
	scheduleId := ss.TrackingSt.ScheduleId.GetScheduleId()
	navs, navsValid := ss.PredicateSchedule.GetIsValidNavState()
	if !navsValid {
		ss.Logger.Sugar().Errorf("navs,navsValid:%s,%t return.%s", navs.String(), navsValid, ss.Info())

		return false
	}

	//判断船闸状态是否是 船舶测速系统有效的 并获取
	navls, navlsValid := ss.PredicateSchedule.GetIsValidShipSpeedNavlState()
	if !navlsValid {
		ss.Logger.Sugar().Errorf("navls,navlsShipSpeedValid:%s,%t return.%s", navls.String(), navlsValid, ss.Info())

		return false
	}

	speed := rdu.Ships[0].Speed
	if speed < 0 {
		speed = -1 * speed
	}

	overSpeed := 0
	//超速的判断需要从数据库获取
	if ss.TrackingSt.Ref.SpeedSetupWrap.CheckOverSpeed(float64(rdu.Ships[0].Distance), float64(rdu.Ships[0].Speed)) {
		overSpeed = 1
	}

	if ss.TrackingSt.Ref.AnimationWrap != nil {
		//设置动画需要的状态
		ss.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(overSpeed, float64(rdu.Ships[0].Distance), float64(speed))
	}

	// 发布playscreen 需要的速度数据
	ShipSpeeds := gzb_schedule.NewShipSpeeds()
	ShipSpeeds.LeftSpeed.IsValid = true
	ShipSpeeds.LeftSpeed.Speed = speed
	ShipSpeeds.LeftSpeed.OverSpeed = (overSpeed == 1)
	ShipSpeeds.RightSpeed = nil
	ss.EventChans.Publish(gzb_schedule.MsgTopic_PlayScreen_ShipSpeeds, ShipSpeeds)

	var navsp gzb_db_web.NavSpeedRaw
	navsp.NavlockId = int64(ss.NavlockIdentifier)
	navsp.ScheduleId = scheduleId
	navsp.ScheduleStatus = int64(navls.TransformToWebDefine())
	navsp.Speed = float64(rdu.Ships[0].Speed)
	if navsp.Speed < 0 {
		navsp.Speed = -1 * navsp.Speed
	}
	navsp.Distance = float64(rdu.Ships[0].Distance)
	if overSpeed == 1 {
		navsp.Warn = "超速"
	}
	navsp.RadarTag = ss.NavlockLocation.String() + "左"

	if ss.PwebDB != nil {
		err := navsp.SaveToDB(ss.PwebDB, ss.Logger)
		if err != nil {
			ss.Logger.Sugar().Errorf("save to web db err:%+v.%s", err, ss.Info())
		}
	}
	if ss.LDB != nil {
		err := navsp.SaveToDB(ss.LDB, ss.Logger)
		if err != nil {
			ss.Logger.Sugar().Errorf("save to sqlite db err:%+v.%s", err, ss.Info())
		}
	}

	return true
}

func (ss *ShipSpeed) receiveRadarDataRight(rightMsg any) (result bool) {

	rdu, ok := rightMsg.(*v0826.RadarDataUnit)
	if !ok {
		ss.Logger.Sugar().Warnf("Right Radar msg transform err:%+v.%s", rdu, ss.Info())
		return false
	}

	if ss.TrackingSt.EnableDebug {
		ss.Logger.Sugar().Debugf("Right Radar receive:%+v.%s", rdu, ss.Info())
	}

	//判断通航状态是否是运营有效的 并获取
	scheduleId := ss.TrackingSt.ScheduleId.GetScheduleId()
	navs, navsValid := ss.PredicateSchedule.GetIsValidNavState()
	if !navsValid {
		ss.Logger.Sugar().Errorf("navs,navsValid:%s,%t return.%s", navs.String(), navsValid, ss.Info())

		return false
	}

	//判断船闸状态是否是 船舶测速系统有效的 并获取
	navls, navlsValid := ss.PredicateSchedule.GetIsValidShipSpeedNavlState()
	if !navlsValid {
		ss.Logger.Sugar().Errorf("navls,navlsShipSpeedValid:%s,%t return.%s", navls.String(), navlsValid, ss.Info())

		return false
	}

	speed := rdu.Ships[0].Speed
	if speed < 0 {
		speed = -1 * speed
	}
	overSpeed := 0
	//超速的判断需要从数据库获取
	if ss.TrackingSt.Ref.SpeedSetupWrap.CheckOverSpeed(float64(rdu.Ships[0].Distance), float64(speed)) {
		overSpeed = 1
	}

	if ss.TrackingSt.Ref.AnimationWrap != nil {
		//设置动画需要的状态
		ss.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(overSpeed, float64(rdu.Ships[0].Distance), float64(speed))
	}

	// 发布playscreen 需要的速度数据
	ShipSpeeds := gzb_schedule.NewShipSpeeds()
	ShipSpeeds.RightSpeed.IsValid = true
	ShipSpeeds.RightSpeed.Speed = speed
	ShipSpeeds.RightSpeed.OverSpeed = (overSpeed == 1)
	ShipSpeeds.LeftSpeed = nil
	ss.EventChans.Publish(gzb_schedule.MsgTopic_PlayScreen_ShipSpeeds, ShipSpeeds)

	var navsp gzb_db_web.NavSpeedRaw
	navsp.NavlockId = int64(ss.NavlockIdentifier)
	navsp.ScheduleId = scheduleId
	navsp.ScheduleStatus = int64(navls.TransformToWebDefine())
	navsp.Speed = float64(rdu.Ships[0].Speed)
	if navsp.Speed < 0 {
		navsp.Speed = -1 * navsp.Speed
	}
	speed64, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", navsp.Speed), 64)
	navsp.Speed = speed64
	navsp.Distance = float64(rdu.Ships[0].Distance)

	if overSpeed == 1 {
		navsp.Warn = "超速"
	}

	navsp.RadarTag = ss.NavlockLocation.String() + "右"
	if ss.PwebDB != nil {
		err := navsp.SaveToDB(ss.PwebDB, ss.Logger)
		if err != nil {
			ss.Logger.Sugar().Errorf("save to web db err:%+v.%s", err, ss.Info())
		}
	}

	if ss.LDB != nil {
		err := navsp.SaveToDB(ss.LDB, ss.Logger)
		if err != nil {
			ss.Logger.Sugar().Errorf("save to sqlite db err:%+v.%s", err, ss.Info())
		}
	}

	return true
}

func (ss *ShipSpeed) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		ss.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (ss *ShipSpeed) Start() (err error) {
	if ss.CptSt == nil || ss.ModelSysSt == nil {
		return fmt.Errorf("is not initialized.%s", ss.CmptInfo())
	} else {
		if err := ss.CptSt.Validate(); err != nil {
			return err
		}
	}

	if ss.State.Load().(bool) {
		return fmt.Errorf("is already Started.%s", ss.Info())
	}

	//------------- worker -----------------
	// 只有MeasureUnit初始化完成后 才能取到合适的参数 所以在Start()里面实现
	if ss.Cnf != nil {
		// -------------left worker -------------------
		if ss.Cnf.LeftSetup != nil {
			cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
				cmpt.IdName("RadarMsgReceiveLeft"),
				0,
				ss.Ctrl().ForkCtxWg())

			dr := navl_md.NewBaseCptCtx(cs,
				navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
					ss.TrackingSt.NavlockIdentifier),
					"",
					"",
					"RadarMsgReceive-Left"))

			workdr := drs_wd.NewWorkDvr(dr, ss.TrackingSt)
			workdr.Fn = ss.LeftWorkDriverFn
			ss.leftwkd = workdr
			ss.leftwkd.EnableDebug = ss.TrackingSt.EnableDebug
		}
		// -------------right worker -------------------
		if ss.Cnf.RightSetup != nil {
			cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
				cmpt.IdName("RadarMsgReceiveRight"),
				0,
				ss.Ctrl().ForkCtxWg())

			dr := navl_md.NewBaseCptCtx(cs,
				navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
					ss.TrackingSt.NavlockIdentifier),
					"",
					"",
					"RadarMsgReceive-Right"))

			workdr := drs_wd.NewWorkDvr(dr, ss.TrackingSt)
			workdr.Fn = ss.RightWorkDriverFn
			ss.rightwkd = workdr
			ss.rightwkd.EnableDebug = ss.TrackingSt.EnableDebug
		}
	}
	// --------------------------------------------
	ss.Logger.Sugar().Infof("--- Starting ---.%s", ss.Info())
	ss.Ctrl().WaitGroup().StartingWait(ss)

	if ss.Cpts != nil && ss.Cpts.Len() > 0 {
		err = ss.Cpts.Start()
		if err != nil {
			ss.Logger.Sugar().Errorf("Components Start() error : %+v.%s", err, ss.Info())
		}
	}

	ss.State.Store(true)
	ss.Logger.Sugar().Infof("--- Started ---.%s", ss.Info())
	return
}

func (ss *ShipSpeed) Stop() (err error) {
	if !ss.State.Load().(bool) {
		return fmt.Errorf("is already Stopped.%s", ss.Info())
	}

	ss.Logger.Sugar().Infof("--- Stopping ---.%s", ss.Info())

	ss.Ctrl().Cancel()
	<-ss.Ctrl().Context().Done()

	if ss.Cpts != nil && ss.Cpts.Len() > 0 {
		err := ss.Cpts.Stop()
		if err != nil {
			ss.Logger.Sugar().Errorf("Stop() error : %+v.%s", err, ss.Info())
		}
	}

	ss.State.Store(false)
	ss.Logger.Sugar().Infof("--- Stopped ---.%s", ss.Info())
	return
}

// todo: 读取配置文件 创建组件
func (ss *ShipSpeed) InitComponentsFromConfig(conf *gzb_cf.SpeedMeasureUnitSetup) error {
	//判断配置文件中是否有 然后判断是否已经创建组件
	navlockId := fmt.Sprintf("#%d", ss.TrackingSt.NavlockIdentifier)
	//------------- 初始化SensorsDvr-----------------
	if ss.Cnf.LeftSetup != nil && ss.Cpts.Cpt(gzb_md_sens.SensorsUnitId_Left) == nil {
		mcs := ss.CptSt.Fork(gzb_md_sens.SensorsUnitKind, gzb_md_sens.SensorsUnitId_Left)
		mss := navl_md.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"",
			string(gzb_md_sens.SensorsUnitKind), string(gzb_md_sens.SensorsUnitId_Left))
		dr := navl_md.NewBaseCptCtx(mcs, mss)
		trks := ss.TrackingSt.CopyNew()

		switch trks.NavlockLocation {
		case gzb_schedule.NavlockLocUp:
			trks.NavlockLocation = gzb_schedule.NavlockLocUpLeft
		case gzb_schedule.NavlockLocDown:
			trks.NavlockLocation = gzb_schedule.NavlockLocDownLeft
		}

		ssu := gzb_md_sens.NewSensorsDvr(dr, trks)
		ssu.Cnf = ss.Cnf.LeftSetup
		ssu.InitComponentsFromConfig(conf.LeftSetup)
		ss.Cpts.AddCpts(ssu)
		ss.LeftSensUnit = ssu
	}
	// --------------------------------------------

	//------------- 初始化SensorsUnit-----------------
	if ss.Cnf.RightSetup != nil && ss.Cpts.Cpt(gzb_md_sens.SensorsUnitId_Right) == nil {
		mcs := ss.CptSt.Fork(gzb_md_sens.SensorsUnitKind, gzb_md_sens.SensorsUnitId_Right)
		mss := navl_md.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"",
			string(gzb_md_sens.SensorsUnitKind), string(gzb_md_sens.SensorsUnitId_Right))
		dr := navl_md.NewBaseCptCtx(mcs, mss)
		trks := ss.TrackingSt.CopyNew()

		switch trks.NavlockLocation {
		case gzb_schedule.NavlockLocUp:
			trks.NavlockLocation = gzb_schedule.NavlockLocUpRight
		case gzb_schedule.NavlockLocDown:
			trks.NavlockLocation = gzb_schedule.NavlockLocDownRight
		}

		ssu := gzb_md_sens.NewSensorsDvr(dr, trks)
		ssu.Cnf = ss.Cnf.RightSetup
		ssu.InitComponentsFromConfig(conf.RightSetup)
		ss.Cpts.AddCpts(ssu)
		ss.RightSensUint = ssu
	}
	// --------------------------------------------
	return nil
}
