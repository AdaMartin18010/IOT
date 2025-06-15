package factory

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"time"

	cm "navigate/common"
	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	navl_model "navigate/iot/navlock/model"

	gzb_cf "navigate/config/iot/navlock/gezhouba"
	gzb_shipspeed "navigate/iot/navlock/gezhouba/model/driver/shipspeed"
	gzb_web "navigate/iot/navlock/gezhouba/model/driver/web"
	gzb_navls "navigate/iot/navlock/gezhouba/model/server/navlstatus"
	gzb_playscreen "navigate/iot/navlock/gezhouba/model/server/playscreen"
	gzb_stopline "navigate/iot/navlock/gezhouba/model/server/stopline"
	gzb_schedule "navigate/iot/navlock/gezhouba/schedule"
)

var (
	//Verify Satisfies interfaces
	_ mdl.WorkerRecover = (*Navlock)(nil)
	_ cmpt.Cpt          = (*Navlock)(nil)
	_ cmpt.CptsOperator = (*Navlock)(nil)
	_ cmpt.CptComposite = (*Navlock)(nil)
	_ cmpt.CptRoot      = (*Navlock)(nil)
)

const (
	cmdWaitInterval = 1000 * time.Millisecond
)

type cmdSetInfo struct {
	ConnectOk  bool // 是否是处于连接状态
	CurrentOn  bool // 当前的状态是否打开
	CmdSetOn   bool // 命令设置的状态
	NeedCmdSet bool // 是否需要执行命令
}

func (cm cmdSetInfo) FmtStr() string {
	return fmt.Sprintf("CmdSetInfo:%+v", cm)
}

// 对应 --- 船闸 = { ShipSpeedMeasure:[上游:[左,右],下游:[左,右]],web,playscreen,stopline,navlstatus,platform}
type Navlock struct {
	*navl_model.NavlockModel
	*gzb_schedule.TrackingSt

	NavlockConf         *gzb_cf.NavlockConf
	radarOverload       bool
	msgChanNavlockState <-chan any
	msgChanNavState     <-chan any

	upCmdSetInfo   *cmdSetInfo
	UpUnit         *gzb_shipspeed.ShipSpeed
	downCmdSetInfo *cmdSetInfo
	DownUnit       *gzb_shipspeed.ShipSpeed
}

func NewNavlockSvr(mcs *navl_model.CptSt, mss *navl_model.ModelSysSt, trks *gzb_schedule.TrackingSt) *Navlock {
	temp := &Navlock{
		TrackingSt:     trks,
		NavlockModel:   navl_model.NewNavlockModel(mcs, mss),
		radarOverload:  false,
		upCmdSetInfo:   &cmdSetInfo{},
		downCmdSetInfo: &cmdSetInfo{},
	}
	temp.WorkerRecover = temp
	return temp
}

func (nl *Navlock) Work() error {
	nl.msgChanNavlockState = nl.EventChans.Subscribe(gzb_schedule.MsgTopic_NavlockStatus)
	nl.msgChanNavState = nl.EventChans.Subscribe(gzb_schedule.MsgTopic_NavStatus)
	defer func() {
		nl.EventChans.UnSubscribe(gzb_schedule.MsgTopic_NavlockStatus, nl.msgChanNavlockState)
		nl.EventChans.UnSubscribe(gzb_schedule.MsgTopic_NavStatus, nl.msgChanNavState)
	}()

	//创建船闸状态检测的时间间隔
	timeIntervalCheckNavl := time.Duration(1) * time.Second
	tickerCheckerNavl := time.NewTicker(timeIntervalCheckNavl)
	tickerCheckerNavlSeted := false
	defer tickerCheckerNavl.Stop()
	tickerCheckerNavl.Stop()
	select {
	case <-tickerCheckerNavl.C:
	default:
	}

	timeIntervalRadarRuntimeChecker := time.Duration(nl.NavlockConf.RadarMaxRuntime) * time.Minute
	tickerRadarRuntimeChecker := time.NewTicker(timeIntervalRadarRuntimeChecker)
	tickerRadarRuntimeCheckerSeted := false
	defer tickerRadarRuntimeChecker.Stop()
	tickerRadarRuntimeChecker.Stop()
	select {
	case <-tickerRadarRuntimeChecker.C:
	default:
	}

	timeIntervalRadarCoolingChecker := time.Duration(nl.NavlockConf.RadarCoolingInterval) * time.Minute
	tickerRadarCoolingChecker := time.NewTicker(timeIntervalRadarCoolingChecker)
	tickerRadarCoolingCheckerSeted := false
	defer tickerRadarCoolingChecker.Stop()
	tickerRadarCoolingChecker.Stop()
	select {
	case <-tickerRadarCoolingChecker.C:
	default:
	}

	runtime.Gosched()
	for {
		select {
		case <-nl.Ctrl().Context().Done():
			{
				if err := nl.Ctrl().Context().Err(); err != nil {
					if !errors.Is(err, context.Canceled) {
						return err
					}
				}
				return nil
			}
		case <-tickerRadarRuntimeChecker.C:
			{
				// drain event
				select {
				case <-tickerRadarRuntimeChecker.C:
				default:
				}
				// 防止打开和关闭定时器 可能出现的触发 异步事件
				if nl.upCmdSetInfo.CurrentOn || nl.downCmdSetInfo.CurrentOn {
					nl.radarOverload = true
				}

				if tickerRadarRuntimeCheckerSeted {
					tickerRadarRuntimeChecker.Stop()
					select {
					case <-tickerRadarRuntimeChecker.C:
					default:
					}
					tickerRadarRuntimeCheckerSeted = false
					if nl.TrackingSt.EnableDebug {
						nl.Logger.Sugar().Debugf(`Stop check radar Runtime OverLoad checker.%s`, nl.Info())
					}
				}
			}
		case <-tickerRadarCoolingChecker.C:
			{
				// drain event
				select {
				case <-tickerRadarCoolingChecker.C:
				default:
				}
				nl.radarOverload = false

				if tickerRadarCoolingCheckerSeted {
					tickerRadarCoolingChecker.Stop()
					select {
					case <-tickerRadarCoolingChecker.C:
					default:
					}
					tickerRadarCoolingCheckerSeted = false
					if nl.TrackingSt.EnableDebug {
						nl.Logger.Sugar().Debugf(`Stop check radar Cooling checker.%s`, nl.Info())
					}
				}

			}
		case <-tickerCheckerNavl.C:
			{
				// drain event
				select {
				case <-tickerCheckerNavl.C:
				default:
				}

				nl.PredicateSwitchInterval()
				nl.CheckingShipSpeed()
				// 在上下游连接都正常的情况 只要其中一个需要执行cmd 就持续检查
				if nl.upCmdSetInfo.ConnectOk && nl.downCmdSetInfo.ConnectOk {
					if nl.upCmdSetInfo.NeedCmdSet || nl.downCmdSetInfo.NeedCmdSet {

						if nl.TrackingSt.EnableDebug {
							nl.Logger.Sugar().Debugf(`UpCmdInfo:%s,DownCmdInfo:%s.Continue Checking cmd.%s`,
								nl.upCmdSetInfo.FmtStr(),
								nl.downCmdSetInfo.FmtStr(),
								nl.Info())
						}

						continue

					} else {

						if nl.TrackingSt.EnableDebug {
							//都连接上并且不再需要设置cmd的情况下 取消持续的检查
							nl.Logger.Sugar().Debugf(`UpCmdInfo:%s,DownCmdInfo:%s.Stop checking cmd.%s`,
								nl.upCmdSetInfo.FmtStr(),
								nl.downCmdSetInfo.FmtStr(),
								nl.Info())
						}

						//不需要再次检查 就取消掉
						tickerCheckerNavl.Stop()
						select {
						case <-tickerCheckerNavl.C:
						default:
						}
						tickerCheckerNavlSeted = false

						// 判断上下游的情况  打开和关闭雷达运行的过载定时
						if !nl.radarOverload {
							// 上下游只要有一个打开的情况下 就启动过载定时器
							if nl.upCmdSetInfo.CurrentOn || nl.downCmdSetInfo.CurrentOn {
								if !tickerRadarRuntimeCheckerSeted {
									tickerRadarRuntimeChecker.Reset(timeIntervalRadarRuntimeChecker)
									tickerRadarRuntimeCheckerSeted = true
									if nl.TrackingSt.EnableDebug {
										nl.Logger.Sugar().Debugf(`Start check radar Runtime OverLoad checker.%s`, nl.Info())
									}
								}
							}
							// 上下游都关闭的情况下 就关闭过载定时器
							if !nl.upCmdSetInfo.CurrentOn && !nl.downCmdSetInfo.CurrentOn {
								if tickerRadarRuntimeCheckerSeted {
									tickerRadarRuntimeChecker.Stop()
									select {
									case <-tickerRadarRuntimeChecker.C:
									default:
									}
									tickerRadarRuntimeCheckerSeted = false
									if nl.TrackingSt.EnableDebug {
										nl.Logger.Sugar().Debugf(`Stop check radar Runtime OverLoad checker.%s`, nl.Info())
									}
								}
							}
						}

						// 判断雷达过载的情况下 打开冷却定时器
						if nl.radarOverload {
							// 上下游都关闭的情况下
							if !nl.upCmdSetInfo.CurrentOn && !nl.downCmdSetInfo.CurrentOn {
								//启动冷却的定时器
								if !tickerRadarCoolingCheckerSeted {
									tickerRadarCoolingChecker.Reset(timeIntervalRadarCoolingChecker)
									tickerRadarCoolingCheckerSeted = true

									if nl.TrackingSt.EnableDebug {
										nl.Logger.Sugar().Debugf(`Start check radar cooling checker.%s`, nl.Info())
									}
								}
							}
						}
					}
				}
			}
		case _, ok := <-nl.msgChanNavlockState:
			{
				if !ok {
					nl.Logger.Sugar().Infof(`Event chan closed.%s`, nl.Info())
					continue
				}

				if nl.TrackingSt.EnableDebug {
					nl.Logger.Sugar().Debugf(`receive navls. UpCmdInfo:%s,DownCmdInfo:%s.%s`,
						nl.upCmdSetInfo.FmtStr(),
						nl.downCmdSetInfo.FmtStr(),
						nl.Info())
				}

				nl.PredicateSwitchInterval()
				nl.CheckingShipSpeed()
				if nl.upCmdSetInfo.ConnectOk && nl.downCmdSetInfo.ConnectOk {
					//在上下游都连接的情况下 需要设置cmd
					if nl.upCmdSetInfo.NeedCmdSet || nl.downCmdSetInfo.NeedCmdSet {
						if !tickerCheckerNavlSeted {
							tickerCheckerNavl.Reset(timeIntervalCheckNavl)
							tickerCheckerNavlSeted = true

							if nl.TrackingSt.EnableDebug {
								nl.Logger.Sugar().Debugf(`start check cmd.%s`, nl.Info())
							}
						}
					}
				} else {
					//上下游只要一个未连接上 启动重试
					if !tickerCheckerNavlSeted {
						tickerCheckerNavl.Reset(timeIntervalCheckNavl)
						tickerCheckerNavlSeted = true

						if nl.TrackingSt.EnableDebug {
							nl.Logger.Sugar().Debugf(`start check cmd.%s`, nl.Info())
						}
					}
				}
			}
		case _, ok := <-nl.msgChanNavState:
			{
				if !ok {
					nl.Logger.Sugar().Infof(`Event chan closed.%s`, nl.Info())
					continue
				}

				if nl.TrackingSt.EnableDebug {
					nl.Logger.Sugar().Debugf(`receive navls. UpCmdInfo:%s,DownCmdInfo:%s.%s`,
						nl.upCmdSetInfo.FmtStr(),
						nl.downCmdSetInfo.FmtStr(),
						nl.Info())
				}

				nl.PredicateSwitchInterval()
				nl.CheckingShipSpeed()
				if nl.upCmdSetInfo.ConnectOk && nl.downCmdSetInfo.ConnectOk {
					//在上下游都连接的情况下 只要一个需要设置cmd 就启动重试
					if nl.upCmdSetInfo.NeedCmdSet || nl.downCmdSetInfo.NeedCmdSet {
						if !tickerCheckerNavlSeted {
							tickerCheckerNavl.Reset(timeIntervalCheckNavl)
							tickerCheckerNavlSeted = true

							if nl.TrackingSt.EnableDebug {
								nl.Logger.Sugar().Debugf(`start check cmd.%s`, nl.Info())
							}
						}
					}
				} else {
					//上下游只要一个未连接上 启动重试
					if !tickerCheckerNavlSeted {
						tickerCheckerNavl.Reset(timeIntervalCheckNavl)
						tickerCheckerNavlSeted = true

						if nl.TrackingSt.EnableDebug {
							nl.Logger.Sugar().Debugf(`start check cmd.%s`, nl.Info())
						}
					}
				}
			}
		}
	}
}

// 关注连接状态的情况下 执行命令
func (nl *Navlock) CheckingShipSpeed() bool {
	//获取上下游的状态 并且重置值 一旦需要设置 就发出命令
	upCmdOK := false
	downCmdOK := false
	nl.upCmdSetInfo.ConnectOk, nl.upCmdSetInfo.CurrentOn = nl.UpUnit.IsConnectedOn()
	if nl.upCmdSetInfo.ConnectOk {
		//对比状态是否需要设置
		nl.upCmdSetInfo.NeedCmdSet = (nl.upCmdSetInfo.CurrentOn != nl.upCmdSetInfo.CmdSetOn)
		if nl.upCmdSetInfo.NeedCmdSet {
			upCmdOK := nl.cmdShipSpeed(nl.UpUnit, nl.upCmdSetInfo)
			if !upCmdOK {
				nl.Logger.Sugar().Errorf(`UpShipSpeed send cmd fail. upCmdInfo:%+v.%s`, nl.upCmdSetInfo, nl.Info())
			}
		}

	} else {
		nl.Logger.Sugar().Errorf(`UpShipSpeed is not connected.%s`, nl.Info())
	}

	nl.downCmdSetInfo.ConnectOk, nl.downCmdSetInfo.CurrentOn = nl.DownUnit.IsConnectedOn()
	if nl.downCmdSetInfo.ConnectOk {
		//对比状态是否需要设置
		nl.downCmdSetInfo.NeedCmdSet = (nl.downCmdSetInfo.CurrentOn != nl.downCmdSetInfo.CmdSetOn)
		if nl.downCmdSetInfo.NeedCmdSet {
			downCmdOK := nl.cmdShipSpeed(nl.DownUnit, nl.downCmdSetInfo)
			if !downCmdOK {
				nl.Logger.Sugar().Errorf(`DownShipSpeed send cmd fail. downCmdInfo:%+v.%s`, nl.downCmdSetInfo, nl.Info())
			}
		}
	} else {
		nl.Logger.Sugar().Errorf(`DownShipSpeed is not connected.%s`, nl.Info())
	}

	return upCmdOK && downCmdOK
}

func (nl *Navlock) cmdShipSpeed(ssp *gzb_shipspeed.ShipSpeed, CmdSetInfo *cmdSetInfo) bool {
	Cmd := gzb_shipspeed.NewShipSpeedCmd()
	Cmd.SetONOrOFF = CmdSetInfo.CmdSetOn
	if !ssp.Commander(Cmd) {
		nl.Logger.Sugar().Warnf(`Commander(),false,CmdInfo:%+v.%s"`, CmdSetInfo, nl.Info())
		return false
	}
	ok := Cmd.WaitTimeout(cmdWaitInterval)
	if ok {
		return true
	} else {
		nl.Logger.Sugar().Errorf("Cmd.WaitTimeout() err:%+v,CmdInfo:%s.%+v", Cmd.Err(), CmdSetInfo.FmtStr(), nl.Info())
		return false
	}
}

func (nl *Navlock) PredicateSwitchInterval() {
	// 雷达过载的情况下 全部关闭
	if nl.radarOverload {
		nl.upCmdSetInfo.CmdSetOn = false
		nl.downCmdSetInfo.CmdSetOn = false

		if nl.TrackingSt.Ref.AnimationWrap != nil {
			nl.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(0, 0.0, 0.0)
			nl.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(0, 0.0, 0.0)
		}

		return
	}

	sid := nl.TrackingSt.ScheduleId.GetScheduleId()
	if len(sid) > 0 {
		navs := nl.TrackingSt.PredicateSchedule.GetNavStatus()
		navls := nl.TrackingSt.PredicateSchedule.GetNavlockStatus()

		if nl.TrackingSt.EnableDebug {
			nl.Logger.Sugar().Debugf(`[%s]:"PredicateSwitchInterval navl: [%s],[%s]",`, nl.Info(), navs.String(), navls.String())
		}

		upOnOrOff := false
		downOnOrOff := false

		switch navs {
		case gzb_schedule.NavUp:
			switch navls {
			case gzb_schedule.NavlockUpGoingIn:
				{
					upOnOrOff = true
					downOnOrOff = false
				}
			case gzb_schedule.NavlockUpGoInDone, gzb_schedule.NavlockUpGoOutDone:
				{
					upOnOrOff = false
					downOnOrOff = false

					if nl.TrackingSt.Ref.AnimationWrap != nil {
						nl.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(0, 0.0, 0.0)
						nl.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(0, 0.0, 0.0)
					}
				}
			case gzb_schedule.NavlockUpGoingOut:
				{
					upOnOrOff = false
					downOnOrOff = true
				}
			default:
				{
					upOnOrOff = false
					downOnOrOff = false

					if nl.TrackingSt.Ref.AnimationWrap != nil {
						nl.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(0, 0.0, 0.0)
						nl.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(0, 0.0, 0.0)
					}
				}
			}
		case gzb_schedule.NavDown:
			switch navls {
			case gzb_schedule.NavlockDownGoingIn:
				{
					upOnOrOff = false
					downOnOrOff = true
				}
			case gzb_schedule.NavlockDownGoInDone, gzb_schedule.NavlockDownGoOutDone:
				{
					upOnOrOff = false
					downOnOrOff = false

					if nl.TrackingSt.Ref.AnimationWrap != nil {
						nl.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(0, 0.0, 0.0)
						nl.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(0, 0.0, 0.0)
					}
				}
			case gzb_schedule.NavlockDownGoingOut:
				{
					upOnOrOff = true
					downOnOrOff = false
				}
			default:
				{
					upOnOrOff = false
					downOnOrOff = false

					if nl.TrackingSt.Ref.AnimationWrap != nil {
						nl.TrackingSt.Ref.AnimationWrap.SetLeftRadarInfo(0, 0.0, 0.0)
						nl.TrackingSt.Ref.AnimationWrap.SetRightRadarInfo(0, 0.0, 0.0)
					}
				}
			}
		}

		//推断需要开启的时候 只判断当前应该设置的值 而不关注连接状态
		nl.upCmdSetInfo.CmdSetOn = upOnOrOff
		nl.downCmdSetInfo.CmdSetOn = downOnOrOff
	}
}

func (nl *Navlock) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		nl.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (nl *Navlock) Start() (err error) {
	if nl.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", nl.CmptInfo())
	}

	nl.Logger.Sugar().Infoln("--------------------------------- Starting ---------------------------------")
	defer nl.Logger.Sugar().Infoln("--------------------------------- Started ---------------------------------")

	nl.Ctrl().WaitGroup().StartingWait(nl.WorkerRecover)
	if nl.Cpts != nil && nl.Cpts.Len() > 0 {
		err = nl.Cpts.Start()
		if err != nil {
			nl.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}

	nl.State.Store(true)
	//--------------- 创建基础的事件发布消息队列 -------------------------
	if nl.EventChans != nil {
		nl.Logger.Sugar().Infof("EventChan:%+v,EventTopics: %s", nl.EventChans, nl.EventChans.Topics())
	}
	//----------------------------------------------------------------

	return
}

func (nl *Navlock) Stop() (err error) {
	if !nl.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", nl.CmptInfo())
	}

	nl.Logger.Sugar().Infoln("--------------------------------- Stopping ---------------------------------")
	defer nl.Logger.Sugar().Infoln("--------------------------------- Stopped ---------------------------------")

	nl.Ctrl().Cancel()
	<-nl.Ctrl().Context().Done()
	if nl.Cpts != nil && nl.Cpts.Len() > 0 {
		err := nl.Cpts.Stop()
		if err != nil {
			nl.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	if nl.EventChans != nil {
		nl.Logger.Sugar().Debugf("EventChan:%+v,EventTopics: %s before close.", nl.EventChans, nl.EventChans.Topics())
		nl.EventChans.Close()
		nl.Logger.Sugar().Debugf("EventChan:%+v,EventTopics: %s after close.", nl.EventChans, nl.EventChans.Topics())
		nl.EventChans.WaitAsync()
	}

	nl.State.Store(false)
	return err
}

// todo: 读取配置文件 创建组件
func (nl *Navlock) InitComponentsFromConfig(conf *gzb_cf.NavlockConf) error {

	if nl.TrackingSt.EnableDebug {
		nl.Logger.Sugar().Debugf("%s is configured From Config: %+v", nl.CmptInfo(), conf)
	}

	if !conf.Enable {
		return nil
	}

	navlockId := fmt.Sprintf("#%d", conf.Identifier)
	// // 上传平台不再使用
	// if conf.PlatformHttp != nil && conf.PlatformHttp.Enable {
	// 	mcs := nl.CptSt.Fork(gzb_platform.PlatformKind, cmpt.IdName(navlockId))
	// 	mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId, "", string(gzb_platform.PlatformKind), "")
	// 	dr := navl_model.NewBaseCptCtx(mcs, mss)
	// 	trks := nl.TrackingSt

	// 	pltf := gzb_platform.NewPlatformSvr(dr, trks)
	// 	pltf.Cnf = conf.PlatformHttp
	// 	nl.Cpts.AddCpts(pltf)
	// }

	if conf.NavlStatusHttp != nil && conf.NavlStatusHttp.Enable {
		mcs := nl.CptSt.Fork(gzb_navls.NavlStatusKind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId, "", string(gzb_navls.NavlStatusKind), "")
		dr := navl_model.NewBaseCptCtx(mcs, mss)
		trks := nl.TrackingSt
		mntd := gzb_navls.NewNavlStatusSvr(dr, trks)
		mntd.Cnf = conf.NavlStatusHttp
		nl.Cpts.AddCpts(mntd)
	}

	if conf.StopLineHttp != nil && conf.StopLineHttp.Enable {
		mcs := nl.CptSt.Fork(gzb_stopline.StoplineKind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId, "", string(gzb_stopline.StoplineKind), "")
		dr := navl_model.NewBaseCptCtx(mcs, mss)
		trks := nl.TrackingSt
		stpl := gzb_stopline.NewStoplineSvr(dr, trks)

		stpl.UpCnf = conf.StopLineHttp.UpHttp
		stpl.DownCnf = conf.StopLineHttp.DownHttp
		nl.Cpts.AddCpts(stpl)
	}

	if err := nl.InitShipSpeedMeasureUnitsFromConfig(conf); err != nil {
		return err
	}

	//至少有组件的情况下 才创建webdr
	if nl.Cpts.Len() > 1 {
		mcs := nl.CptSt.Fork(gzb_web.WebKind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId, "", "", string(gzb_web.WebKind))
		dr := navl_model.NewBaseCptCtx(mcs, mss)
		trks := nl.TrackingSt
		web := gzb_web.NewWebDvr(dr, trks)
		web.Cnf = conf
		nl.Cpts.AddCpts(web)
	}

	return nil
}

func (nl *Navlock) InitShipSpeedMeasureUnitsFromConfig(conf *gzb_cf.NavlockConf) error {

	navlockId := fmt.Sprintf("#%d", conf.Identifier)

	if conf.NavlockShipSpeedConf.PlayScreenSetup != nil && conf.NavlockShipSpeedConf.PlayScreenSetup.Enable {
		if conf.NavlockShipSpeedConf.PlayScreenSetup.EnableAutoTest {
			if conf.NavlockShipSpeedConf.PlayScreenSetup.AutoTestDirector != "" {
				//确定在服务执行中当前目录创建 后续根据闸次创建目录
				conf.NavlockShipSpeedConf.PlayScreenSetup.AutoTestDirector = cm.DealWithExecutingCurrentFilePath(conf.NavlockShipSpeedConf.PlayScreenSetup.AutoTestDirector)
				if err := cm.CreatePathDir(conf.NavlockShipSpeedConf.PlayScreenSetup.AutoTestDirector); err != nil {
					nl.Logger.Sugar().Fatalf(" Create directory err: %+v", err)
				}
			}
		}
		// TODO: playscreen 播放屏 暂时不启用
		mcs := nl.CptSt.Fork(gzb_playscreen.Kind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId, "", string(gzb_playscreen.Kind), "")
		sr := navl_model.NewBaseCptCtx(mcs, mss)

		trks := nl.TrackingSt.CopyNew()

		PlayScreen := gzb_playscreen.NewPlayScreenSvr(sr, trks)
		PlayScreen.Cnf = conf.NavlockShipSpeedConf.PlayScreenSetup
		nl.Cpts.AddCpts(PlayScreen)

		if err := PlayScreen.ReadConfigJson(); err != nil {
			nl.Logger.Sugar().Fatalf(" Create PlayScreen Component err: %+v", err)
		}
	}

	if conf.NavlockShipSpeedConf.UpSetup != nil {
		mcs := nl.CptSt.Fork(gzb_shipspeed.Kind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"", string(gzb_shipspeed.Kind), string(gzb_shipspeed.ShipSpeedId_Up))
		dr := navl_model.NewBaseCptCtx(mcs, mss)

		trks := nl.TrackingSt.CopyNew()
		trks.NavlockLocation = gzb_schedule.NavlockLocUp

		mus := gzb_shipspeed.NewShipSpeedDvr(dr, trks)
		mus.Cnf = conf.NavlockShipSpeedConf.UpSetup

		mus.InitComponentsFromConfig(mus.Cnf)
		nl.UpUnit = mus
		nl.Cpts.AddCpts(mus)
	}

	if conf.NavlockShipSpeedConf.DownSetup != nil {
		mcs := nl.CptSt.Fork(gzb_shipspeed.Kind, cmpt.IdName(navlockId))
		mss := navl_model.NewModelSysSt(mcs.CmptInfo(), navlockId,
			"", string(gzb_shipspeed.Kind), string(gzb_shipspeed.ShipSpeedId_Down))
		dr := navl_model.NewBaseCptCtx(mcs, mss)

		trks := nl.TrackingSt.CopyNew()
		trks.NavlockLocation = gzb_schedule.NavlockLocDown

		mus := gzb_shipspeed.NewShipSpeedDvr(dr, trks)
		mus.Cnf = conf.NavlockShipSpeedConf.DownSetup
		mus.InitComponentsFromConfig(mus.Cnf)
		nl.DownUnit = mus
		nl.Cpts.AddCpts(mus)
	}

	return nil
}
