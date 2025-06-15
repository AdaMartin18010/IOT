package web

import (
	"context"
	"fmt"
	"runtime"
	"time"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

var (
	_ cm.WorkerRecover  = (*Web)(nil)
	_ cmpt.Cpt          = (*Web)(nil)
	_ cmpt.CptsOperator = (*Web)(nil)
	_ cmpt.CptComposite = (*Web)(nil)
	_ cmpt.CptRoot      = (*Web)(nil)
)

const (
	WebKind = cmpt.KindName("web")
)

// 实现和web对接的处理逻辑
type Web struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	Cnf              *gzb_cf.NavlockConf
	tmpNavState      gzb_sche.NavStatus //通航调度状态 --判断完整的一个航程-上行或者下行
	tmpSid           string
	msgChanNavState  <-chan any
	msgChanNavlState <-chan any
}

func NewWebDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Web {
	tmp := &Web{
		BaseCptCtx: dr,
		TrackingSt: trs,
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (wd *Web) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		wd.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (wd *Web) Start() (err error) {
	if wd.CptSt == nil || wd.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", wd.CmptInfo())
	} else {
		if err = wd.CptSt.Validate(); err != nil {
			return err
		}
	}

	if wd.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", wd.CmptInfo())
	}

	if wd.Cpts != nil && wd.Cpts.Len() > 0 {
		err = wd.Cpts.Stop()
		if err != nil {
			wd.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}

	wd.Logger.Sugar().Infoln("--- Starting ---")
	wd.Ctrl().WaitGroup().StartingWait(wd.WorkerRecover)
	wd.State.Store(true)
	wd.Logger.Sugar().Infoln("--- Started ---")
	return nil
}

func (wd *Web) Stop() (err error) {
	if !wd.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", wd.Info())
	}

	wd.Ctrl().Cancel()
	wd.Logger.Sugar().Infoln("--- Stopping ---")
	<-wd.Ctrl().Context().Done()

	if wd.Cpts != nil && wd.Cpts.Len() > 0 {
		err = wd.Cpts.Stop()
		if err != nil {
			wd.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	wd.State.Store(false)
	wd.Logger.Sugar().Infoln("--- Stopped ---")
	return
}

func (wd *Web) Work() error {
	wd.Logger.Sugar().Infof(`[Dr:%s,%s]:"Work Starting"`,
		"WebDriver", wd.Info())

	wd.msgChanNavState = wd.EventChans.Subscribe(gzb_sche.MsgTopic_NavStatus)
	wd.msgChanNavlState = wd.EventChans.Subscribe(gzb_sche.MsgTopic_NavlockStatus)
	defer func() {
		wd.EventChans.UnSubscribe(gzb_sche.MsgTopic_NavStatus, wd.msgChanNavState)
		wd.EventChans.UnSubscribe(gzb_sche.MsgTopic_NavlockStatus, wd.msgChanNavlState)
	}()

	timeIntervaSetuplCheck := time.Duration(1500) * time.Millisecond
	tickerSetupChecker := time.NewTicker(timeIntervaSetuplCheck)
	defer tickerSetupChecker.Stop()

	timeIntervalOperatorCheck := time.Duration(1500) * time.Millisecond
	tickerOperatorChecker := time.NewTicker(timeIntervalOperatorCheck)
	defer tickerOperatorChecker.Stop()

	for {
		runtime.Gosched()
		select {
		case <-wd.Ctrl().Context().Done():

			if err := wd.Ctrl().Context().Err(); err != context.Canceled && err != context.DeadlineExceeded {
				wd.Logger.Sugar().Warnf(`%s-(%s),%s:"err : %+v"`,
					wd.CmptInfo(), "WebDriver", wd.Info(), err)
			}
			return nil
		case <-tickerSetupChecker.C:
			{
				// drain event
				select {
				case <-tickerSetupChecker.C:
				default:
				}

				if wd.Cnf.EnableAutoTest && wd.LDB != nil {
					wd.DBLoadSetupWrap(wd.LDB, wd.Logger)
				}

				// 先从本地加载 然后再从web数据库加载 这样web数据库可以覆盖上次的加载 以web数据库为准
				if wd.PwebDB != nil {
					wd.DBLoadSetupWrap(wd.PwebDB, wd.Logger)
				}

			}
		case <-tickerOperatorChecker.C:
			{
				// drain event
				select {
				case <-tickerOperatorChecker.C:
				default:
				}

				if wd.Cnf.EnableAutoTest && wd.LDB != nil {
					wd.DBOperationWrap(wd.LDB, wd.Logger)
				}

				if wd.PwebDB != nil {
					wd.DBOperationWrap(wd.PwebDB, wd.Logger)
				}
			}
		case _, ok := <-wd.msgChanNavState:
			{
				if !ok {
					wd.Logger.Sugar().Infof(`[%s]:"Event chan closed"`, wd.Info())
					continue
				}

				navs := wd.TrackingSt.PredicateSchedule.GetNavStatus()
				//初始化
				if wd.tmpNavState == gzb_sche.NavUnknown {
					switch navs {
					case gzb_sche.NavUp, gzb_sche.NavDown:
						wd.tmpNavState = navs
						wd.tmpSid = wd.TrackingSt.ScheduleId.GetScheduleId()
					}
					continue
				}

				//只判断通航有数据 并且通航状态切换后 就做统计的工作
				if wd.tmpNavState != navs {
					if wd.tmpSid != wd.TrackingSt.ScheduleId.GetScheduleId() {
						wd.StatisWrap(wd.tmpSid)
					}
				}

				switch navs {
				case gzb_sche.NavUp, gzb_sche.NavDown:
					{
						wd.tmpNavState = navs
						wd.tmpSid = wd.TrackingSt.ScheduleId.GetScheduleId()
					}
				}

			}
		case _, ok := <-wd.msgChanNavlState:
			{
				if !ok {
					wd.Logger.Sugar().Infof(`[%s]:"Event chan closed"`, wd.Info())
					continue
				}

				wd.Logger.Sugar().Debugf(`[%s]:"receive navl state change",`, wd.Info())
				if wd.Cnf.EnableAutoTest && wd.LDB != nil {
					wd.DBOperationWrap(wd.LDB, wd.Logger)
				}

				//操作web数据库 动画等
				if wd.PwebDB != nil {
					wd.DBOperationWrap(wd.PwebDB, wd.Logger)
				}
			}
		}
	}

}

func (wd *Web) StatisWrap(sdids string) {
	//sdids := wd.TrackingSt.ScheduleId.GetScheduleId()
	{ //速度数据的统计处理
		if wd.PwebDB != nil {
			err := wd.TrackingSt.Ref.SpeedStatisWrap.LoadSpeedDataFromDB(sdids, wd.PwebDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("SpeedStatisWrap.LoadSpeedDataFromDB ScheduleId,err : %+v", sdids, err)
			}
			err = wd.TrackingSt.Ref.SpeedStatisWrap.SpeedStatisCompute(sdids, wd.PwebDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("SpeedStatisWrap.SpeedStatisCompute ScheduleId,err : %+v", sdids, err)
			}
		}

		if wd.Cnf.EnableAutoTest && wd.LDB != nil {
			err := wd.TrackingSt.Ref.SpeedStatisWrap.LoadSpeedDataFromDB(sdids, wd.LDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("SpeedStatisWrap.LoadSpeedDataFromDB ScheduleId,err : %+v", sdids, err)
			}
			err = wd.TrackingSt.Ref.SpeedStatisWrap.SpeedStatisCompute(sdids, wd.LDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("SpeedStatisWrap.SpeedStatisCompute ScheduleId,err : %+v", sdids, err)
			}
		}
	}

	{ //禁停线 数据的统计处理
		if wd.PwebDB != nil {
			err := wd.TrackingSt.Ref.StopLineStatisWrap.LoadStopLineDataFromDB(sdids, wd.PwebDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("StopLineStatisWrap.LoadStopLineDataFromDB ScheduleId,err : %+v", sdids, err)
			}
			err = wd.TrackingSt.Ref.StopLineStatisWrap.StatisCompute(sdids, wd.PwebDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("StopLineStatisWrap.LoadStopLineDataFromDB ScheduleId,err : %+v", sdids, err)
			}
		}

		if wd.Cnf.EnableAutoTest && wd.LDB != nil {
			err := wd.TrackingSt.Ref.StopLineStatisWrap.LoadStopLineDataFromDB(sdids, wd.LDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("StopLineStatisWrap.LoadStopLineDataFromDB ScheduleId,err : %+v", sdids, err)
			}
			err = wd.TrackingSt.Ref.StopLineStatisWrap.StatisCompute(sdids, wd.LDB, wd.Logger)
			if err != nil {
				wd.Logger.Sugar().Errorf("StopLineStatisWrap.LoadStopLineDataFromDB ScheduleId,err : %+v", sdids, err)
			}
		}
	}
}

func (wd *Web) DBOperationWrap(db *gorm.DB, l *zap.Logger) (err error) {

	if wd.TrackingSt.Ref.AnimationWrap == nil {
		return
	}
	// 动画数据的更新和写入到数据库
	if err = wd.TrackingSt.Ref.AnimationWrap.UpdateToDB(db, l); err != nil {
		l.Sugar().Errorf(`%s-(%s),%s:"Animation UpdateToDB err : %+v"`,
			wd.CmptInfo(), "WebDriver", wd.Info(), err)
	}
	l.Sugar().Debugf(`%s-(%s),%s:"Animation UpdateToDB: %s"`,
		wd.CmptInfo(), "WebDriver", wd.Info(), wd.TrackingSt.Ref.AnimationWrap.Infos())

	return

}

func (wd *Web) DBLoadSetupWrap(db *gorm.DB, l *zap.Logger) (err error) {
	navs := wd.TrackingSt.PredicateSchedule.GetNavStatus()
	navls := wd.TrackingSt.PredicateSchedule.GetNavlockStatus()
	//只有通航上下行 进闸中 出闸中 才去获取
	if navs == gzb_sche.NavUp || navs == gzb_sche.NavDown {
		switch navls {
		case gzb_sche.NavlockUpGoingIn, gzb_sche.NavlockUpGoingOut, gzb_sche.NavlockDownGoingIn, gzb_sche.NavlockDownGoingOut:
			{
				//加载ship speed setup
				if err = wd.TrackingSt.Ref.SpeedSetupWrap.LoadFromDB(db, l); err != nil {
					l.Sugar().Errorf(`%s-(%s),%s:"Ship Speed  setup  load form db  err : %+v"`,
						wd.CmptInfo(), "WebDriver", wd.Info(), err)
				}
				l.Sugar().Debugf(`%s-(%s),%s:"Ship Speed  setup LoadFromDB: %+v"`,
					wd.CmptInfo(), "WebDriver", wd.Info(), wd.TrackingSt.Ref.SpeedSetupWrap)
			}
		}
	}

	return
}
