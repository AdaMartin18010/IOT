package workdriver

import (
	"context"
	"fmt"
	"runtime"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

var (
	//Verify Satisfies interfaces
	_ cm.WorkerRecover  = (*Work)(nil)
	_ cmpt.Cpt          = (*Work)(nil)
	_ cmpt.CptsOperator = (*Work)(nil)
	_ cmpt.CptComposite = (*Work)(nil)
	_ cmpt.CptRoot      = (*Work)(nil)
)

const (
	WorkDriverKind = cmpt.KindName("Work")
)

type WorkFn func(ctx context.Context)

type Work struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt
	Fn WorkFn

	EnableDebug bool
}

func NewWorkDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Work {
	tmp := &Work{
		BaseCptCtx: dr,
		TrackingSt: trs,
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (wd *Work) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		wd.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (wd *Work) Start() (err error) {
	if wd.CptSt == nil || wd.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", wd.CmptInfo())
	} else {
		if err = wd.CptSt.Validate(); err != nil {
			return err
		}
	}

	if wd.State.Load().(bool) {
		return fmt.Errorf("%s is already Started,Ctrl_info: %s", wd.CmptInfo(), wd.Ctrl().DebugInfo())
	}

	if wd.EnableDebug {
		wd.Logger.Sugar().Debugln("--- Starting ---")
		defer wd.Logger.Sugar().Debugf("--- Started --- : Ctrl_info: %s", wd.Ctrl().DebugInfo())
	}

	wd.Ctrl().WaitGroup().StartingWait(wd.WorkerRecover)
	if wd.Cpts != nil && wd.Cpts.Len() > 0 {
		err = wd.Cpts.Start()
		if err != nil {
			wd.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}
	wd.State.Store(true)

	return
}

func (wd *Work) Stop() (err error) {
	if !wd.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", wd.CmptInfo())
	}

	wd.Ctrl().Cancel()
	if wd.EnableDebug {
		wd.Logger.Sugar().Debugln("--- Stopping ---")
		wd.Logger.Sugar().Debugf("Ctrl_info: %s ", wd.Ctrl().DebugInfo())
		defer wd.Logger.Sugar().Debugln("--- Stopped ---")
	}

	<-wd.Ctrl().Context().Done()
	if wd.Cpts != nil && wd.Cpts.Len() > 0 {
		err = wd.Cpts.Stop()
		if err != nil {
			wd.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}
	wd.State.Store(false)

	return err
}

func (wd *Work) Work() (err error) {
	if wd.EnableDebug {
		wd.Logger.Sugar().Debugln("--- WorkDriver Running.")
		defer wd.Logger.Sugar().Debugln("---WorkDriver Exited.")
	}

	runtime.Gosched()
	wd.Fn(wd.Ctrl().Context())
	<-wd.Ctrl().Context().Done()
	return
}
