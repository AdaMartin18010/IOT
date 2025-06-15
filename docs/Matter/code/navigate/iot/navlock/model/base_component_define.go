package model

import (
	"context"
	"errors"
	"fmt"
	"runtime"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
)

var (
	//检查组件实现接口的约束
	_ cm.WorkerRecover = (*BaseCptCtx)(nil)
	// _ cmpt.Cpt          = (*BaseCptCtx)(nil)
	// _ cmpt.CptsOperator = (*BaseCptCtx)(nil)
	_ cmpt.CptComposite = (*BaseCptCtx)(nil)
	_ cmpt.CptRoot      = (*BaseCptCtx)(nil)

	_ cm.WorkerRecover = (*NavlockModel)(nil)
	// _ cmpt.Cpt          = (*NavlockModel)(nil)
	// _ cmpt.CptsOperator = (*NavlockModel)(nil)
	_ cmpt.CptComposite = (*NavlockModel)(nil)
	_ cmpt.CptRoot      = (*NavlockModel)(nil)
)

const (
	NavlockModelKind = cmpt.KindName("Navlock")
)

// NavlockModel 对应到不同的船闸
// 实现基本的测速单元控制逻辑
type NavlockModel struct {
	*BaseCptCtx
}

func NewNavlockModel(mcs *CptSt, mss *ModelSysSt) *NavlockModel {
	tmp := &NavlockModel{
		BaseCptCtx: &BaseCptCtx{
			CptSt:      mcs,
			ModelSysSt: mss,
		},
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (nl *NavlockModel) Stop() (err error) {
	if !nl.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", nl.CmptInfo())
	}

	nl.Ctrl().Cancel()
	nl.Logger.Sugar().Debugln("--- Stopping ---")
	<-nl.Ctrl().Context().Done()

	if nl.Cpts != nil && nl.Cpts.Len() > 0 {
		err = nl.Cpts.Stop()
		if err != nil {
			nl.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}
	//nl.Ctrl().WaitGroup().WaitAsync()
	// wait for all components to finish working.
	if nl.EventChans != nil {
		//释放掉eventchans
		nl.EventChans.Close()
		nl.EventChans.WaitAsync()
	}
	nl.State.Store(false)
	nl.Logger.Sugar().Debugln("--- Stopped ---")
	return err
}

type BaseCptCtx struct {
	*CptSt
	*ModelSysSt
}

func NewBaseCptCtx(mcs *CptSt, mss *ModelSysSt) *BaseCptCtx {
	tmp := &BaseCptCtx{
		CptSt:      mcs,
		ModelSysSt: mss,
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (dr *BaseCptCtx) Start() (err error) {
	if dr.CptSt == nil || dr.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", dr.CmptInfo())
	} else {
		if err = dr.CptSt.Validate(); err != nil {
			return err
		}
	}

	if dr.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", dr.CmptInfo())
	}

	dr.Logger.Sugar().Debugln("--- Starting ---")
	// todo: invoke this method of each Component .RecoverWorker shall be assigned
	dr.Ctrl().WaitGroup().StartingWait(dr.WorkerRecover)
	if dr.Cpts != nil && dr.Cpts.Len() > 0 {
		err = dr.Cpts.Start()
		if err != nil {
			dr.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}
	dr.State.Store(true)
	dr.Logger.Sugar().Debugln("--- Started ---", dr.CmptInfo())
	return err
}

// todo: implement this method of each Component driver
func (dr *BaseCptCtx) Work() (err error) {
	<-dr.Ctrl().Context().Done()
	runtime.Gosched()
	if err = dr.Ctrl().Context().Err(); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		if errors.Is(err, context.DeadlineExceeded) {
			dr.Logger.Sugar().Errorf("Work timeout error : %+v", err)
			return nil
		}
		return err
	}
	return nil
}

func (dr *BaseCptCtx) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		dr.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (dr *BaseCptCtx) Stop() (err error) {
	if !dr.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", dr.CmptInfo())
	}

	dr.Ctrl().Cancel()
	dr.Logger.Sugar().Debugln("--- Stopping ---")
	<-dr.Ctrl().Context().Done()

	if dr.Cpts != nil && dr.Cpts.Len() > 0 {
		err = dr.Cpts.Stop()
		if err != nil {
			dr.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	dr.Ctrl().WaitGroup().WaitAsync()
	dr.State.Store(false)
	dr.Logger.Sugar().Debugln("--- Stopped ---")
	return err
}

func (dr *BaseCptCtx) Finalize() error {
	<-dr.Ctrl().Context().Done()
	dr.Ctrl().WaitGroup().WaitAsync()
	return nil
}
