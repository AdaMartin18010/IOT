package once

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

var (
	//Verify Satisfies interfaces
	_ cm.WorkerRecover  = (*Once)(nil)
	_ cmpt.Cpt          = (*Once)(nil)
	_ cmpt.CptsOperator = (*Once)(nil)
	_ cmpt.CptComposite = (*Once)(nil)
	_ cmpt.CptRoot      = (*Once)(nil)
)

const (
	OnceKind = cmpt.KindName("Once")
	//避免阻塞
	chanlen = 1
)

type Result struct {
	Val any
	Err error
}

type OnceFn func(ctx context.Context) *Result

type Once struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	EnableDebug bool

	rchan      chan *Result
	needClosed bool
	mu         *sync.Mutex

	Fn OnceFn
}

func NewOnceDvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Once {

	tmp := &Once{
		BaseCptCtx: dr,
		TrackingSt: trs,
		mu:         &sync.Mutex{},
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (od *Once) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		od.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (od *Once) Start() (err error) {
	if od.CptSt == nil || od.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", od.CmptInfo())
	} else {
		if err = od.CptSt.Validate(); err != nil {
			return err
		}
	}

	if od.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", od.CmptInfo())
	}

	if od.EnableDebug {
		od.Logger.Sugar().Debugln("--- Starting ---")
		od.Logger.Sugar().Debugf("Ctrl_info: %s ", od.Ctrl().DebugInfo())
		defer od.Logger.Sugar().Debugln("--- Started ---", od.CmptInfo())
	}

	od.mu.Lock()
	if od.needClosed {
		close(od.rchan)
		od.rchan = nil
		od.needClosed = false
	} else {
		od.rchan = make(chan *Result, chanlen)
		od.needClosed = true
	}
	od.mu.Unlock()

	od.Ctrl().WaitGroup().StartingWait(od.WorkerRecover)

	if od.Cpts != nil {
		err = od.Cpts.Start()
		if err != nil {
			od.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}
	od.State.Store(true)

	return err
}

func (od *Once) Stop() (err error) {
	if !od.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", od.CmptInfo())
	}

	od.Ctrl().Cancel()

	if od.EnableDebug {
		od.Logger.Sugar().Debugln("--- Stopping ---")
		od.Logger.Sugar().Debugf("Ctrl_info: %s ", od.Ctrl().DebugInfo())
		defer od.Logger.Sugar().Debugln("--- Stopped ---")
	}
	<-od.Ctrl().Context().Done()
	if od.Cpts != nil {
		err = od.Cpts.Stop()
		if err != nil {
			od.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}
	od.mu.Lock()
	if od.needClosed {
		close(od.rchan)
		od.needClosed = false
	}
	//让外部永久阻塞
	od.rchan = nil
	od.mu.Unlock()

	od.State.Store(false)
	return err
}

func (od *Once) Chan() <-chan *Result {
	defer od.mu.Unlock()
	od.mu.Lock()
	return od.rchan
}

func (od *Once) Work() (err error) {
	//
	defer func() {
		od.mu.Lock()
		if od.needClosed {
			close(od.rchan)
			od.needClosed = false
		}
		od.mu.Unlock()
	}()

	if od.EnableDebug {
		od.Logger.Sugar().Debugln("--- Worker Running.")
		defer od.Logger.Sugar().Debugln("---Worker Exited.")
	}

	for {
		runtime.Gosched()
		select {
		case <-od.Ctrl().Context().Done():
			{
				od.mu.Lock()
				od.rchan <- &Result{Val: nil, Err: od.Ctrl().Context().Err()}
				od.mu.Unlock()
				// if od.EnableDebug {
				// 	od.Logger.Sugar().Debugln("---Worker Exited.")
				// }
				return nil
			}
		default:
			{
				od.mu.Lock()
				od.rchan <- od.Fn(od.Ctrl().Context())
				od.mu.Unlock()
				// if od.EnableDebug {
				// 	od.Logger.Sugar().Debugln("---Worker Exited.")
				// }
				return nil
			}
		}
	}
}
