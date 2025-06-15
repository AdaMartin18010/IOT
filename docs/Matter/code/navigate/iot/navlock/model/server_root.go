package model

import (
	"fmt"
	"runtime"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	evchs "navigate/common/model/eventchans"
)

const (
	ServiceRootKind = cmpt.KindName("rt")
)

var (
	L = mdl.L

	_ mdl.WorkerRecover = (*ServerRoot)(nil)
	_ cmpt.Cpt          = (*ServerRoot)(nil)
	_ cmpt.CptsOperator = (*ServerRoot)(nil)
	_ cmpt.CptComposite = (*ServerRoot)(nil)
	_ cmpt.CptRoot      = (*ServerRoot)(nil)
)

type ServerRoot struct {
	*cmpt.CptMetaSt
	cmpt.Cpts
	evchs.EventChans
	cmpt.Cmder
}

// ServerRoot abstrect system model
func NewServerRoot(ctr *mdl.CtrlSt) *ServerRoot {
	temp := &ServerRoot{
		CptMetaSt:  cmpt.NewCptMetaSt(ServiceRootKind, ctr),
		EventChans: nil,
		Cmder:      nil,
		Cpts:       cmpt.NewCpts(),
	}
	temp.WorkerRecover = temp
	return temp
}

func (sr *ServerRoot) Ctrl() *mdl.CtrlSt {
	return sr.CptMetaSt.Ctrl()
}

func (sr *ServerRoot) Work() error {
	return nil
}

func (sr *ServerRoot) Recover() {
	if rc := recover(); rc != nil {
		var buf [4096]byte
		n := runtime.Stack(buf[:], false)
		L.Sugar().Warnf(`%s Work recover :%+v,stack trace: %s`, sr.CmptInfo(), rc, buf[:n])
	}
}

func (sr *ServerRoot) Start() error {
	if sr.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", sr.CmptInfo())
	}
	L.Sugar().Infof("%s is Starting.", sr.CmptInfo())

	sr.Ctrl().WaitGroup().StartingWait(sr)
	if sr.Cpts != nil && sr.Cpts.Len() > 0 {
		err := sr.Cpts.Start()
		if err != nil {
			L.Sugar().Errorf("%s Components is Starting,err: %+v", sr.CmptInfo(), err)
			return err
		}
	}
	sr.State.Store(true)
	L.Sugar().Infof("%s is Started.", sr.CmptInfo())
	return nil
}

func (sr *ServerRoot) Stop() error {
	if !sr.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", sr.CmptInfo())
	}
	L.Sugar().Infof("%s is Stopping", sr.CmptInfo())
	defer L.Sugar().Infof("%s is Stopped.", sr.CmptInfo())

	sr.Ctrl().Cancel()
	<-sr.Ctrl().Context().Done()

	if sr.Cpts != nil && sr.Cpts.Len() > 0 {
		err := sr.Cpts.Stop()
		if err != nil {
			L.Sugar().Errorf("%s Components is Stopping ,err: %+v", sr.CmptInfo(), err)
		}
	}
	sr.State.Store(false)
	return nil
}

func (sr *ServerRoot) Finalize() error {
	// <-sr.Ctrl().Context().Done()
	//sr.Ctrl().WaitGroup().WaitAsync()
	return nil
}
