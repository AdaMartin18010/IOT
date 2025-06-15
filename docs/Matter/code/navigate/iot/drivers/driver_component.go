package drivers

import (
	"fmt"
	cm "navigate/common/model"
	cmpt "navigate/common/model/component"

	//evchs "navigate/common/model/eventchans"

	"go.uber.org/zap"
)

var (
	//Verify Satisfies interfaces
	_ cmpt.CptRoot     = (*Driver)(nil)
	_ cmpt.Cpt         = (*Driver)(nil)
	_ cm.WorkerRecover = (*Driver)(nil)
)

const (
	DriverComponentKind = cmpt.KindName("DriverComponent")
)

type Driver struct {
	*cmpt.CptMetaSt
	//evchs.EventChans
	cmpt.Cmder

	Logger *zap.Logger //日志分支
}

func NewDriver(kn cmpt.KindName, in cmpt.IdName, log *zap.Logger) *Driver {
	return &Driver{
		CptMetaSt: cmpt.NewCptMetaSt(kn, in),
		Cmder:     nil,
		Logger:    log,
	}
}

func (dc *Driver) Start() error {
	if dc.State.Load().(bool) {
		return fmt.Errorf("component:Kind-%s Id-%s is already Started", dc.KindStr, dc.Id())
	}
	dc.Logger.Sugar().Debugf("component:Kind-%s Id-%s is Starting", dc.KindStr, dc.Id())
	dc.Ctrl().WaitGroup().StartingWait(dc)
	//dc.Ctrl().WaitGroup().StartAsync()
	dc.State.Store(true)
	dc.Logger.Sugar().Debugf("component:Kind-%s Id-%s is Started", dc.KindStr, dc.Id())
	return nil
}

func (dc *Driver) Stop() error {
	if !dc.State.Load().(bool) {
		return fmt.Errorf("component:Kind-%s Id-%s is already Stopped", dc.KindStr, dc.Id())
	}
	dc.Logger.Sugar().Debugf("component:Kind-%s Id-%s is Stopping", dc.KindStr, dc.Id())
	dc.Ctrl().Cancel()
	dc.Logger.Sugar().Debugf("component:Kind-%s Id-%s is Stopped", dc.KindStr, dc.Id())
	dc.State.Store(false)
	return nil
}
