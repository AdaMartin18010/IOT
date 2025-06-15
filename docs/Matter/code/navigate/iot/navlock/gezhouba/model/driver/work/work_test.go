package workdriver_test

import (
	"context"
	"runtime"
	"testing"
	"time"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	g "navigate/global"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

const (
	WorkDriverTest = cmpt.IdName("HttpClientDriverTest")
)

type httpWorkers struct {
	*cm.CtrlSt
	wk0 *drs_wd.Work
	wk1 *drs_wd.Work
}

func NewWorkDriverTest() *httpWorkers {
	g.MainCtr = cm.NewCtrlSt(context.Background())
	ServerMaster := navl_md.NewServerRoot(g.MainCtr)
	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 0, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "k-test", "i-test", "#1")
	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	mcs0 := navl_md.NewCptSt(drs_wd.WorkDriverKind, WorkDriverTest, 0, NavlockModel_01.Ctrl().ForkCtxWg())
	mss0 := navl_md.NewModelSysSt(mcs0.CmptInfo(), "navlock#1", "navlock", "httpdrTest", "#0")
	dr0 := navl_md.NewBaseCptCtx(mcs0, mss0)
	trks0 := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	httpdr0 := drs_wd.NewWorkDvr(dr0, trks0)

	mcs1 := navl_md.NewCptSt(drs_wd.WorkDriverKind, WorkDriverTest, 0, NavlockModel_01.Ctrl().ForkCtxWg())
	mss1 := navl_md.NewModelSysSt(mcs1.CmptInfo(), "navlock#1", "navlock", "httpdrTest", "#1")
	dr1 := navl_md.NewBaseCptCtx(mcs1, mss1)
	trks1 := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	httpdr1 := drs_wd.NewWorkDvr(dr1, trks1)

	hw := &httpWorkers{wk0: httpdr0, wk1: httpdr1}
	hw.CtrlSt = NavlockModel_01.Ctrl().ForkCtxWg()
	hw.wk0.Fn = hw.Work0
	hw.wk1.Fn = hw.Work1

	return hw
}

func (hwks *httpWorkers) Work0(ctx context.Context) {
	hwks.wk0.Logger.Sugar().Infof(`"Work0"`)
}

func (hwks *httpWorkers) Work1(ctx context.Context) {
	hwks.wk1.Logger.Sugar().Infof(`"Work1"`)
}

func (hwks *httpWorkers) Start() error {
	if err := hwks.wk0.Start(); err != nil {
		return err
	}

	if err := hwks.wk1.Start(); err != nil {
		return err
	}

	return nil
}

func (hwks *httpWorkers) Stop() error {
	if err := hwks.wk0.Stop(); err != nil {
		return err
	}

	if err := hwks.wk1.Stop(); err != nil {
		return err
	}

	return nil
}

func (hwks *httpWorkers) WorkerTestStart() error {
	hwks.wk0.Ctrl().WithCtrl(hwks.ForkCtxWg())
	hwks.wk1.Ctrl().WithCtrl(hwks.ForkCtxWg())
	if err := hwks.Start(); err != nil {
		return err
	}
	return nil
}

func (hwks *httpWorkers) WorkerTestStop() error {
	hwks.wk0.Logger.Sugar().Infof(`"---------Stop workers------------"`)
	if err := hwks.Stop(); err != nil {
		return err
	}
	return nil
}

func TestHttpWorker(t *testing.T) {
	httpdr := NewWorkDriverTest()
	httpdr.Start()
	for i := 0; i < 1; i++ {
		//httpdr.WorkerTestStart()
		runtime.Gosched()
		httpdr.WorkerTestStop()
	}
	httpdr.Stop()
}
