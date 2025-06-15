package once_test

import (
	"context"
	"runtime"
	"testing"
	"time"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	g "navigate/global"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

const (
	HttpClientDriverTest = cmpt.IdName("HttpClientDriverTest")
)

type httpWorkers struct {
	*cm.CtrlSt
	wk0 *drs_once.Once
	wk1 *drs_once.Once
}

func NewhttpDriverTest() *httpWorkers {
	g.MainCtr = cm.NewCtrlSt(context.Background())
	ServerMaster := navl_md.NewServerRoot(g.MainCtr)
	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 0, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "k-test", "i-test", "#1")
	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	mcs0 := navl_md.NewCptSt(drs_once.OnceKind, HttpClientDriverTest, 0, NavlockModel_01.Ctrl().ForkCtxWg())
	mss0 := navl_md.NewModelSysSt(mcs0.CmptInfo(), "navlock#1", "navlock", "httpdrTest", "#0")
	dr0 := navl_md.NewBaseCptCtx(mcs0, mss0)
	trks0 := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	httpdr0 := drs_once.NewOnceDvr(dr0, trks0)

	mcs1 := navl_md.NewCptSt(drs_once.OnceKind, HttpClientDriverTest, 0, NavlockModel_01.Ctrl().ForkCtxWg())
	mss1 := navl_md.NewModelSysSt(mcs1.CmptInfo(), "navlock#1", "navlock", "httpdrTest", "#1")
	dr1 := navl_md.NewBaseCptCtx(mcs1, mss1)
	trks1 := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	httpdr1 := drs_once.NewOnceDvr(dr1, trks1)

	hw := &httpWorkers{wk0: httpdr0, wk1: httpdr1}
	hw.wk0.Fn = hw.Work0
	hw.wk1.Fn = hw.Work1
	hw.CtrlSt = NavlockModel_01.Ctrl().ForkCtxWg()
	return hw
}

func (hwks *httpWorkers) Work0(ctx context.Context) *drs_once.Result {
	hwks.wk0.Logger.Sugar().Infof(`"Work0"`)
	resp := "work0--value"
	return &drs_once.Result{
		Val: resp,
		Err: nil,
	}
}

func (hwks *httpWorkers) Work1(ctx context.Context) *drs_once.Result {
	hwks.wk1.Logger.Sugar().Infof(`"Work1"`)
	resp := "work1--value"
	return &drs_once.Result{
		Val: resp,
		Err: nil,
	}
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
	hwks.wk0.Logger.Sugar().Infof(`"---------fork workers------------"`)
	hwks.wk0.Ctrl().WithCtrl(hwks.ForkCtxWg())
	hwks.wk0.Logger.Sugar().Infof(`"hdrt.wk0.StartWork(hdrt.wk0)"`)

	hwks.wk1.Logger.Sugar().Infof(`"---------fork workers------------"`)
	hwks.wk1.Ctrl().WithCtrl(hwks.ForkCtxWg())
	hwks.wk1.Logger.Sugar().Infof(`"hdrt.wk1.StartWork(hdrt.wk1)"`)

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
	httpdr := NewhttpDriverTest()
	httpdr.Start()
	for i := 0; i < 1; i++ {
		httpdr.WorkerTestStart()
		runtime.Gosched()
		httpdr.WorkerTestStop()
	}
	httpdr.Stop()
}
