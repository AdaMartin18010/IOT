package shipspeed_test

import (
	"context"
	"testing"
	"time"

	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	g "navigate/global"
	gzb_shipspeed "navigate/iot/navlock/gezhouba/model/driver/shipspeed"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

func TestSpeedMeasureUnitsAdd(t *testing.T) {
	g.MainCtr = cm.NewCtrlSt(context.Background())
	ServerMaster := navl_md.NewServerRoot(g.MainCtr)

	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())

	mcs := NavlockModel_01.Fork(gzb_shipspeed.Kind, gzb_shipspeed.ShipSpeedId_Up)
	mss := navl_md.NewModelSysSt(mcs.CmptInfo(), "#1", "navlock", "MeasureUnits", "MeasureUnits_up")
	dr := navl_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 10*time.Second, true)
	mus := gzb_shipspeed.NewShipSpeedDvr(dr, trks)

	NavlockModel_01.Cpts.AddCpts(mus)
	t.Logf("NavlockModel_01.Components len: %d", NavlockModel_01.Cpts.Len())

	ServerMaster.Start()
	ServerMaster.Ctrl().WaitGroup().StartAsync()
	ServerMaster.Stop()
	ServerMaster.Finalize()
}
