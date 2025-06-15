package navlstatus_test

import (
	"testing"
	"time"

	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	g "navigate/global"
	gzb_navls "navigate/iot/navlock/gezhouba/model/server/navlstatus"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	gzb_md "navigate/iot/navlock/model"
)

func TestData(t *testing.T) {
	navlDyndata0 := &gzb_sche.NavlockDynamicStatusData{NavState: 3}
	navlDyndata1 := *navlDyndata0
	navlDyndata1.NavState = 1
	t.Logf("navlDyndata0:%s,navlDyndata1:%s", navlDyndata0.String(), navlDyndata1.String())
}

func TestMonitor(t *testing.T) {
	ServerMaster := gzb_md.NewServerRoot(g.MainCtr)
	mcs01 := gzb_md.NewCptSt(gzb_md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := gzb_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := gzb_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())
	mcs := NavlockModel_01.Fork(cmpt.KindName("HttpDriver"), cmpt.IdName("monitor"))
	mss := gzb_md.NewModelSysSt(mcs.CmptInfo(), "navlock#1", "navlock", "#1", "mnt")
	dr := gzb_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 10*time.Second, true)
	mnt := gzb_navls.NewNavlStatusSvr(dr, trks)

	mnt.Cnf = &gzb_cf.HttpClientConfig{
		Enable: false,
		//Url: "http://httpbin.org/post",
		Url:                "http://172.60.222.115:8881/SmartShip/m/jtx/data",
		ConnectReadTimeout: 1000,
		DoTimeInterval:     100,
	}

	NavlockModel_01.Cpts.AddCpts(mnt)
	ServerMaster.Start()
	time.Sleep(3 * time.Second)
	ServerMaster.Stop()
	mnt.Start()

	mnt.Stop()
	mnt.Start()
	mnt.Stop()
	ServerMaster.Finalize()
}
