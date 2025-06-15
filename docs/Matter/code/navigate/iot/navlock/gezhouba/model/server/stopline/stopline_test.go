package stopline_test

import (
	"testing"
	"time"

	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	g "navigate/global"
	gzb_stl "navigate/iot/navlock/gezhouba/model/server/stopline"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

func TestStringCompare(t *testing.T) {
	str := ""
	t.Logf("%t", str == "")
	str = "123"
	t.Logf("%t", str != "")
}

func TestStoplineS(t *testing.T) {
	ServerMaster := navl_md.NewServerRoot(g.MainCtr)
	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")

	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())

	mcs := NavlockModel_01.Fork(cmpt.KindName("HttpDriver"), cmpt.IdName("stopline"))
	mss := navl_md.NewModelSysSt(mcs.CmptInfo(), "#1", "navlock", "httpdr", "stpl")
	dr := navl_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 10*time.Second, true)
	stopline := gzb_stl.NewStoplineSvr(dr, trks)

	stopline.UpCnf = &gzb_cf.HttpClientConfig{
		Enable: true,
		Url:    "http://172.60.222.115:8881/SmartShip/m/jtx/data",
		//Url:                "http://httpbin.org/get",
		Tag:                "Up",
		ConnectReadTimeout: 1000,
		DoTimeInterval:     100,
	}

	stopline.DownCnf = &gzb_cf.HttpClientConfig{
		Enable: true,
		//Url:                "http://172.60.222.135:8881/SmartShip/m/jtx/data",
		Url:                "http://httpbin.org/get",
		Tag:                "Down",
		ConnectReadTimeout: 1000,
		DoTimeInterval:     100,
	}

	NavlockModel_01.Cpts.AddCpts(stopline)
	ServerMaster.Start()
	ServerMaster.Ctrl().WaitGroup().StartAsync()
	time.Sleep(2 * time.Second)
	ServerMaster.Stop()
	ServerMaster.Finalize()
}
