package platform_test

import (
	"testing"
	"time"

	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	g "navigate/global"
	gzb_plfm "navigate/iot/navlock/gezhouba/model/server/platform"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

func TestStoplineS(t *testing.T) {
	ServerMaster := navl_md.NewServerRoot(g.MainCtr)
	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())
	mcs := NavlockModel_01.Fork(cmpt.KindName("HttpDriver"), cmpt.IdName("platform"))
	mss := navl_md.NewModelSysSt(mcs.CmptInfo(), "#1", "navlock", "httpdr", "platform")
	dr := navl_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 10*time.Second, true)
	platform := gzb_plfm.NewPlatformSvr(dr, trks)

	platform.Cnf = &gzb_cf.PlatformConfig{
		Enable:             true,
		Url:                "http://httpbin.org/get",
		Tag:                "Up",
		ConnectReadTimeout: 3000,
		DoTimeInterval:     100,
	}

	NavlockModel_01.Cpts.AddCpts(platform)
	ServerMaster.Start()
	time.Sleep(2 * time.Second)
	ServerMaster.Stop()
	platform.Start()
	platform.Stop()
	ServerMaster.Finalize()
}
