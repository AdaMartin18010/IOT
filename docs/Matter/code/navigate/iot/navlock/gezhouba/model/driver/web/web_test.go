package web_test

import (
	cmpt "navigate/common/model/component"
	g "navigate/global"
	gzb_web "navigate/iot/navlock/gezhouba/model/driver/web"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	md "navigate/iot/navlock/model"

	"testing"
	"time"
)

func TestStoplineS(t *testing.T) {
	ServerMaster := md.NewServerRoot(g.MainCtr)
	mcs01 := md.NewCptSt(md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())

	mcs := md.NewCptSt(gzb_web.WebKind, cmpt.IdName("web"), 1, NavlockModel_01.Ctrl().ForkCtxWg())
	mss := md.NewModelSysSt(mcs.CmptInfo(), "#1", "navlock", "webdr", "web")
	dr := md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 10*time.Second, true)
	web := gzb_web.NewWebDvr(dr, trks)

	NavlockModel_01.Cpts.AddCpts(web)
	ServerMaster.Start()
	ServerMaster.Ctrl().WaitGroup().StartAsync()
	time.Sleep(1 * time.Second)
	ServerMaster.Stop()
	ServerMaster.Finalize()
}
