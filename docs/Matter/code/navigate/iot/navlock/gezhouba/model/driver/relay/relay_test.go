package relay_test

import (
	"testing"
	"time"

	cmpt "navigate/common/model/component"
	iot_cf "navigate/config/iot/drivers"
	g "navigate/global"
	navl_rly "navigate/iot/navlock/gezhouba/model/driver/relay"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

func TestRelayDriver(t *testing.T) {

	ServerMaster := navl_md.NewServerRoot(g.MainCtr)
	mcs01 := navl_md.NewCptSt(navl_md.NavlockModelKind, cmpt.IdName("navlock#1"), 0, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := navl_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := navl_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())
	mcs := navl_md.NewCptSt(cmpt.KindName("RelayDriver"), cmpt.IdName("Relay"), 0, NavlockModel_01.Ctrl().ForkCtxWg())
	mss := navl_md.NewModelSysSt(mcs.CmptInfo(), "#1", "navlock", "RelayDriver", "Relay")
	dr := navl_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	relay := navl_rly.NewRelayDvr(dr, trks)
	relay.Cnf = &iot_cf.RelayConfig{
		Enable:                     true,
		EnableAutoTest:             true,
		Ipaddress:                  "127.0.0.1",
		Port:                       1300,
		ConnectReadTimeout:         2000,
		ReConnectRetryTimeInterval: 1,
		SlaveId:                    1,
	}
	NavlockModel_01.Cpts.AddCpts(relay)
	ServerMaster.Start()
	ServerMaster.Ctrl().WaitGroup().StartAsync()

	cmd := navl_rly.NewRelayCmd()
	cmdok := relay.Commander(cmd)
	t.Logf("Relay command sended : %+v", cmdok)
	if cmdok {
		ok := cmd.WaitTimeout(1 * time.Second)
		t.Logf("Relay command: %+v, err: %+v", ok, cmd.Err())
	}
	time.Sleep(1 * time.Second)

	ServerMaster.Stop()
	ServerMaster.Finalize()
}
