package dbtest_test

import (
	"testing"

	mdl "navigate/common/model"
	g "navigate/global"
	dbt "navigate/store/iot/navlock/dbtest"
	gzb_db_web_wrapper "navigate/store/iot/navlock/gezhouba/web/wrapper"
)

func TestDB(t *testing.T) {
	if g.ProduceDB != nil {
		t.Logf("ProduceDB: %+v", g.ProduceDB)
	}

	if g.LocalDB != nil {
		t.Logf("LocalDB: %+v", g.LocalDB)
	}

	if g.ProduceWebDB != nil {
		t.Logf("ProduceWebDB: %+v", g.ProduceWebDB)
	}
}

func TestLocalDB(t *testing.T) {

}

func TestMysqlWebDB(t *testing.T) {
	//TestDB(t)
	TestAnimationWrap(t)
}

func TestAnimationWrap(t *testing.T) {
	amr := gzb_db_web_wrapper.NewNavAnimationWrap(3)
	t.Logf("AnimationRecord IsNull   : %+v", amr.IsNull())

	dbt.AnimationRecordSave(amr)

	t.Logf("AnimationRecord: %+v", amr.Infos())
	err := amr.UpdateToDB(g.LocalDB, mdl.L)
	if err != nil {
		t.Logf("UpdateToDB err: %+v", err)
		t.Fail()
	}
	t.Logf("AnimationRecord: %+v", amr.Infos())

	dbt.AnimationRecordSave0(amr)
	t.Logf("AnimationRecord: %+v", amr.Infos())
	err = amr.UpdateToDB(g.LocalDB, mdl.L)
	if err != nil {
		t.Logf("UpdateToDB err: %+v", err)
		t.Fail()
	}
	t.Logf("AnimationRecord: %+v", amr.Infos())
}

func TestLedSetupWrap(t *testing.T) {
	lsw := gzb_db_web_wrapper.NewNavPlayScreenSetupWrap(2)
	_, err := lsw.LoadFromDB(g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("LedSetup Loadform db err: %+v", err)
		t.Fail()
	}

	t.Logf("LedSetup: %+v", lsw.Infos())
	t.Logf("LedSetup: %+v", lsw.GetSetup())
}

func TestShipSpeedSetupWrap(t *testing.T) {
	ssw := gzb_db_web_wrapper.NewNavShipSpeedSetupWrap(1)
	err := ssw.LoadFromDB(g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("shipsetup Loadform db err: %+v", err)
		t.Fail()
	}

	t.Logf("shipsetup: %+v", ssw.Infos())
	t.Logf("shipsetup Info: %+v", ssw)
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(0, 2.41))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(0, 2.39))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(14, 2.41))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(14, 2.39))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(50, 2.41))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(50, 2.39))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(51, 1.71))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(51, 1.7))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(75, 1.71))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(75, 1.7))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(100, 1.71))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(100, 1.7))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(100.2, 2.1))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(100.2, 2))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(130.9, 2.1))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(130.9, 2))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(150, 2.1))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(150, 2))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(150.9, 2.41))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(150.9, 2.4))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(200, 2.41))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(200, 2.4))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(290, 2.1))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(290, 2))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(350, 2.1))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(350, 2))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(351, 20))
	t.Logf("check speed : %+v", ssw.CheckOverSpeed(1000, 2))
}

func TestStoplineStatisWrap(t *testing.T) {
	nslsw := gzb_db_web_wrapper.NewNavStopLineStatisWrap(2)
	err := nslsw.LoadStopLineDataFromDB("#02#2022-08-03-10-36-30", g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("LoadStopLineDataFromDB err: %+v", err)
		t.Fail()
	}

	err = nslsw.StatisCompute("#02#2022-08-03-10-36-30", g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("StatisCompute err: %+v", err)
	}

}

func TestSpeedStatisWrap(t *testing.T) {
	ssw := gzb_db_web_wrapper.NewNavSpeedStatisWrap(1)
	err := ssw.LoadSpeedDataFromDB("2020-9-13-100", g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("LoadSpeedDataFromDB err: %+v", err)
		t.Fail()
	}

	err = ssw.SpeedStatisCompute("2020-9-13-100", g.ProduceWebDB, mdl.L)
	if err != nil {
		t.Logf("StatisCompute err: %+v", err)
	}
}

func TestMysqlDB(t *testing.T) {

}

func CaseSpeedData(t *testing.T) {

}

func CaseStoplineData(t *testing.T) {

}

func CaseComponentsData(t *testing.T) {

}

func CaseAnimationData(t *testing.T) {

}
