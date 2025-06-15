package schedule_test

import (
	"testing"

	cm "navigate/common/model"
	g "navigate/global"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	_ "navigate/store/iot/navlock/dbtest"
)

const (
	up_czz_0 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T12:33:57.2070097+08:00\",\"EndTime\":\"2022-11-01T12:33:57.2070097+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.7\",\"DownInnerWaterLast\":\"64.7\",\"tNavState\":\"5\",\"tUpWater\":\"64.71\",\"tUpInnerWater\":\"64.7\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"64.7\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.74\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_czz_1 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T12:43:29.5596635+08:00\",\"EndTime\":\"2022-11-01T12:43:29.5596635+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.72\",\"DownInnerWaterLast\":\"64.74\",\"tNavState\":\"5\",\"tUpWater\":\"64.73\",\"tUpInnerWater\":\"64.72\",\"tDownWater\":\"40.08\",\"tDownInnerWater\":\"64.74\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_czz_2 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T12:43:32.8049344+08:00\",\"EndTime\":\"2022-11-01T12:43:32.8049344+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.72\",\"DownInnerWaterLast\":\"64.74\",\"tNavState\":\"5\",\"tUpWater\":\"64.73\",\"tUpInnerWater\":\"64.72\",\"tDownWater\":\"40.08\",\"tDownInnerWater\":\"64.74\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"

	down_jzz_0 = "{\"NavState\":\"6\",\"NavlockState\":\"6\",\"BeginTime\":\"2022-11-01T12:43:36.2593112+08:00\",\"EndTime\":\"2022-11-01T12:43:36.2593112+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"6\",\"UpInnerWaterLast\":\"64.73\",\"DownInnerWaterLast\":\"64.74\",\"tNavState\":\"6\",\"tUpWater\":\"64.73\",\"tUpInnerWater\":\"64.73\",\"tDownWater\":\"40.09\",\"tDownInnerWater\":\"64.74\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":1},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	down_jzz_1 = "{\"NavState\":\"0\",\"NavlockState\":\"6\",\"BeginTime\":\"2022-11-01T12:43:39.7413863+08:00\",\"EndTime\":\"2022-11-01T12:43:39.7413863+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"6\",\"UpInnerWaterLast\":\"64.73\",\"DownInnerWaterLast\":\"64.74\",\"tNavState\":\"6\",\"tUpWater\":\"64.74\",\"tUpInnerWater\":\"64.73\",\"tDownWater\":\"40.09\",\"tDownInnerWater\":\"64.74\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":1},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	down_jzz_2 = "{\"NavState\":\"0\",\"NavlockState\":\"6\",\"BeginTime\":\"2022-11-01T12:43:43.0489174+08:00\",\"EndTime\":\"2022-11-01T12:43:43.0489174+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"6\",\"UpInnerWaterLast\":\"64.73\",\"DownInnerWaterLast\":\"64.74\",\"tNavState\":\"6\",\"tUpWater\":\"64.75\",\"tUpInnerWater\":\"64.73\",\"tDownWater\":\"40.09\",\"tDownInnerWater\":\"64.74\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":1},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"

	down_jzwb_0 = "{\"NavState\":\"0\",\"NavlockState\":\"7\",\"BeginTime\":\"2022-11-01T12:44:04.6399659+08:00\",\"EndTime\":\"2022-11-01T12:44:04.6399659+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"7\",\"UpInnerWaterLast\":\"64.76\",\"DownInnerWaterLast\":\"64.77\",\"tNavState\":\"6\",\"tUpWater\":\"64.76\",\"tUpInnerWater\":\"64.76\",\"tDownWater\":\"40.09\",\"tDownInnerWater\":\"64.77\",\"tUpGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"99.27\"},\"Right\":{\"State\":3,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	down_jzwb_1 = "{\"NavState\":\"0\",\"NavlockState\":\"7\",\"BeginTime\":\"2022-11-01T12:47:19.5985441+08:00\",\"EndTime\":\"2022-11-01T12:47:19.5985441+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"7\",\"UpInnerWaterLast\":\"64.95\",\"DownInnerWaterLast\":\"64.96\",\"tNavState\":\"6\",\"tUpWater\":\"64.88\",\"tUpInnerWater\":\"64.95\",\"tDownWater\":\"40.08\",\"tDownInnerWater\":\"64.96\",\"tUpGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"0.46\"},\"Right\":{\"State\":3,\"OpenAngle\":\"0.54\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	down_jzwb_2 = "{\"NavState\":\"0\",\"NavlockState\":\"7\",\"BeginTime\":\"2022-11-01T12:47:19.5985441+08:00\",\"EndTime\":\"2022-11-01T12:47:19.5985441+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"7\",\"UpInnerWaterLast\":\"64.95\",\"DownInnerWaterLast\":\"64.96\",\"tNavState\":\"6\",\"tUpWater\":\"64.88\",\"tUpInnerWater\":\"64.95\",\"tDownWater\":\"40.08\",\"tDownInnerWater\":\"64.96\",\"tUpGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"0.46\"},\"Right\":{\"State\":3,\"OpenAngle\":\"0.54\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.13\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"

	down_czz_0 = "{\"NavState\":\"0\",\"NavlockState\":\"9\",\"BeginTime\":\"2022-11-01T13:03:26.8076887+08:00\",\"EndTime\":\"2022-11-01T13:03:26.8076887+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"9\",\"UpInnerWaterLast\":\"40.05\",\"DownInnerWaterLast\":\"40.05\",\"tNavState\":\"6\",\"tUpWater\":\"64.73\",\"tUpInnerWater\":\"40.05\",\"tDownWater\":\"40.08\",\"tDownInnerWater\":\"40.05\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":1}}"
	down_czz_1 = "{\"NavState\":\"0\",\"NavlockState\":\"9\",\"BeginTime\":\"2022-11-01T13:03:59.5980637+08:00\",\"EndTime\":\"2022-11-01T13:04:00.0876789+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"9\",\"UpInnerWaterLast\":\"39.96\",\"DownInnerWaterLast\":\"40.01\",\"tNavState\":\"6\",\"tUpWater\":\"64.74\",\"tUpInnerWater\":\"39.96\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"40.01\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":1}}"
	down_czz_2 = "{\"NavState\":\"0\",\"NavlockState\":\"9\",\"BeginTime\":\"2022-11-01T13:04:11.0974858+08:00\",\"EndTime\":\"2022-11-01T13:04:11.0974858+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"9\",\"UpInnerWaterLast\":\"39.97\",\"DownInnerWaterLast\":\"40.01\",\"tNavState\":\"6\",\"tUpWater\":\"64.74\",\"tUpInnerWater\":\"39.97\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"40.01\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":1}}"

	up_jzz_0 = "{\"NavState\":\"5\",\"NavlockState\":\"1\",\"BeginTime\":\"2022-11-01T13:04:14.2509682+08:00\",\"EndTime\":\"2022-11-01T13:04:14.2509682+08:00\",\"NavStateLast\":\"6\",\"NavlockStateLast\":\"1\",\"UpInnerWaterLast\":\"39.97\",\"DownInnerWaterLast\":\"40.01\",\"tNavState\":\"6\",\"tUpWater\":\"64.75\",\"tUpInnerWater\":\"39.97\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"40.01\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":1,\"Right\":2}}"
	up_jzz_1 = "{\"NavState\":\"0\",\"NavlockState\":\"1\",\"BeginTime\":\"2022-11-01T13:27:35.4034284+08:00\",\"EndTime\":\"2022-11-01T13:27:35.4034485+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"1\",\"UpInnerWaterLast\":\"40.06\",\"DownInnerWaterLast\":\"40.04\",\"tNavState\":\"6\",\"tUpWater\":\"64.91\",\"tUpInnerWater\":\"40.06\",\"tDownWater\":\"40.06\",\"tDownInnerWater\":\"40.04\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":1,\"Right\":2}}"
	up_jzz_2 = "{\"NavState\":\"0\",\"NavlockState\":\"1\",\"BeginTime\":\"2022-11-01T13:28:22.7148377+08:00\",\"EndTime\":\"2022-11-01T13:28:22.7148377+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"1\",\"UpInnerWaterLast\":\"40.01\",\"DownInnerWaterLast\":\"40.01\",\"tNavState\":\"5\",\"tUpWater\":\"64.95\",\"tUpInnerWater\":\"40.01\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"40.01\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.06\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":1,\"Right\":2}}"

	up_jzwb_0 = "{\"NavState\":\"0\",\"NavlockState\":\"2\",\"BeginTime\":\"2022-11-01T13:28:41.1692522+08:00\",\"EndTime\":\"2022-11-01T13:28:41.1692522+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"2\",\"UpInnerWaterLast\":\"40\",\"DownInnerWaterLast\":\"40\",\"tNavState\":\"5\",\"tUpWater\":\"64.95\",\"tUpInnerWater\":\"40\",\"tDownWater\":\"40.03\",\"tDownInnerWater\":\"40\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"100.46\"},\"Right\":{\"State\":3,\"OpenAngle\":\"99.86\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_jzwb_1 = "{\"NavState\":\"0\",\"NavlockState\":\"2\",\"BeginTime\":\"2022-11-01T13:31:41.8051465+08:00\",\"EndTime\":\"2022-11-01T13:31:41.8051465+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"2\",\"UpInnerWaterLast\":\"40.01\",\"DownInnerWaterLast\":\"40.02\",\"tNavState\":\"5\",\"tUpWater\":\"64.76\",\"tUpInnerWater\":\"40.01\",\"tDownWater\":\"40.05\",\"tDownInnerWater\":\"40.02\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"2.04\"},\"Right\":{\"State\":3,\"OpenAngle\":\"1.88\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_jzwb_2 = "{\"NavState\":\"0\",\"NavlockState\":\"2\",\"BeginTime\":\"2022-11-01T13:32:01.5524656+08:00\",\"EndTime\":\"2022-11-01T13:32:01.5524656+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"2\",\"UpInnerWaterLast\":\"40.01\",\"DownInnerWaterLast\":\"40.01\",\"tNavState\":\"5\",\"tUpWater\":\"64.74\",\"tUpInnerWater\":\"40.01\",\"tDownWater\":\"40.05\",\"tDownInnerWater\":\"40.01\",\"tUpGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"-0.32\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.54\"}},\"tDownGates\":{\"Left\":{\"State\":3,\"OpenAngle\":\"0.39\"},\"Right\":{\"State\":3,\"OpenAngle\":\"0.13\"}},\"tUpSignalLights\":{\"Left\":2,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"

	up_czz_1_0 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T13:47:07.7338776+08:00\",\"EndTime\":\"2022-11-01T13:47:07.7338776+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.63\",\"DownInnerWaterLast\":\"64.67\",\"tNavState\":\"5\",\"tUpWater\":\"64.62\",\"tUpInnerWater\":\"64.63\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"64.67\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.19\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_czz_1_1 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T13:49:42.0838782+08:00\",\"EndTime\":\"2022-11-01T13:49:42.0838782+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.95\",\"DownInnerWaterLast\":\"64.98\",\"tNavState\":\"5\",\"tUpWater\":\"64.97\",\"tUpInnerWater\":\"64.95\",\"tDownWater\":\"40.04\",\"tDownInnerWater\":\"64.98\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.81\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.19\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
	up_czz_1_2 = "{\"NavState\":\"0\",\"NavlockState\":\"4\",\"BeginTime\":\"2022-11-01T14:00:22.8001839+08:00\",\"EndTime\":\"2022-11-01T14:00:22.8001839+08:00\",\"NavStateLast\":\"0\",\"NavlockStateLast\":\"4\",\"UpInnerWaterLast\":\"64.78\",\"DownInnerWaterLast\":\"64.81\",\"tNavState\":\"5\",\"tUpWater\":\"64.78\",\"tUpInnerWater\":\"64.78\",\"tDownWater\":\"40.03\",\"tDownInnerWater\":\"64.81\",\"tUpGates\":{\"Left\":{\"State\":2,\"OpenAngle\":\"100.13\"},\"Right\":{\"State\":2,\"OpenAngle\":\"100.88\"}},\"tDownGates\":{\"Left\":{\"State\":4,\"OpenAngle\":\"0.19\"},\"Right\":{\"State\":4,\"OpenAngle\":\"-0.13\"}},\"tUpSignalLights\":{\"Left\":1,\"Right\":2},\"tDownSignalLights\":{\"Left\":2,\"Right\":2}}"
)

func LoadJsonString(t *testing.T, ss string) (rt *gzb_sche.NavlockDynamicStatusData) {
	rt = &gzb_sche.NavlockDynamicStatusData{}
	if err := cm.Json.UnmarshalFromString(ss, rt); err != nil {
		t.Logf("Json decode error: %v", err)
		t.Fail()
	}
	return
}

func TestLoadJsonString(t *testing.T) {
	nds0 := LoadJsonString(t, up_czz_0)
	t.Logf("data: %+v", nds0)
	nds1 := LoadJsonString(t, up_czz_1)
	t.Logf("data: %+v", nds1)
	nds2 := LoadJsonString(t, up_czz_2)
	t.Logf("data: %+v", nds2)
}

func Push_Up_czz(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, up_czz_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, up_czz_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, up_czz_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Down_jzz(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, down_jzz_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzz_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, down_jzz_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzz_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, down_jzz_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzz_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Down_jzwb(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, down_jzwb_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzwb_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, down_jzwb_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzwb_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, down_jzwb_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_jzwb_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Down_czz(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, down_czz_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_czz_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, down_czz_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_czz_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, down_czz_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,down_czz_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Up_1_jzz(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, up_jzz_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzz_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, up_jzz_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzz_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, up_jzz_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzz_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Up_1_jzwb(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, up_jzwb_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzwb_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, up_jzwb_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzwb_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, up_jzwb_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_jzwb_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func Push_Up_czz1(t *testing.T, trs *gzb_sche.TrackingSt) {
	nds0 := LoadJsonString(t, up_czz_1_0)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds0, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_1_0: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds1 := LoadJsonString(t, up_czz_1_1)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds1, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_1_1: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
	nds2 := LoadJsonString(t, up_czz_1_2)
	trs.PredicateSchedule.PushNavlockAutoStatus(nds2, trs, g.LocalDB, g.ProduceDB, cm.L)
	t.Logf("navs:%s,navlocks:%s,up_czz_1_2: %+v", trs.PredicateSchedule.GetNavStatus(), trs.PredicateSchedule.GetNavlockStatus(), trs.PredicateSchedule.String())
}

func TestNavlockStatus(t *testing.T) {
	trks := gzb_sche.NewTrackingSt(1, 2, true)
	Push_Up_czz(t, trks)
	Push_Down_jzz(t, trks)
	Push_Down_jzwb(t, trks)
	Push_Down_czz(t, trks)
	Push_Up_1_jzz(t, trks)
	Push_Up_1_jzwb(t, trks)
	Push_Up_czz1(t, trks)
}

func TestPredicateComposite(t *testing.T) {

}
