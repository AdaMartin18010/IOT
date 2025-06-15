package schedule_test

import (
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	"testing"
)

func TestStatusWeb(t *testing.T) {

	//var ts gzb_sche.NavlockStatus
	ts := gzb_sche.NavlockUpWaterTrendUp
	t.Logf("%+v", ts.TransformToWebDefine())
	ts = gzb_sche.NavlockDownWaterTrendDown
	t.Logf("%+v", ts.TransformToWebDefine())
}
