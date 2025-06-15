package schedule_test

import (
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	"testing"
)

func TestShipSpeed(t *testing.T) {
	sps0 := gzb_sche.NewShipSpeeds()
	sps0.LeftSpeed.IsValid = true
	sps0.LeftSpeed.OverSpeed = false
	sps0.LeftSpeed.Speed = 1.23

	sps0.RightSpeed.IsValid = true
	sps0.RightSpeed.OverSpeed = false
	sps0.RightSpeed.Speed = 1.45

	str0, err0 := sps0.String()
	t.Logf("shipspeeds sps0 : %+v,error : %+v", str0, err0)

	sps := gzb_sche.NewShipSpeeds()
	sps.LeftSpeed.IsValid = true
	sps.LeftSpeed.OverSpeed = true
	sps.LeftSpeed.Speed = 2.12

	sps.RightSpeed.IsValid = true
	sps.RightSpeed.OverSpeed = true
	sps.RightSpeed.Speed = 1.57

	str, err := sps.String()
	t.Logf("shipspeeds sps: %+v,error : %+v", str, err)

	sps0.RightSpeed = nil
	sps.AggregationWith(sps0)
	str1, err1 := sps.String()
	t.Logf("shipspeeds AggregationWith: %+v,error : %+v", str1, err1)

}
