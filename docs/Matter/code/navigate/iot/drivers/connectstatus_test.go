package drivers_test

import (
	"fmt"
	dir "navigate/iot/drivers"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	"testing"
	"time"
)

func TestDriverConnectStatusWraper(t *testing.T) {
	fmt.Printf("now():%s\n", time.Now().String())
	dcsr := dir.NewDriverConnectStatus()
	dcsr.SetState(dir.Connecting)
	dcsr.SetState(dir.Connecting)
	dcsr.SetState(dir.Connecting)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.Connecting)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.Connected)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.DoDisConnected)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.Connecting)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.Connected)
	time.Sleep(300 * time.Millisecond)
	dcsr.SetState(dir.BeDisConnected)

	fmt.Printf("DriverCommonStatusWraper:%+v\n", dcsr.String())

}

func TestScheduleIdWraper(t *testing.T) {
	fmt.Printf("now():%s\n", time.Now().String())
	sid := gzb_sche.NewScheduleIdWrapper(1)
	ss := sid.GetScheduleId()
	fmt.Printf("ScheduleId:%s\n", ss)
	time.Sleep(1000 * time.Millisecond)
	ss = sid.GenerationGet()
	fmt.Printf("ScheduleId:%s\n", ss)
}
