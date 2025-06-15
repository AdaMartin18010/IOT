package schedule

import (
	"sync"
)

// 定义基础的状态和转换逻辑  模型当下只用于葛洲坝的定义
type ScheduleStatusWrapper struct {
	navStateStr     string
	navlockStateStr string
	navState        NavStatus
	navlockState    NavlockStatus

	rwm *sync.RWMutex
}

func NewScheduleStatusWrapper() *ScheduleStatusWrapper {
	scdsp := &ScheduleStatusWrapper{
		navStateStr:     "",
		navlockStateStr: "",
		navlockState:    0,
		navState:        0,
		rwm:             &sync.RWMutex{},
	}
	scdsp.navStateStr = scdsp.navState.String()
	scdsp.navlockStateStr = scdsp.navlockState.String()
	return scdsp
}

func (ssw *ScheduleStatusWrapper) GetNavlockState() NavlockStatus {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	return ssw.navlockState
}

func (ssw *ScheduleStatusWrapper) GetNavStateStr() string {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	return ssw.navStateStr
}

func (ssw *ScheduleStatusWrapper) GetNavlockStateStr() string {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	return ssw.navlockStateStr
}

func (ssw *ScheduleStatusWrapper) GetStateStrs() (string, string) {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	return ssw.navStateStr, ssw.navlockStateStr
}

func (ssw *ScheduleStatusWrapper) SetNavStatus(navs NavStatus) string {
	ssw.rwm.Lock()
	defer ssw.rwm.Unlock()
	ssw.navState = navs
	ssw.navStateStr = ssw.navState.String()
	return ssw.navlockStateStr
}

func (ssw *ScheduleStatusWrapper) SetNavlockStatus(navls NavlockStatus) string {
	ssw.rwm.Lock()
	defer ssw.rwm.Unlock()
	ssw.navlockState = navls
	ssw.navlockStateStr = ssw.navlockState.String()
	return ssw.navlockStateStr
}

func (ssw *ScheduleStatusWrapper) SetStatus(navs NavStatus, navls NavlockStatus) (string, string) {
	ssw.rwm.Lock()
	defer ssw.rwm.Unlock()
	ssw.navState = navs
	ssw.navlockState = navls
	ssw.navStateStr = ssw.navState.String()
	ssw.navlockStateStr = ssw.navlockState.String()
	return ssw.navStateStr, ssw.navlockStateStr
}
