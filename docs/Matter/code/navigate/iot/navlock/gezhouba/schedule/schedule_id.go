package schedule

import (
	"fmt"
	"sync"
	"time"
)

// 统一生成闸次ID-调度号 标识每次船闸状态转换的封装
// 默认DI 需要全局唯一
type ScheduleIdWrapper struct {
	id        string
	navlockid int

	rwm *sync.RWMutex
}

func NewScheduleIdWrapper(navigationlockid int) *ScheduleIdWrapper {
	return &ScheduleIdWrapper{
		id:        "",
		navlockid: navigationlockid,
		rwm:       &sync.RWMutex{},
	}
}

func (sid *ScheduleIdWrapper) GetScheduleId() string {
	sid.rwm.RLock()
	defer sid.rwm.RUnlock()
	return sid.id
}

// 可以用来标识倒换闸
func (sid *ScheduleIdWrapper) SetScheduleId(scheId string) {
	sid.rwm.Lock()
	defer sid.rwm.Unlock()
	sid.id = scheId
}

func (sid *ScheduleIdWrapper) GenerationGet() string {
	sid.rwm.Lock()
	defer sid.rwm.Unlock()
	sid.id = getDefaultScheduleId(sid.navlockid)
	return sid.id
}

// 获取 根据时间和船闸编号来生成的闸次ID 23位
func getDefaultScheduleId(navigationlockid int) (scid string) {
	now := time.Now()
	t1 := now.Year()   //年
	t2 := now.Month()  //月
	t3 := now.Day()    //日
	t4 := now.Hour()   //小时
	t5 := now.Minute() //分钟
	t6 := now.Second() //秒
	//t7 := now.Nanosecond() //纳秒
	scid = fmt.Sprintf("#%01d-%04d%02d%02d%02d%02d%02d", navigationlockid, t1, t2, t3, t4, t5, t6)
	return
}
