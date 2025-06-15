package wrapper

import (
	"fmt"

	"sync"

	cm "navigate/common/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavStoplineWarn = gzb_db_web.NavStopLineWarn
type NavStoplineStatis = gzb_db_web.NavStopLineStatis

type NavStoplineArray []NavStoplineWarn

type NavStopLineStatisWrap struct {
	rwm               *sync.RWMutex
	navStoplineStatis *NavStoplineStatis
	tmpArray          NavStoplineArray
}

// 初始化的时候传入 navlockId 后续的查询和统计都是基于此时传入的参数
func NewNavStopLineStatisWrap(navlockId int) *NavStopLineStatisWrap {

	return &NavStopLineStatisWrap{
		rwm:               &sync.RWMutex{},
		navStoplineStatis: &gzb_db_web.NavStopLineStatis{NavlockId: navlockId},
		tmpArray:          nil,
	}
}

func (sls *NavStopLineStatisWrap) Infos() string {
	sls.rwm.RLock()
	defer sls.rwm.RUnlock()

	rs, _ := cm.Json.Marshal(sls.navStoplineStatis)
	return string(rs)
}

func (sls *NavStopLineStatisWrap) LoadStopLineDataFromDB(scheduleId string, db *gorm.DB, l *zap.Logger) (err error) {
	if db == nil || l == nil {
		return fmt.Errorf("db or l cannot be nil")
	}

	sls.rwm.Lock()
	defer sls.rwm.Unlock()

	dbUsed := db.Debug()
	dbT := dbUsed.Begin()

	res := new([]NavStoplineWarn)
	result := dbT.Model(&NavStoplineWarn{}).Where("navLockId = ? and scheduleId = ?", sls.navStoplineStatis.NavlockId, scheduleId).Order("createdAt ASC").Find(res)
	if result.Error != nil {
		l.Sugar().Errorf(`get Stopline from  db table %s error: %+v`, (&NavStoplineWarn{}).TableName(), result.Error)
		result.Rollback()
		return result.Error
	}

	if result.RowsAffected > 0 {
		sls.tmpArray = *res
		l.Sugar().Debugf("%+v", sls.tmpArray)
	}
	result.Commit()

	return nil
}

func (sls *NavStopLineStatisWrap) StatisCompute(scheduleId string, db *gorm.DB, l *zap.Logger) (err error) {
	if db == nil || l == nil {
		return fmt.Errorf("db or l cannot be nil")
	}

	sls.rwm.Lock()
	defer sls.rwm.Unlock()

	//在存入数据库无错误的情况下 清空navStoplineStatis 方便重新计算
	defer func() {
		if err == nil {
			sls.navStoplineStatis = &gzb_db_web.NavStopLineStatis{NavlockId: sls.navStoplineStatis.NavlockId}
			sls.tmpArray = nil
		}
	}()

	if sls.tmpArray == nil || len(sls.tmpArray) == 0 {
		return nil
	}

	sls.navStoplineStatis.ScheduleStartTime = sls.tmpArray[0].Createdat
	sls.navStoplineStatis.ScheduleId = scheduleId
	CrossCount := 0 //越线数
	UpCount := 0    //上游越线数
	DownCount := 0  //下游越线数
	WarnCount := 0  //警告数
	AlarmCount := 0 //报警数
	for i := range sls.tmpArray {
		slr := sls.tmpArray[i]
		switch slr.CrossLevel {
		case "1", "警告":
			{
				CrossCount++
				WarnCount++
				switch slr.CrossLocation {
				case "上游":
					UpCount++
				case "下游":
					DownCount++
				}
			}
		case "2", "报警":
			{
				CrossCount++
				AlarmCount++
				switch slr.CrossLocation {
				case "上游":
					UpCount++
				case "下游":
					DownCount++
				}
			}
		}
	}

	sls.navStoplineStatis.CrossCount = int64(CrossCount)
	sls.navStoplineStatis.UpCount = int64(UpCount)
	sls.navStoplineStatis.DownCount = int64(DownCount)
	sls.navStoplineStatis.WarnCount = int64(WarnCount)
	sls.navStoplineStatis.AlarmCount = int64(AlarmCount)

	l.Sugar().Debugf("NavStoplineStatis: %+v", sls.navStoplineStatis)
	if err = sls.navStoplineStatis.SaveToDB(db, l); err != nil {
		return err
	}

	return nil
}
