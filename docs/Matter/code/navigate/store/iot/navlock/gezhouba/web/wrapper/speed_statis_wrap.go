package wrapper

import (
	"fmt"
	"math"
	"time"

	"sync"

	cm "navigate/common/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavSpeedRaw = gzb_db_web.NavSpeedRaw
type NavSpeedCurve = gzb_db_web.NavSpeedCurve
type NavSpeedStatis = gzb_db_web.NavSpeedStatis

type NavSpeedRawDataArray []NavSpeedRaw

type NavSpeedStatisWrap struct {
	rwm                  *sync.RWMutex
	navSpeedStatis       *NavSpeedStatis
	navUpInSpeedCurve    *NavSpeedCurve
	navUpOutSpeedCurve   *NavSpeedCurve
	navDownInSpeedCurve  *NavSpeedCurve
	navDownOutSpeedCurve *NavSpeedCurve
	tmpArray             NavSpeedRawDataArray
}

type SpeedData struct {
	Time     time.Time `json:"Time"`
	Distance float64   `json:"Distance"`
	Speed    float64   `json:"Speed"`
}

type DataSeq struct {
	DataSeq []SpeedData `json:"DataSeq"`
}

func (d *DataSeq) JsonFormat(l *zap.Logger) []byte {
	json, err := cm.Json.Marshal(d)
	if err != nil {
		l.Sugar().Errorf("Json.Marshal err: %+v", err)
	}
	return json
}

// 初始化的时候传入 navlockId 后续的查询和统计都是基于此时传入的参数
func NewNavSpeedStatisWrap(navlockId int) *NavSpeedStatisWrap {

	return &NavSpeedStatisWrap{
		rwm:                  &sync.RWMutex{},
		navSpeedStatis:       &gzb_db_web.NavSpeedStatis{NavlockId: navlockId},
		navUpInSpeedCurve:    &gzb_db_web.NavSpeedCurve{NavlockId: int64(navlockId)},
		navUpOutSpeedCurve:   &gzb_db_web.NavSpeedCurve{NavlockId: int64(navlockId)},
		navDownInSpeedCurve:  &gzb_db_web.NavSpeedCurve{NavlockId: int64(navlockId)},
		navDownOutSpeedCurve: &gzb_db_web.NavSpeedCurve{NavlockId: int64(navlockId)},
		tmpArray:             nil,
	}
}

func (sls *NavSpeedStatisWrap) Infos() string {
	sls.rwm.RLock()
	defer sls.rwm.RUnlock()

	rs, _ := cm.Json.Marshal(sls.navSpeedStatis)
	return string(rs)
}

func (sls *NavSpeedStatisWrap) LoadSpeedDataFromDB(scheduleId string, db *gorm.DB, l *zap.Logger) (err error) {
	if db == nil || l == nil {
		return fmt.Errorf("db or l cannot be nil")
	}

	sls.rwm.Lock()
	defer sls.rwm.Unlock()

	dbUsed := db.Debug()

	res := new([]NavSpeedRaw)
	result := dbUsed.Model(&NavSpeedRaw{}).Where("navLockId = ? and scheduleId = ?", sls.navSpeedStatis.NavlockId, scheduleId).Order("createdAt ASC").Find(res)
	if result.Error != nil {
		l.Sugar().Errorf(`get NavSpeedRaw from  db table %s error: %+v`, (&NavSpeedRaw{}).TableName(), result.Error)
		return result.Error
	}

	if result.RowsAffected > 0 {
		sls.tmpArray = *res
		l.Sugar().Debugf("%+v", sls.tmpArray)
	}

	return nil
}

func (sls *NavSpeedStatisWrap) SpeedStatisCompute(scheduleId string, db *gorm.DB, l *zap.Logger) (err error) {
	if db == nil || l == nil {
		return fmt.Errorf("db or l cannot be nil")
	}

	sls.rwm.Lock()
	defer sls.rwm.Unlock()

	//在存入数据库无错误的情况下 清空navStoplineStatis 方便重新计算
	defer func() {
		if err == nil {
			sls.navSpeedStatis = &gzb_db_web.NavSpeedStatis{NavlockId: sls.navSpeedStatis.NavlockId}
			sls.navUpInSpeedCurve = &gzb_db_web.NavSpeedCurve{NavlockId: sls.navUpInSpeedCurve.NavlockId}
			sls.navUpOutSpeedCurve = &gzb_db_web.NavSpeedCurve{NavlockId: sls.navUpInSpeedCurve.NavlockId}
			sls.navDownInSpeedCurve = &gzb_db_web.NavSpeedCurve{NavlockId: sls.navDownInSpeedCurve.NavlockId}
			sls.navDownOutSpeedCurve = &gzb_db_web.NavSpeedCurve{NavlockId: sls.navDownInSpeedCurve.NavlockId}
			sls.tmpArray = nil
		}
	}()

	if sls.tmpArray == nil || len(sls.tmpArray) == 0 {
		return nil
	}

	sls.navSpeedStatis.ScheduleStartTime = sls.tmpArray[0].CreatedAt
	sls.navUpInSpeedCurve.ScheduleStartTime = sls.tmpArray[0].CreatedAt
	sls.navUpOutSpeedCurve.ScheduleStartTime = sls.tmpArray[0].CreatedAt
	sls.navDownInSpeedCurve.ScheduleStartTime = sls.tmpArray[0].CreatedAt
	sls.navDownOutSpeedCurve.ScheduleStartTime = sls.tmpArray[0].CreatedAt

	sls.navSpeedStatis.ScheduleId = scheduleId
	sls.navUpInSpeedCurve.ScheduleId = scheduleId
	sls.navUpOutSpeedCurve.ScheduleId = scheduleId
	sls.navDownInSpeedCurve.ScheduleId = scheduleId
	sls.navDownOutSpeedCurve.ScheduleId = scheduleId

	UpInSpeedCurve := new([]SpeedData)
	UpOutSpeedCurve := new([]SpeedData)
	DownInSpeedCurve := new([]SpeedData)
	DownOutSpeedCurve := new([]SpeedData)

	OverSpeedCount := 0 //总超速次数

	UpInOverSpeedCount := 0    //上行进闸超速次数
	UpOutOverSpeedCount := 0   //上行出闸超速次数
	DownInOverSpeedCount := 0  //下行进闸超速次数
	DownOutOverSpeedCount := 0 //下行出闸超速次数

	MaxSpeed := 0.0        //最高速度
	UpInMaxSpeed := 0.0    //上行进闸最高速度
	UpOutMaxSpeed := 0.0   //上行出闸最高速度
	DownInMaxSpeed := 0.0  //下行进闸最高速度
	DownOutMaxSpeed := 0.0 //下行出闸最高速度

	UpInAvgSpeed := 0.0   //上行进闸平均速度
	UpInCount := 0        //上行进闸船次
	UpInSpeedTotal := 0.0 //上行进闸总速度值

	UpOutAvgSpeed := 0.0   //上行出闸平均速度
	UpOutCount := 0        //上行进闸船次
	UpOutSpeedTotal := 0.0 // 上行进闸总速度值

	DownInAvgSpeed := 0.0   //下行进闸平均速度
	DownInCount := 0        // 下行进闸船次
	DownInSpeedTotal := 0.0 // 下行进闸总速度值

	DownOutAvgSpeed := 0.0   //下行出闸平均速度
	DownOutCount := 0        //下行进闸船次
	DownOutSpeedTotal := 0.0 //下行进闸总速度值

	for i := range sls.tmpArray {
		spr := sls.tmpArray[i]
		//上下游进出闸标志(上行进闸1,上行出闸2,下行进闸3,下行出闸4)
		switch spr.ScheduleStatus {
		case 1:
			{
				if spr.Speed > UpInMaxSpeed {
					UpInMaxSpeed = spr.Speed
				}

				switch spr.Warn {
				case "超速":
					{
						OverSpeedCount++
						UpInOverSpeedCount++
					}
				}
				*UpInSpeedCurve = append(*UpInSpeedCurve, SpeedData{Time: spr.CreatedAt, Distance: spr.Distance, Speed: spr.Speed})
				UpInCount++
				UpInSpeedTotal += spr.Speed
			}
		case 2:
			{
				if spr.Speed > UpOutMaxSpeed {
					UpOutMaxSpeed = spr.Speed
				}

				switch spr.Warn {
				case "超速":
					{
						OverSpeedCount++
						UpOutOverSpeedCount++
					}
				}
				*UpOutSpeedCurve = append(*UpOutSpeedCurve, SpeedData{Time: spr.CreatedAt, Distance: spr.Distance, Speed: spr.Speed})
				UpOutCount++
				UpOutSpeedTotal += spr.Speed
			}
		case 3:
			{
				if spr.Speed > DownInMaxSpeed {
					DownInMaxSpeed = spr.Speed
				}

				switch spr.Warn {
				case "超速":
					{
						OverSpeedCount++
						DownInOverSpeedCount++
					}
				}

				*DownInSpeedCurve = append(*DownInSpeedCurve, SpeedData{Time: spr.CreatedAt, Distance: spr.Distance, Speed: spr.Speed})
				DownInCount++
				DownInSpeedTotal += spr.Speed
			}
		case 4:
			{
				if spr.Speed > DownOutMaxSpeed {
					DownOutMaxSpeed = spr.Speed
				}

				switch spr.Warn {
				case "超速":
					{
						OverSpeedCount++
						DownOutOverSpeedCount++
					}
				}

				*DownOutSpeedCurve = append(*DownOutSpeedCurve, SpeedData{Time: spr.CreatedAt, Distance: spr.Distance, Speed: spr.Speed})
				DownOutCount++
				DownOutSpeedTotal += spr.Speed
			}
		}
	}

	MaxSpeed = math.Max(UpInMaxSpeed, UpOutMaxSpeed)
	MaxSpeed = math.Max(MaxSpeed, DownInMaxSpeed)
	MaxSpeed = math.Max(MaxSpeed, DownOutMaxSpeed)

	if UpInCount > 0 {
		UpInAvgSpeed = UpInSpeedTotal / float64(UpInCount)
	}

	if UpOutCount > 0 {
		UpOutAvgSpeed = UpOutSpeedTotal / float64(UpOutCount)
	}

	if DownInCount > 0 {
		DownInAvgSpeed = DownInSpeedTotal / float64(DownInCount)
	}

	if DownOutCount > 0 {
		DownOutAvgSpeed = DownOutSpeedTotal / float64(DownOutCount)
	}

	sls.navSpeedStatis.OverSpeedCount = int64(OverSpeedCount)
	sls.navSpeedStatis.UpInCount = int64(UpInOverSpeedCount)
	sls.navSpeedStatis.UpInSpeed = UpInAvgSpeed
	sls.navSpeedStatis.UpOutCount = int64(UpOutOverSpeedCount)
	sls.navSpeedStatis.UpOutSpeed = UpOutAvgSpeed
	sls.navSpeedStatis.DownInCount = int64(DownInOverSpeedCount)
	sls.navSpeedStatis.DownInSpeed = DownInAvgSpeed
	sls.navSpeedStatis.DownOutCount = int64(DownOutOverSpeedCount)
	sls.navSpeedStatis.DownOutSpeed = DownOutAvgSpeed
	sls.navSpeedStatis.MaxSpeed = MaxSpeed
	if len(*UpInSpeedCurve) > 0 {
		sls.navUpInSpeedCurve.ScheduleStatus = 1
		sls.navUpInSpeedCurve.SpeedMax = UpInMaxSpeed
		sls.navUpInSpeedCurve.InSpeed = UpInAvgSpeed
		sls.navUpInSpeedCurve.OutSpeed = 0.0
		sls.navUpInSpeedCurve.DataSeq = (&DataSeq{DataSeq: *UpInSpeedCurve}).JsonFormat(l)
		if err = sls.navUpInSpeedCurve.SaveToDB(db, l); err != nil {
			l.Sugar().Errorf("navUpInSpeedCurve.SaveToDB err : %+v", err)
		}
	}

	if len(*UpOutSpeedCurve) > 0 {
		sls.navUpOutSpeedCurve.ScheduleStatus = 1
		sls.navUpOutSpeedCurve.SpeedMax = UpOutMaxSpeed
		sls.navUpOutSpeedCurve.InSpeed = 0.0
		sls.navUpOutSpeedCurve.OutSpeed = UpOutAvgSpeed
		sls.navUpOutSpeedCurve.DataSeq = (&DataSeq{DataSeq: *UpOutSpeedCurve}).JsonFormat(l)
		if err = sls.navUpOutSpeedCurve.SaveToDB(db, l); err != nil {
			l.Sugar().Errorf("navUpOutSpeedCurve.SaveToDB err : %+v", err)
		}
	}

	if len(*DownInSpeedCurve) > 0 {
		sls.navDownInSpeedCurve.ScheduleStatus = 1
		sls.navDownInSpeedCurve.SpeedMax = DownInMaxSpeed
		sls.navDownInSpeedCurve.InSpeed = DownInAvgSpeed
		sls.navDownInSpeedCurve.OutSpeed = 0.0
		sls.navDownInSpeedCurve.DataSeq = (&DataSeq{DataSeq: *DownInSpeedCurve}).JsonFormat(l)
		if err = sls.navDownInSpeedCurve.SaveToDB(db, l); err != nil {
			l.Sugar().Errorf("navDownInSpeedCurve.SaveToDB err : %+v", err)
		}
	}

	if len(*DownOutSpeedCurve) > 0 {
		sls.navDownOutSpeedCurve.ScheduleStatus = 1
		sls.navDownOutSpeedCurve.SpeedMax = DownOutMaxSpeed
		sls.navDownOutSpeedCurve.InSpeed = 0.0
		sls.navDownOutSpeedCurve.OutSpeed = DownOutAvgSpeed
		sls.navDownOutSpeedCurve.DataSeq = (&DataSeq{DataSeq: *DownOutSpeedCurve}).JsonFormat(l)
		if err = sls.navDownOutSpeedCurve.SaveToDB(db, l); err != nil {
			l.Sugar().Errorf("navDownOutSpeedCurve.SaveToDB err : %+v", err)
		}
	}

	l.Sugar().Debugf("NavStoplineStatis: %+v", sls.navSpeedStatis)
	if err = sls.navSpeedStatis.SaveToDB(db, l); err != nil {
		l.Sugar().Errorf("navSpeedStatis.SaveToDB err : %+v", err)
	}

	return nil
}
