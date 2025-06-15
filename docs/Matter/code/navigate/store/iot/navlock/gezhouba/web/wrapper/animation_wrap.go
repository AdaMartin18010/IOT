package wrapper

import (
	"fmt"
	"sync"
	"time"

	cm "navigate/common/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavAnimationRecord = gzb_db_web.NavAnimationRecord
type NavAnimationHistory = gzb_db_web.NavAnimationHistory

type NavAnimationWrap struct {
	rwm              *sync.RWMutex
	navAnimationLast *NavAnimationRecord
	navAnimation     *NavAnimationRecord
	isNotFirst       bool
}

func NewNavAnimationWrap(Navlockid int) *NavAnimationWrap {
	tmp := &NavAnimationWrap{
		rwm:              &sync.RWMutex{},
		navAnimationLast: &NavAnimationRecord{NavlockId: Navlockid},
		navAnimation:     &NavAnimationRecord{NavlockId: Navlockid},
		isNotFirst:       false,
	}
	return tmp
}

func (naw *NavAnimationWrap) Infos() string {
	naw.rwm.RLock()
	defer naw.rwm.RUnlock()

	rs, _ := cm.Json.Marshal(naw.navAnimation)
	return string(rs)
}

func (naw *NavAnimationWrap) IsNull() bool {
	naw.rwm.RLock()
	defer naw.rwm.RUnlock()

	return naw.navAnimation.IsNull(naw.navAnimation.NavlockId)
}

func (naw *NavAnimationWrap) UpdateToDB(db *gorm.DB, l *zap.Logger) (err error) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	if db == nil || l == nil {
		return fmt.Errorf("db or l cannot be nil")
	}

	//如果为空值 则不更新数据库
	if naw.navAnimation.IsNull(naw.navAnimation.NavlockId) {
		return nil
	}

	//如果不是第一次更新
	if naw.isNotFirst {
		//则对比记录相同 则不更新数据库 否则  更新数据库 并交换数据
		if naw.navAnimation.CompareSame(naw.navAnimationLast) {
			return nil
		}
	}

	dbDebug := db.Debug()
	txDelete := dbDebug.Begin()
	//1. 删除动画表中存在的历史数据
	if err := txDelete.Where("navLockId = ?", naw.navAnimation.NavlockId).Delete(naw.navAnimation).Error; err != nil {
		l.Sugar().Errorf(`delete db nav_animation error: %+v`, err)
		txDelete.Rollback()
	}
	txDelete.Commit()

	//2. 新建插入数据库
	txInsert := dbDebug.Begin()
	naw.navAnimation.Id = 0
	if err = txInsert.Create(naw.navAnimation).Error; err != nil {
		l.Sugar().Errorf(`insert to db nav_animation error: %+v`, err)
		txInsert.Rollback()
	}
	txInsert.Commit()

	// //1. 先更新
	// result := tx.Model(&naw.navAnimation).Select("*").Omit("id", "navLockId").Where("navLockId = ?", naw.navAnimation.NavlockId).Updates(naw.navAnimation)
	// if err = result.Error; err != nil {
	// 	l.Sugar().Errorf(`Update to db nav_animation error: %+v`, err)
	// 	tx.Rollback()
	// 	return err
	// }
	// tx.Commit()

	// l.Sugar().Debugf("result.RowsAffected: %d , naw.navAnimation.Id: %d", result.RowsAffected, naw.navAnimation.Id)
	// //2. 如果没有更新 说明是刚开始创建的表没有记录
	// //设置成零值值 gorm 就不会直接写入
	// naw.navAnimation.Id = 0
	// if result.RowsAffected == 0 {
	// 	if err = dbDebug.Create(naw.navAnimation).Error; err != nil {
	// 		l.Sugar().Errorf(`insert to db nav_animation error: %+v`, err)
	// 		return err
	// 	}
	// }

	navAnihis := (*NavAnimationHistory)(naw.navAnimation)
	//设置成零值值 gorm 就不会直接写入 而是新建插入避免主键冲突
	navAnihis.Id = 0
	txCreate := dbDebug.Begin()
	if err = txCreate.Create(navAnihis).Error; err != nil {
		l.Sugar().Errorf(`insert to db nav_animation_history error: %+v`, err)
		txCreate.Rollback()
		return err
	}
	txCreate.Commit()

	//更新完成后 直接交换
	tmp := naw.navAnimation
	naw.navAnimation = naw.navAnimationLast
	naw.navAnimationLast = tmp
	naw.isNotFirst = true

	return err
}

// 0: "通航状态未知",1: "通航换向-上行换下行",2: "通航换向-下行换上行",3: "通航允许",4: "通航禁止",5: "通航上行",6: "通航下行"
func (naw *NavAnimationWrap) SetNavStatus(NavStatus int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.NavStatus = NavStatus
	naw.navAnimation.Createdat = time.Now()
}

// 设置闸次号 和 调度状态 0:"闸室运行状态未知",1:"上行进闸中",2:"上行进闸完毕",3:"上行出闸中",4:"上行出闸完毕",5:"下行进闸中",6:"下行进闸完毕",7:"下行出闸中",8:"下行出闸完毕"
func (naw *NavAnimationWrap) SetScheduleIdAndNavlockStatus(Scheduleid string, NavlockStatus int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.ScheduleId = Scheduleid
	naw.navAnimation.NavlockStatus = NavlockStatus
	naw.navAnimation.Createdat = time.Now()
}

// 设置闸门状态上下左右 0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终"
func (naw *NavAnimationWrap) SetGatesStatus(UpLeftGateS int, UpRightGateS int, DownLeftGateS int, DownRightGateS int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.GateUpLeftStatus = UpLeftGateS
	naw.navAnimation.GateUpRightStatus = UpRightGateS
	naw.navAnimation.GateDownLeftStatus = DownLeftGateS
	naw.navAnimation.GateDownRightStatus = DownRightGateS
	naw.navAnimation.Createdat = time.Now()
}

// 设置上游禁停线状态 禁停状态(1正常，0失联)  禁停线下游告警(0无，1，告警，2警报)
func (naw *NavAnimationWrap) SetUpStoplineInfo(upS int, UpWidth int, UpWarn int, UpDistance float64) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.StoplineUpStatus = upS
	naw.navAnimation.StoplineUpWidth = UpWidth
	naw.navAnimation.StoplineUpWarn = UpWarn
	naw.navAnimation.StoplineUpDistance = UpDistance
	naw.navAnimation.Createdat = time.Now()
}

// 设置上游禁停线状态 禁停状态(1正常，0失联)  禁停线下游告警(0无，1，告警，2警报)
func (naw *NavAnimationWrap) SetDownStoplineInfo(downS int, downWidth int, downWarn int, downDistance float64) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.StoplineDownStatus = downS
	naw.navAnimation.StoplineDownWidth = downWidth
	naw.navAnimation.StoplineDownWarn = downWarn
	naw.navAnimation.StoplineDownDistance = downDistance
	naw.navAnimation.Createdat = time.Now()
}

// 船速告警: 0:无 1:超速
func (naw *NavAnimationWrap) SetLeftRadarInfo(warn int, distance float64, speed float64) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.ShipLeftWarn = warn
	naw.navAnimation.ShipLeftDistance = distance
	naw.navAnimation.ShipLeftSpeed = speed
	naw.navAnimation.Createdat = time.Now()
}

// 船速告警: 0:无 1:超速
func (naw *NavAnimationWrap) SetRightRadarInfo(warn int, distance float64, speed float64) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.ShipRightWarn = warn
	naw.navAnimation.ShipRightDistance = distance
	naw.navAnimation.ShipRightSpeed = speed
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetUpSensorsInfo(UpLeftState int, UpRightState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfUpLeftState = UpLeftState
	naw.navAnimation.SelfUpRightState = UpRightState
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetUpLeftSensorsInfo(UpLeftState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfUpLeftState = UpLeftState
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetUpRightSensorsInfo(UpRightState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfUpRightState = UpRightState
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetDownSensorsInfo(DownLeftState int, DownRightState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfDownLeftState = DownLeftState
	naw.navAnimation.SelfDownRightState = DownRightState
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetDownLeftSensorsInfo(DownLeftState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfDownLeftState = DownLeftState
	naw.navAnimation.Createdat = time.Now()
}

// 自有设备状态 0:正常 1:开关量故障 2:云台故障 3:雷达故障 4:开关量和云台故障 5:开关量和雷达故障 6:云台和雷达故障 7:全故障
func (naw *NavAnimationWrap) SetDownRightSensorsInfo(DownRightState int) {
	naw.rwm.Lock()
	defer naw.rwm.Unlock()

	naw.navAnimation.SelfDownRightState = DownRightState
	naw.navAnimation.Createdat = time.Now()
}
