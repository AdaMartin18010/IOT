package web

import (
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavSpeedCurve struct {
	Id                uint64    `gorm:"column:id" json:"id"`
	ScheduleStartTime time.Time `gorm:"column:scheduleStartTime;index:idx_sc_navlid_sche" json:"scheduleStartTime"` //闸次开始时间
	NavlockId         int64     `gorm:"column:navLockId;index:idx_sc_navlid_sche" json:"navLockId"`                 //船闸名称
	ScheduleId        string    `gorm:"column:scheduleId;index:idx_sc_navlid_sche" json:"scheduleId"`               //闸次号
	SpeedMax          float64   `gorm:"column:speedMax" json:"speedMax"`                                            //最大速度
	InSpeed           float64   `gorm:"column:inSpeed" json:"inSpeed"`                                              //进闸平均速度
	OutSpeed          float64   `gorm:"column:outSpeed" json:"outSpeed"`                                            //出闸平均速度
	ScheduleStatus    int64     `gorm:"column:scheduleStatus" json:"scheduleStatus"`                                //航向
	DataSeq           any       `gorm:"TYPE:json;column:dataSeq" json:"dataSeq"`
	//{"DataSeq": [{"Time": "2022-09-13T15:54:52+08:00", "Speed": 2, "Distance": 30}, {"Time": "2022-09-13T15:54:55+08:00", "Speed": 2, "Distance": 30}]}
}

func (nsc *NavSpeedCurve) TableName() string {
	return "nav_speed_curve_data"
}

func (nsc *NavSpeedCurve) SaveToDB(db *gorm.DB, l *zap.Logger) error {
	sqldb, err := db.DB()
	if err != nil {
		l.Sugar().Errorf("err: %+v", err)
		return err
	}

	if err = sqldb.Ping(); err != nil {
		l.Sugar().Errorf("err: %+v", err)
		return err
	}

	dbd := db.Debug()
	tx := dbd.Begin()
	if err = tx.Create(nsc).Error; err != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, err)
		tx.Rollback()
		return err
	}
	tx.Commit()
	return tx.Error
}
