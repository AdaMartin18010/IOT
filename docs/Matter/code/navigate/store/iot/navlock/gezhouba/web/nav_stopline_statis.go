package web

import (
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavStopLineStatis struct {
	Id                uint64    `gorm:"column:id" json:"id"`
	ScheduleStartTime time.Time `gorm:"column:scheduleStartTime;index:idx_sls_navlid_sche" json:"scheduleStartTime"` //闸次开始时间
	NavlockId         int       `gorm:"column:navLockId;index:idx_sls_navlid_sche" json:"navLockId"`                 //船闸名称编码
	ScheduleId        string    `gorm:"column:scheduleId;index:idx_sls_navlid_sche" json:"scheduleId"`               //闸次
	CrossCount        int64     `gorm:"column:crossCount" json:"crossCount"`                                         //越线数
	UpCount           int64     `gorm:"column:upCount" json:"upCount"`                                               //上游越线数
	DownCount         int64     `gorm:"column:downCount" json:"downCount"`                                           //下游越线数
	WarnCount         int64     `gorm:"column:warnCount" json:"warnCount"`                                           //警告数
	AlarmCount        int64     `gorm:"column:alarmCount" json:"alarmCount"`                                         //报警数
}

func (nsls *NavStopLineStatis) TableName() string {
	return "nav_stopline_statis"
}

func (nsls *NavStopLineStatis) SaveToDB(db *gorm.DB, l *zap.Logger) error {
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
	if err = db.Create(nsls).Error; err != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, err)
		tx.Rollback()
		return err
	}
	tx.Commit()
	return tx.Error
}
