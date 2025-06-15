package web

import (
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavStopLineWarn struct {
	Id             int64     `gorm:"column:id"`
	Createdat      time.Time `gorm:"column:createdAt;index:idx_slw_navlid_sche"` //  发生时间
	NavlockId      int       `gorm:"column:navLockId;index:idx_slw_navlid_sche;comment:'船闸标识'"`
	ScheduleId     string    `gorm:"column:scheduleId;index:idx_slw_navlid_sche;size:255;comment:'闸次-[闸次号]'"`
	ScheduleStatus string    `gorm:"column:scheduleStatus;index:idx_slw_navlid_sche;size:255;comment:'船闸调度状态-[上下行出入闸]'"`
	DeviceTag      string    `gorm:"column:deviceTag;size:255;comment:'设备标识-[ip地址]'"`
	CrossLocation  string    `gorm:"column:crossLocation;size:255;comment:'越线位置-[上游,下游]'"`
	CrossLevel     string    `gorm:"column:crossLevel;size:255;comment:'越线等级-[警告,报警]'"`
	StoplineWidth  int       `gorm:"column:stoplineWidth;size:255;comment:'禁停线绿色区域宽度-[CM]'"`
	CrossDistance  int       `gorm:"column:crossDistance;size:255;comment:'越线距离-[CM]'"`
}

func (nstpl *NavStopLineWarn) TableName() string {
	return "nav_stopline_warn"
}

func (nstpl *NavStopLineWarn) SaveToDB(db *gorm.DB, l *zap.Logger) error {
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
	if txerr := tx.Create(nstpl).Error; txerr != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, txerr)
		tx.Rollback()
		return txerr
	}
	tx.Commit()
	return tx.Error
}
