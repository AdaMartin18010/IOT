package web

import (
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavSpeedStatis struct {
	Id                uint64    `gorm:"column:id" json:"id"`
	ScheduleStartTime time.Time `gorm:"column:scheduleStartTime;index:idx_nss_navlid_sche" json:"scheduleStartTime"` //闸次开始时间
	NavlockId         int       `gorm:"column:navLockId;index:idx_nss_navlid_sche" json:"navLockId"`                 //船闸名称
	ScheduleId        string    `gorm:"column:scheduleId;index:idx_nss_navlid_sche" json:"scheduleId"`               //闸次号
	OverSpeedCount    int64     `gorm:"column:overSpeedCount" json:"overSpeedCount"`                                 //超速次
	UpInCount         int64     `gorm:"column:upInCount" json:"upInCount"`                                           //上行进闸超速次
	UpOutCount        int64     `gorm:"column:upOutCount" json:"upOutCount"`                                         //上行出闸超速次
	DownInCount       int64     `gorm:"column:downInCount" json:"downInCount"`                                       //下行进闸超速次
	DownOutCount      int64     `gorm:"column:downOutCount" json:"downOutCount"`                                     //下行出闸超速次
	MaxSpeed          float64   `gorm:"column:maxSpeed" json:"maxSpeed"`                                             //最高速度
	UpInSpeed         float64   `gorm:"column:upInSpeed" json:"upInSpeed"`                                           //上行进闸平均速度
	UpOutSpeed        float64   `gorm:"column:upOutSpeed" json:"upOutSpeed"`                                         //上行出闸平均速度
	DownInSpeed       float64   `gorm:"column:downInSpeed" json:"downInSpeed"`                                       //下行进闸平均速度
	DownOutSpeed      float64   `gorm:"column:downOutSpeed" json:"downOutSpeed"`                                     //下行出闸平均速度
}

func (nss *NavSpeedStatis) TableName() string {
	return "nav_speed_statis"
}

func (nss *NavSpeedStatis) SaveToDB(db *gorm.DB, l *zap.Logger) error {
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
	if err = tx.Create(nss).Error; err != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, err)
		tx.Rollback()
		return err
	}
	tx.Commit()
	return tx.Error
}
