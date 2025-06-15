package web

import (
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavSpeedRaw struct {
	Id             uint64    `gorm:"column:id" json:"id"`
	CreatedAt      time.Time `gorm:"column:createdAt;index:idx_sr_navlid_sche" json:"createdAt"`   //发生时间
	NavlockId      int64     `gorm:"column:navLockId;index:idx_sr_navlid_sche" json:"navLockId"`   //船闸名称(1号闸 1,2号闸2,3号闸3)
	ScheduleStatus int64     `gorm:"column:scheduleStatus" json:"scheduleStatus"`                  //上下游进出闸标志(上行进闸1,上行出闸2,下行进闸3,下行出闸4)
	ScheduleId     string    `gorm:"column:scheduleId;index:idx_sr_navlid_sche" json:"scheduleId"` //闸次
	Warn           string    `gorm:"column:warn" json:"warn"`                                      //预警
	Distance       float64   `gorm:"column:distance" json:"distance"`                              //距雷达距离
	Speed          float64   `gorm:"column:speed" json:"speed"`                                    //速度
	RadarTag       string    `gorm:"column:radarTag" json:"radarTag"`                              //方位
}

func (nosw *NavSpeedRaw) TableName() string {
	return "nav_overspeed_warn"
}

func (nosw *NavSpeedRaw) SaveToDB(db *gorm.DB, l *zap.Logger) error {
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
	if err = tx.Create(nosw).Error; err != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, err)
		tx.Rollback()
		return err
	}
	tx.Commit()
	return tx.Error
}
