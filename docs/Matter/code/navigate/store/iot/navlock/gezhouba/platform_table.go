package gezhouba

import (
	"time"

	mdl "navigate/common/model"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

var (
	L = mdl.L
)

// 索引编号-13
type NavlGzbSyncRecord struct {
	gorm.Model
	NavlockId string `gorm:"index:idx_navlock_13;size:255;comment:'船闸标识-[船闸名称]'"`
	Content   string `gorm:"size:255;comment:'同步表名字-[speedlimit,stopline]'"`
	FinishIdx int    `gorm:"comment:'同步完成的id-[id]'"`
}

func (spl *NavlGzbSyncRecord) TableName() string {
	return "sync_record"
}

// 临时平台同步表
type NavlGzbSpeedLimit struct {
	Sid            uint      `gorm:"column:Sid;primarykey;comment:'序号-int'"`
	Time           time.Time `gorm:"column:Time;string;comment:'发生时间-UTC'"`
	NavlockId      string    `gorm:"column:NavlockId;string;comment:'船闸名称-str'"`
	ScheduleId     string    `gorm:"column:ScheduleId;string;comment:'闸次-str'"`
	ScheduleStatus string    `gorm:"column:ScheduleStatus;string;comment:'船闸调度状态-str'"`
	DeviceTag      string    `gorm:"column:DeviceTag;string;comment:'设备标识IP-str'"`
	Warn           string    `gorm:"column:Warn;string;comment:'警告Flag-str'"`
	Speed          float32   `gorm:"column:Speed;string;comment:'船速(m/s)-str'"`
}

func (spl *NavlGzbSpeedLimit) TableName() string {
	return "speedlimit"
}

func (spl *NavlGzbSpeedLimit) SaveToDB(db *gorm.DB, l *zap.Logger) error {
	sqldb, err := db.DB()
	if err != nil {
		l.Sugar().Errorf("err: %+v", err)
		return err
	}
	if err := sqldb.Ping(); err != nil {
		l.Sugar().Errorf("err: %+v", err)
		return err
	}

	dbd := db.Debug()
	tx := dbd.Begin()
	if err := tx.Create(spl).Error; err != nil {
		l.Sugar().Errorf(`insert to db error: %+v`, err)
		tx.Rollback()
		return err
	}
	tx.Commit()
	return tx.Error
}

// type NavlGzbStopLine struct {
// 	Sid           uint      `gorm:"column:Sid;primarykey;comment:'序号-int'"`
// 	Time          time.Time `gorm:"column:Time;string;comment:'发生时间-UTC'"`
// 	NavlockId     string    `gorm:"column:NavlockId;string;comment:'船闸名称-str'"`
// 	ScheduleId    string    `gorm:"column:ScheduleId;string;comment:'闸次-str'"`
// 	CrossLocation string    `gorm:"column:CrossLocation;string;comment:'越线位置-str'"`
// 	CrossLevel    string    `gorm:"column:CrossLevel;string;comment:'越线等级-str'"`
// 	DeviceTag     string    `gorm:"column:DeviceTag;string;comment:'设备标识IP-str'"`
// 	StoplineWidth int       `gorm:"column:StoplineWidth;string;comment:'禁停线宽度cm-int'"`
// 	CrossDistance int       `gorm:"column:CrossDistance;string;comment:'越线距离cm-int'"`
// }

// func (stpl *NavlGzbStopLine) TableName() string {
// 	return "stopline"
// }

// func (stpl *NavlGzbStopLine) SaveToDB(db *gorm.DB, l *zap.Logger) error {
// 	sqldb, err := db.DB()
// 	if err != nil {
// 		l.Sugar().Errorf("err: %+v", err)
// 		return err
// 	}
// 	if err := sqldb.Ping(); err != nil {
// 		l.Sugar().Errorf("err: %+v", err)
// 		return err
// 	}

// 	dbd := db.Debug()
// 	tx := dbd.Begin()
// 	if err := tx.Create(stpl).Error; err != nil {
// 		l.Sugar().Errorf(`insert to db error: %+v`, err)
// 		tx.Rollback()
// 		return err
// 	}
// 	tx.Commit()
// 	return nil
// }

// 索引编号-14
type NavlGzbStopLineWarns struct {
	gorm.Model
	NavigationlockId string `gorm:"index:idx_navlock_14;size:255;comment:'船闸标识-[船闸名称]'"`
	ScheduleId       string `gorm:"index:idx_scheduleid_14;size:255;comment:'闸次-[闸次号]'"`
	ScheduleStatus   string `gorm:"index:idx_schedulestatus_14;size:255;comment:'船闸调度状态-[上下行出入闸]'"`
	DeviceTag        string `gorm:"index:idx_devicetag_14;size:255;comment:'设备标识-[ip地址]'"`
	CrossLocation    string `gorm:"comment:'越线位置-[上游,下游]'"`
	CrossLevel       string `gorm:"comment:'越线等级-[警告,报警]'"`
	StoplineWidth    int    `gorm:"comment:'禁停线绿色区域宽度-[CM]'"`
	CrossDistance    int    `gorm:"comment:'越线距离-[CM]'"`
}

func (nstpl *NavlGzbStopLineWarns) TableName() string {
	return "nav_stopline_warns"
}

func (nstpl *NavlGzbStopLineWarns) SaveToDB(db *gorm.DB, l *zap.Logger) error {
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
