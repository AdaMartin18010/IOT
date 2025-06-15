package web

// 从数据库中读取
type NavShipSpeedSetup struct {
	Id        uint64  `gorm:"column:id" json:"id"`
	NavlockId int64   `gorm:"column:navLockId;index:idx_sss_navdis" json:"navLockId"` //船闸名称
	Distance  float64 `gorm:"column:distance;index:idx_sss_navdis" json:"distance"`   //距离雷达距离
	SpeedMax  float64 `gorm:"column:speedMax" json:"speedMax"`                        //最大速度
}

func (nss *NavShipSpeedSetup) TableName() string {
	return "nav_shipspeed_setup"
}
