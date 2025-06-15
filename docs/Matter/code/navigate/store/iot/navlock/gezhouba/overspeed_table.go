package gezhouba

// //索引编号-11
// // 同步到平台的表
// type NavlGzbShipSpeed struct {
// 	gorm.Model
// 	NavlockId      string  `gorm:"index:idx_navlock_11;size:255;comment:'船闸标识-[船闸名称]'"`
// 	ScheduleId     string  `gorm:"index:idx_scheduleid_11;size:255;comment:'闸次-[闸次号]'"`
// 	ScheduleStatus string  `gorm:"index:idx_schedulestatus_11;size:255;comment:'船闸调度状态-[上下行出入闸]'"`
// 	DeviceTag      string  `gorm:"index:idx_devicetag_11;size:255;comment:'设备标识-ip地址'"`
// 	Warn           string  `gorm:"comment:'警告-[超速]'"`
// 	Speed          float32 `gorm:"comment:'船速-[m/s]'"`
// 	RadarTag       string  `gorm:"comment:'雷达tag-[上右,下左]'"`
// }

// func (ss *NavlGzbShipSpeed) TableName() string {
// 	return "shipspeed"
// }
