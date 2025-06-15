package gezhouba

import "gorm.io/gorm"

// 索引编号-12
type NavlGzbStoplineWarn struct {
	gorm.Model
	NavlockId      string `gorm:"index:idx_navlock_12;size:255;comment:'船闸标识-[船闸名称]'"`
	ScheduleId     string `gorm:"index:idx_scheduleid_12;size:255;comment:'闸次-[闸次号]'"`
	ScheduleStatus string `gorm:"index:idx_schedulestatus_12;size:255;comment:'船闸调度状态-[上下行出入闸]'"`
	DeviceTag      string `gorm:"index:idx_devicetag_12;size:255;comment:'设备标识-[ip地址]'"`
	CrossLocation  string `gorm:"comment:'越线位置-[上游,下游]'"`
	CrossLevel     string `gorm:"comment:'越线等级-[警告,报警]'"`
	StoplineWidth  int    `gorm:"comment:'禁停线绿色区域宽度-[CM]'"`
	CrossDistance  int    `gorm:"comment:'越线距离-[CM]'"`
}
