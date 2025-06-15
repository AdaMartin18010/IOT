package model

import (
	"gorm.io/gorm"
)

// 数据库相关的定义和表结构
// sqlite 索引是全局的 mysql的索引是表结构相关的
// 适配sqlite的索引命名 要注意索引编号

// 由于同步到平台 同步后的记录会软删除并分时进行硬删除
// 只有船速相关的信息做逻辑备份 所以不分表和分库

// 规避复合索引 和 索引的单独命名 适配基础的建模sql的大部分特性
// 系统所有数据的基础组成部分
// type SysScheduleBase struct {
// 	gorm.Model
// 	NavigationlockId string `gorm:"index:idx_navigationlock,priority:1"` // 船闸标识
// 	ScheduleId       string `gorm:"index:idx_navigationlock,priority:2"` // 调度ID -闸次号 一个闸次标识一次上行或者下行通航的完成
// 	ScheduleStatus   string `gorm:"index:idx_navigationlock"`            // 通航状态-上下行出入闸 取最粗略的状态
// }

const (
	// 信息版本从0开始小于10000 由各个不同的子模块定义
	// 如果为测试的信息则从10000 开始也就是信息类型的值加上10000
	FormatVersionTest = 10000
	LevelDebug        = 1
	LevelInfo         = 2
	LevelWarn         = 3
	LevelError        = 4
	LevelFatal        = 5
)

// 索引编号 0-10
// 各个系统状态信息变更-记录表 用来记录各种服务 系统 驱动的运行时和状态转换的信息
// 比如:
//
//	船舶测速系统 上游测速单元  闸门左测速单元 云台和开关量的运行时信息
//	禁停系统 上游雷达 下游雷达  禁停区域等
//	船闸监测系统  上游门灯水位 通航状态 下游门灯水位等
type IotNavlCpntState struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_1;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_1;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_1;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_1;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`
	// 系统名称
	SystemName string `gorm:"index:idx_component_1;size:255;comment:'系统名称-1#-监测,2#-禁停,3#-测速'"`
	// 服务名称
	ServerName string `gorm:"index:idx_server_1;size:255;comment:'服务名称-监测,禁停,测速'"`
	// 系统驱动名称
	DriverName string `gorm:"index:idx_component_1;size:255;comment:'程序驱动名称-1#-监测,2#-禁停-下左,3#-测速-上右'"`
	// 信息格式的版本 ---- 与服务系统驱动一起来区分信息的格式 大于10000的是测试类的信息
	FormatVersion uint `gorm:"comment:'信息格式的版本号>=10000是测试类信息'"`
	// 信息等级 --- 信息
	InfoLevel uint `gorm:"comment:'信息等级-1-info,2-警告,3-告警,4-错误,等'"`
	// 信息内容 json --- 格式适配不一样 根据FormatVersion来区别
	Info any `gorm:"TYPE:json;comment:'信息内容-json格式'"`
}

// 记录所有服务的状态
type IotNavlCpntStatus struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_2;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_2;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_2;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_2;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`
	// 系统名称
	SystemName string `gorm:"index:idx_component_2;size:255;comment:'系统名称-1#-监测,2#-禁停,3#-测速'"`
	// 服务名称
	ServerName string `gorm:"index:idx_server_2;size:255;comment:'服务名称-监测,禁停,测速'"`
	// 系统驱动名称
	DriverName string `gorm:"index:idx_component_2;size:255;comment:'程序驱动名称-1#-监测,2#-禁停-下左,3#-测速-上右'"`
	// 信息格式的版本 ---- 与服务系统驱动一起来区分信息的格式 大于10000的是测试类的信息
	FormatVersion uint `gorm:"comment:'信息格式的版本号>=10000是测试类信息'"`
	// 信息等级 --- 信息
	InfoLevel uint `gorm:"comment:'信息等级-1-info,2-警告,3-告警,4-错误,等'"`
	// 信息内容 json --- 格式适配不一样 根据FormatVersion来区别
	Info any `gorm:"TYPE:json;comment:'信息内容-json格式'"`
}

// 船闸状态转换信息表: 上行入闸 上行出闸 下行入闸 下行出闸 水位信号灯闸门开闭等 汇总了所有服务的信息综合集成
// Navigationlock 船闸状态
type IotNavlStatus struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_3;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_3;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_3;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_3;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`

	FormatVersion uint `gorm:"comment:'信息格式的版本号'"`
	Info          any  `gorm:"TYPE:json;comment:'信息内容-json格式'"` //船闸状态的记录json格式
}

type IotNavStatus struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_4;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_4;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_4;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_4;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`

	FormatVersion uint `gorm:"comment:'信息格式的版本号'"`
	Info          any  `gorm:"TYPE:json;comment:'信息内容-json格式'"` //船闸状态的记录json格式
}

// 船闸调度闸次内的船速记录信息 筛选正确和过滤合并后的雷达测速的数据
type IotNavlSpeedInfo struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_5;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_5;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_5;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_5;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`

	FormatVersion uint `gorm:"comment:'信息格式的版本号'"`
	Info          any  `gorm:"TYPE:json;comment:'信息内容-json格式'"` // 船舶速度距离等 json 信息记录
}

// 所有设备的状态 历史状态记录表
type IotNavlDevicesStatus struct {
	gorm.Model
	// 船闸标识
	NavlockId string `gorm:"index:idx_navlockid_6;priority:1;size:255;comment:'船闸标识'"`
	// 通航状态
	NavStatus string `gorm:"index:idx_navstatus_6;size:255;comment:'通航状态-通航上下行 通航换向-上行换下行-下行换上行'"`
	// 船闸状态 - 上下行出入闸
	NavlockStatus string `gorm:"index:idx_navlockstatus_6;size:255;comment:'船闸状态-上下行出入闸'"`
	// 调度ID  - 闸次 一个闸次标识一次上行或者下行通航的完成
	ScheduleId string `gorm:"index:idx_scheduleid_6;priority:2;size:255;comment:'闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成'"`

	DeviceTag     string `gorm:"index:idx_devicetag_6;size:255;comment:'设备标签-[上游左雷达..]'"`
	FormatVersion uint   `gorm:"comment:'信息格式的版本号'"`
	Info          any    `gorm:"TYPE:json;comment:'信息内容-json格式'"` // 设备状态的不同json格式
}
