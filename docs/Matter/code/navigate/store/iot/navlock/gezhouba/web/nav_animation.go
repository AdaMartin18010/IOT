package web

import (
	"time"

	cm "navigate/common"
)

const floatCompareMin = 0.01

type NavAnimationRecord struct {
	Id            int       `gorm:"column:id;primarykey;auto_increment" json:"id"`
	Createdat     time.Time `gorm:"column:createdAt;index:idx_nar_navlid_sche" json:"createdAt"`   //发生时间
	NavlockId     int       `gorm:"column:navLockId;index:idx_nar_navlid_sche" json:"navLockId"`   //船闸名称
	ScheduleId    string    `gorm:"column:scheduleId;index:idx_nar_navlid_sche" json:"scheduleId"` //闸次号
	NavStatus     int       `gorm:"column:navStatus" json:"navStatus"`                             //0: "通航状态未知",1: "通航换向-上行换下行",2: "通航换向-下行换上行",3: "通航允许",4: "通航禁止",5: "通航上行",6: "通航下行"
	NavlockStatus int       `gorm:"column:navLockStatus" json:"navLockStatus"`                     //0:"闸室运行状态未知",1:"上行进闸中",2:"上行进闸完毕",3:"上行出闸中",4:"上行出闸完毕",5:"下行进闸中",6:"下行进闸完毕",7:"下行出闸中",8:"下行出闸完毕"

	GateUpLeftStatus    int `gorm:"column:gateLeftUpStatus" json:"gateLeftUpStatus"`       //左上闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateUpRightStatus   int `gorm:"column:gateRightUpStatus" json:"gateRightUpStatus"`     //右上闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateDownLeftStatus  int `gorm:"column:gateLeftDownStatus" json:"gateLeftDownStatus"`   //左下闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateDownRightStatus int `gorm:"column:gateRightDownStatus" json:"gateRightDownStatus"` //右下闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",

	StoplineUpStatus   int     `gorm:"column:stoplineUpStatus" json:"stoplineUpStatus"`     //禁停状态(1正常，0失联)
	StoplineUpWidth    int     `gorm:"column:stoplineUpWidth" json:"stoplineUpWidth"`       //禁停线上游宽度
	StoplineUpWarn     int     `gorm:"column:stoplineUpWarn" json:"stoplineUpWarn"`         //禁停线上游告警(0无，1，告警，2警报)
	StoplineUpDistance float64 `gorm:"column:stoplineUpDistance" json:"stoplineUpDistance"` //上游超出禁停线距离

	StoplineDownStatus   int     `gorm:"column:stoplineDownStatus" json:"stoplineDownStatus"`     //禁停状态(1正常，0失联)
	StoplineDownWidth    int     `gorm:"column:stoplineDownWidth" json:"stoplineDownWidth"`       //禁停线下游宽度
	StoplineDownWarn     int     `gorm:"column:stoplineDownWarn" json:"stoplineDownWarn"`         //禁停线下游告警(0无，1，告警，2警报)
	StoplineDownDistance float64 `gorm:"column:stoplineDownDistance" json:"stoplineDownDistance"` //下游超出禁停线距离

	ShipLeftDistance float64 `gorm:"column:shipLeftDistance" json:"shipLeftDistance"` //左船雷达距离
	ShipLeftSpeed    float64 `gorm:"column:shipLeftSpeed" json:"shipLeftSpeed"`       //左船雷达速度
	ShipLeftWarn     int     `gorm:"column:shipLeftWarn" json:"shipLeftWarn"`         //左船告警(0无，1，超速)

	ShipRightDistance float64 `gorm:"column:shipRightDistance" json:"shipRightDistance"` //右船雷达距离
	ShipRightSpeed    float64 `gorm:"column:shipRightSpeed" json:"shipRightSpeed"`       //右船雷达速度
	ShipRightWarn     int     `gorm:"column:shipRightWarn" json:"shipRightWarn"`         //右船告警（0无，1，超速）

	SelfUpLeftState  int `gorm:"column:selfLefeUpState" json:"selfLefeUpState"`   //上左自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
	SelfUpRightState int `gorm:"column:selfRightUpState" json:"selfRightUpState"` //上右自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障

	SelfDownLeftState  int `gorm:"column:selfLefeDownState" json:"selfLefeDownState"`   //下左自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
	SelfDownRightState int `gorm:"column:selfRightDownState" json:"selfRightDownState"` //下右自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
}

func (nar *NavAnimationRecord) TableName() string {
	return "nav_animation"
}

func (nar NavAnimationRecord) IsNull(navlockid int) bool {
	return nar == (NavAnimationRecord{NavlockId: navlockid})
}

// 按照逻辑对比 注意实现浮点数的对比精度 和整型精度?
// todo: 确认整型的精度 和 浮点数的精度 不然一大堆数据很难清理
func (nar *NavAnimationRecord) CompareSame(other *NavAnimationRecord) bool {
	if nar.ScheduleId != other.ScheduleId {
		return false
	}

	if nar.NavlockStatus != other.NavlockStatus {
		return false
	}

	if nar.GateUpLeftStatus != other.GateUpLeftStatus ||
		nar.GateUpRightStatus != other.GateUpRightStatus ||
		nar.GateDownLeftStatus != other.GateDownLeftStatus ||
		nar.GateDownRightStatus != other.GateDownRightStatus {
		return false
	}

	if nar.SelfUpLeftState != other.SelfUpLeftState ||
		nar.SelfUpRightState != other.SelfUpRightState ||
		nar.SelfDownLeftState != other.SelfDownLeftState ||
		nar.SelfDownRightState != other.SelfDownRightState {

		return false
	}

	if !(nar.StoplineUpStatus == other.StoplineUpStatus &&
		nar.StoplineUpWarn == other.StoplineUpWarn &&
		nar.StoplineUpWidth == other.StoplineUpWidth &&
		cm.FloatIsEqual(nar.StoplineUpDistance, other.StoplineUpDistance, floatCompareMin) &&
		nar.StoplineDownStatus == other.StoplineDownStatus &&
		nar.StoplineDownWarn == other.StoplineDownWarn &&
		nar.StoplineDownWidth == other.StoplineDownWidth &&
		cm.FloatIsEqual(nar.StoplineDownDistance, other.StoplineDownDistance, floatCompareMin)) {
		return false
	}

	if !(nar.ShipLeftWarn == other.ShipLeftWarn &&
		cm.FloatIsEqual(nar.ShipLeftDistance, other.ShipLeftDistance, floatCompareMin) &&
		cm.FloatIsEqual(nar.ShipLeftSpeed, other.ShipLeftSpeed, floatCompareMin) &&
		nar.ShipRightWarn == other.ShipRightWarn &&
		cm.FloatIsEqual(nar.ShipRightDistance, other.ShipRightDistance, floatCompareMin) &&
		cm.FloatIsEqual(nar.ShipRightSpeed, other.ShipRightSpeed, floatCompareMin)) {

		return false
	}

	return true
}

// type NavAnimationHistory NavAnimationRecord
// 兼容数据库的创建 不直接用类型别名 需要tag标签来创建数据表
type NavAnimationHistory struct {
	Id            int       `gorm:"column:id;primarykey;auto_increment" json:"id"`
	Createdat     time.Time `gorm:"column:createdAt;index:idx_nar_navlid_sche0" json:"createdAt"`   //发生时间
	NavlockId     int       `gorm:"column:navLockId;index:idx_nar_navlid_sche0" json:"navLockId"`   //船闸名称
	ScheduleId    string    `gorm:"column:scheduleId;index:idx_nar_navlid_sche0" json:"scheduleId"` //闸次号
	NavStatus     int       `gorm:"column:navStatus" json:"navStatus"`                              // 0: "通航状态未知",1: "通航换向-上行换下行",2: "通航换向-下行换上行",3: "通航允许",4: "通航禁止",5: "通航上行",6: "通航下行"
	NavlockStatus int       `gorm:"column:navLockStatus" json:"navLockStatus"`                      //0:"闸室运行状态未知",1:"上行进闸中",2:"上行进闸完毕",3:"上行出闸中",4:"上行出闸完毕",5:"下行进闸中",6:"下行进闸完毕",7:"下行出闸中",8:"下行出闸完毕"

	GateUpLeftStatus    int `gorm:"column:gateLeftUpStatus" json:"gateLeftUpStatus"`       //左上闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateUpRightStatus   int `gorm:"column:gateRightUpStatus" json:"gateRightUpStatus"`     //右上闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateDownLeftStatus  int `gorm:"column:gateLeftDownStatus" json:"gateLeftDownStatus"`   //左下闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",
	GateDownRightStatus int `gorm:"column:gateRightDownStatus" json:"gateRightDownStatus"` //右下闸门状态0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终",

	StoplineUpStatus   int     `gorm:"column:stoplineUpStatus" json:"stoplineUpStatus"`     //禁停状态(1正常，0失联)
	StoplineUpWidth    int     `gorm:"column:stoplineUpWidth" json:"stoplineUpWidth"`       //禁停线上游宽度
	StoplineUpWarn     int     `gorm:"column:stoplineUpWarn" json:"stoplineUpWarn"`         //禁停线上游告警(0无，1，告警，2警报)
	StoplineUpDistance float64 `gorm:"column:stoplineUpDistance" json:"stoplineUpDistance"` //上游超出禁停线距离

	StoplineDownStatus   int     `gorm:"column:stoplineDownStatus" json:"stoplineDownStatus"`     //禁停状态(1正常，0失联)
	StoplineDownWidth    int     `gorm:"column:stoplineDownWidth" json:"stoplineDownWidth"`       //禁停线下游宽度
	StoplineDownWarn     int     `gorm:"column:stoplineDownWarn" json:"stoplineDownWarn"`         //禁停线下游告警(0无，1，告警，2警报)
	StoplineDownDistance float64 `gorm:"column:stoplineDownDistance" json:"stoplineDownDistance"` //下游超出禁停线距离

	ShipLeftDistance float64 `gorm:"column:shipLeftDistance" json:"shipLeftDistance"` //左船雷达距离
	ShipLeftSpeed    float64 `gorm:"column:shipLeftSpeed" json:"shipLeftSpeed"`       //左船雷达速度
	ShipLeftWarn     int     `gorm:"column:shipLeftWarn" json:"shipLeftWarn"`         //左船告警(0无，1，超速)

	ShipRightDistance float64 `gorm:"column:shipRightDistance" json:"shipRightDistance"` //右船雷达距离
	ShipRightSpeed    float64 `gorm:"column:shipRightSpeed" json:"shipRightSpeed"`       //右船雷达速度
	ShipRightWarn     int     `gorm:"column:shipRightWarn" json:"shipRightWarn"`         //右船告警（0无，1，超速）

	SelfUpLeftState  int `gorm:"column:selfLefeUpState" json:"selfLefeUpState"`   //上左自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
	SelfUpRightState int `gorm:"column:selfRightUpState" json:"selfRightUpState"` //上右自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障

	SelfDownLeftState  int `gorm:"column:selfLefeDownState" json:"selfLefeDownState"`   //下左自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
	SelfDownRightState int `gorm:"column:selfRightDownState" json:"selfRightDownState"` //下右自有设备状态0：正常；1：开关量故障；2：云台故障；3：雷达故障；4：开关量和云台故障；5：开关量和雷达故障；6：云台和雷达故障；7：全故障
}

func (NavAnimationHistory) TableName() string {
	return "nav_animation_history"
}
