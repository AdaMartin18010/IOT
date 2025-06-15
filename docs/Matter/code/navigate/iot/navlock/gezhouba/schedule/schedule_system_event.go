package schedule

import "time"

// *********************************************************************************
// 船闸系统调度的基本系统 事件概念范畴定义 程序设置调度系统事件
// 1.通航 --- 分船闸倒向 上下行切换 和正常的运营即实际有船通航 --- 需要连续并持续的信息判断
// 2.船闸状态  --- 只和当前的获取的船闸状态值有关系 与实际场景运营层面的通不通船无关
// 船闸状态的判断 --- 从信号灯的角度来判断的 需要针对不同情况判断
// 3.船闸系统调度事件的产生  --- 只有符合通航和船闸状态判断条件的情况下 才触发系统事件
// 4.船闸系统调度事件 --- 是系统产生控制和行为的状态判断
// 5.闸次 --- 是系统产生用来标识通航通船的所有有效数据的标识
// 包含实际有通船的完整的一个上下行进闸到出闸完成的过程
// ******************************
// todo: 调度系统事件 包括雷达测速 测到有效的数据 距离船闸的基本距离 速度阈值等运营层面的事件
// todo: 其他的网络联通性 和 请求超时 获取数据不正常这些网络和程序设计层面的事件由另外事件来定义
// **********************************************************************************

type MeasureEventType uint

const (
	EventUnknown      MeasureEventType = 0 //测速事件未知
	NotConnected      MeasureEventType = 1 //测速设备未连接
	NoTargetInArea    MeasureEventType = 2 //测速区域内无目标
	TargetMoveInArea  MeasureEventType = 3 //测速区域内有目标驶入
	TargetMoveOutArea MeasureEventType = 4 //测速区域内有目标驶出
)

type ScheduleMeasureEvent struct {
	From       string           `json:"From"`              //事件来源
	EvType     MeasureEventType `json:"EvType,string"`     //事件类型
	EvTypeLast MeasureEventType `json:"EvTypeLast,string"` //上次事件类型
	BeginTime  time.Time        `json:"BeginTime"`         //事件开始时间
	EndTime    time.Time        `json:"EndTime"`           //事件截止时间
}
