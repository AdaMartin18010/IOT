package schedule

import (
	"fmt"
	"time"

	gzb_db_web_wrapper "navigate/store/iot/navlock/gezhouba/web/wrapper"
)

type TrackingRef struct {
	AnimationWrap       *gzb_db_web_wrapper.NavAnimationWrap       // 动画需要的状态封装 设置状态和存入
	SpeedSetupWrap      *gzb_db_web_wrapper.NavShipSpeedSetupWrap  // 距离和速度限制的设置封装 获取状态
	PlayScreenSetupWrap *gzb_db_web_wrapper.NavPlayScreenSetupWrap // led setup的获取状态
	StopLineStatisWrap  *gzb_db_web_wrapper.NavStopLineStatisWrap  // stopline 禁停数据的统计
	SpeedStatisWrap     *gzb_db_web_wrapper.NavSpeedStatisWrap     // speedStatis 闸次速度数据的统计
}

// navlock船闸的地理和调度概念公共结构 --------- 与实际的船闸方位名词定义对应
type TrackingSt struct {
	EnableDebug           bool                      // 是否启动debug调试日志
	NavlockName           string                    // 船闸名称
	NavlockType           int                       // 船闸类型 1 单船单行 2 双船并行
	NavlockIdentifier     int                       // 从配置文件中获取的船闸编号或者序号
	NavlockDistance       float32                   // 物理设备处于船闸的位置距离
	NavlockLength         float32                   // 船闸的长度 米
	NavlockWidth          float32                   // 船闸的宽度 米
	SensorsLayout         int                       // 传感器的物理部署类型 1: 上下游 左右单边部署 2: 上下游 左右双边部署
	DetectionDistance     float32                   // 雷达的探测 有效距离 启动雷达时 探测到的目标 距离大于该值 会被过滤掉 米
	DetectionSpeedMax     float32                   // 雷达的探测 目标最大速度 大于该值 会被过滤掉 m/s
	TargetAccelerationMax float32                   // 雷达的探测 目标临界加速度 加速度绝对值大于该值会被过滤掉 绝对值m/s
	NavlockLocation       NavlockLocation           // 驱动设备处于船闸的方位
	ScheduleId            *ScheduleIdWrapper        // 船闸-闸次ID的生成 封装 与闸次号区分
	PredicateSchedule     *PredicateScheduleWrapper // 综合判断船闸通航状态 封装

	Ref *TrackingRef
}

func NewTrackingSt(navlockid int, navStatusInterval time.Duration, enableAnimation bool) *TrackingSt {
	var AnimationWrap *gzb_db_web_wrapper.NavAnimationWrap
	AnimationWrap = nil

	if enableAnimation {
		AnimationWrap = gzb_db_web_wrapper.NewNavAnimationWrap(navlockid)
	}

	tmp := &TrackingSt{
		NavlockIdentifier: navlockid,
		ScheduleId:        NewScheduleIdWrapper(navlockid),
		Ref: &TrackingRef{
			AnimationWrap:       nil,
			SpeedSetupWrap:      gzb_db_web_wrapper.NewNavShipSpeedSetupWrap(navlockid),
			PlayScreenSetupWrap: gzb_db_web_wrapper.NewNavPlayScreenSetupWrap(navlockid),
			StopLineStatisWrap:  gzb_db_web_wrapper.NewNavStopLineStatisWrap(navlockid),
			SpeedStatisWrap:     gzb_db_web_wrapper.NewNavSpeedStatisWrap(navlockid),
		},
	}

	tmp.Ref.AnimationWrap = AnimationWrap
	tmp.PredicateSchedule = NewPredicateScheduleWrapper(navStatusInterval, tmp)
	return tmp
}

func (ts *TrackingSt) Info() string {
	ss0, ss1 := ts.PredicateSchedule.GetScheduleStatus()
	return fmt.Sprintf(`(Navlock:[%d],Location:[%s],ScduId-[%s],ScduSt:[%s,%s])`,
		ts.NavlockIdentifier,
		ts.NavlockLocation.String(),
		ts.ScheduleId.GetScheduleId(),
		ss0.String(), ss1.String())
}

// 创建新结构 并且复制公共结构
func (ts *TrackingSt) CopyNew() *TrackingSt {
	tmp := &TrackingSt{
		EnableDebug:           ts.EnableDebug,
		NavlockType:           ts.NavlockType,
		NavlockIdentifier:     ts.NavlockIdentifier,
		NavlockDistance:       ts.NavlockDistance,
		NavlockLocation:       ts.NavlockLocation,
		NavlockLength:         ts.NavlockLength,
		NavlockWidth:          ts.NavlockWidth,
		SensorsLayout:         ts.SensorsLayout,
		DetectionDistance:     ts.DetectionDistance,
		DetectionSpeedMax:     ts.DetectionSpeedMax,
		TargetAccelerationMax: ts.TargetAccelerationMax,
		ScheduleId:            ts.ScheduleId,
		PredicateSchedule:     ts.PredicateSchedule,
		Ref:                   ts.Ref,
	}

	return tmp
}

func (ts *TrackingSt) AnimationInfo() string {
	if ts.Ref.AnimationWrap != nil {
		return ts.Ref.AnimationWrap.Infos()
	} else {
		return ""
	}
}
