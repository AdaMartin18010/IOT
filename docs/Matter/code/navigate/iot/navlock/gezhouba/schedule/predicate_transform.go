package schedule

import (
	"fmt"
	"time"

	cm "navigate/common/model"
	bs "navigate/internal/bytestr"
	gzbapi_navls "navigate/iot/navlock/gezhouba/api/navlstatus"
)

// 对应到监测PLC 能采集到的有效信息
// 船闸方位的定义    ---  以自然水位上游:为其方位的上,以自然水位下游:为其方位的下,以人背对上游面向下游区分方位左右.
// 船闸的结构和机构  ---  闸门:左右开合闸门,上下游上下闸门封闭空间为闸室.
//
//	--- 信号灯安装于闸面上方,与闸门对应分上下左右,控制上下行进闸和出闸.
//	--- 上游水位  对应 自然上游水位 下游水位对应 自然下游水位; 闸室内侧水位区分,上下游封闭动态的水位.
type NavlockDynamicStatusData struct {
	Description string `json:"Desc"`
	//推断--设置的字段类别
	NavState     NavStatus `json:"NavState,string"`     //推断的通航调度状态
	NavStateLast NavStatus `json:"NavStateLast,string"` //上一次的通航调度状态

	NavlockState     NavlockStatus `json:"NavlockState,string"`     //推断的船闸调度状态
	NavlockStateLast NavlockStatus `json:"NavlockStateLast,string"` //上一次的船闸状态

	//调用方--设置的字段类别
	BeginTime          time.Time `json:"BeginTime"`                 //推断动态数据的开始时间或者第一次的时间
	EndTime            time.Time `json:"EndTime"`                   //推断动态数据的结束时间或者第二次的时间
	UpInnerWaterLast   float32   `json:"UpInnerWaterLast,string"`   //上游闸室内侧水位上次的值
	DownInnerWaterLast float32   `json:"DownInnerWaterLast,string"` //下游闸室内侧水位上次的值

	//转换函数--设置的字段类别
	TNavState         NavStatus    `json:"tNavState,string"`       //通航原始调度状态
	TUpWater          float32      `json:"tUpWater,string"`        //上游水位 --- 上游闸室外侧水位
	TUpInnerWater     float32      `json:"tUpInnerWater,string"`   //上游闸室内侧水位
	TDownWater        float32      `json:"tDownWater,string"`      //下游水位 --- 下游闸室外侧水位
	TDownInnerWater   float32      `json:"tDownInnerWater,string"` //下游闸室内侧水位
	TUpGates          Gates        `json:"tUpGates"`               //上游门
	TDownGates        Gates        `json:"tDownGates"`             //下游门
	TUpSignalLights   SignalLights `json:"tUpSignalLights"`        //上游信号灯
	TDownSignalLights SignalLights `json:"tDownSignalLights"`      //下游信号灯
}

// 重置-转换设置的值 为初始化零值
// resetInvokedFields is used to reset invoked fields 重置调用方设置的所有字段为零值
func (nds *NavlockDynamicStatusData) ReSet(resetInvokedFields bool) {
	//转换设置的值  从原始结构获取转换过来的值
	nds.TNavState = NavUnknown
	nds.TDownGates.Left.State = 0
	nds.TDownGates.Left.OpenAngle = 0.0
	nds.TDownGates.Right.State = 0
	nds.TDownGates.Right.OpenAngle = 0.0
	nds.TUpGates.Left.State = 0
	nds.TUpGates.Left.OpenAngle = 0.0
	nds.TUpGates.Right.State = 0
	nds.TUpGates.Right.OpenAngle = 0.0
	nds.TUpSignalLights.Left = 0
	nds.TUpSignalLights.Right = 0
	nds.TDownSignalLights.Left = 0
	nds.TDownSignalLights.Right = 0
	nds.TDownWater = 0.0
	nds.TUpWater = 0.0
	nds.TDownInnerWater = 0.0
	nds.TUpInnerWater = 0.0

	//推断程序设置的值 推断程序会自己再次设置
	nds.NavState = 0
	nds.NavlockState = 0

	//重置 调用方设置的字段 为零值 -- 给调用方有效设置的值
	if resetInvokedFields {
		nds.NavStateLast = 0
		nds.NavlockStateLast = 0
		nds.UpInnerWaterLast = 0.0
		nds.DownInnerWaterLast = 0.0
		nds.BeginTime = time.Time{}
		nds.EndTime = time.Time{}
	}
}

func (nds *NavlockDynamicStatusData) JsonData() ([]byte, error) {
	return cm.Json.Marshal(nds)
}

func (nds *NavlockDynamicStatusData) String() string {
	sbuf, _ := cm.Json.Marshal(nds)
	return bs.BytesToString(sbuf)
}

// 转换船闸门的状态 注意需要先重置闸门的初始值
func (nds *NavlockDynamicStatusData) TransformGates(mntif *gzbapi_navls.NavlStatusResp) error {

	//1. 验证原始数据的取值
	if mntif.SzzmKyx+mntif.SzzmGyx+mntif.SzzmKz+mntif.SzzmGz > 1 {
		return fmt.Errorf("invalid navigationlock status: %s", "上游左闸门状态非唯一")
	}

	if mntif.SzzmKyx+mntif.SzzmGyx+mntif.SzzmKz+mntif.SzzmGz == 0 {
		return fmt.Errorf("invalid navigationlock status: %s", "上游左闸门状态非确定")
	}

	if mntif.SyzmKyx+mntif.SyzmGyx+mntif.SyzmKz+mntif.SyzmGz > 1 {
		return fmt.Errorf("invalid navigationlock status: %s", "上游右闸门状态非唯一")
	}

	if mntif.SyzmKyx+mntif.SyzmGyx+mntif.SyzmKz+mntif.SyzmGz == 0 {
		return fmt.Errorf("invalid navigationlock status: %s", "上游右闸门状态非确定")
	}

	if mntif.XzzmKyx+mntif.XzzmGyx+mntif.XzzmKz+mntif.XzzmGz > 1 {
		return fmt.Errorf("invalid navigationlock status: %s", "下游左闸门状态非唯一")
	}

	if mntif.XzzmKyx+mntif.XzzmGyx+mntif.XzzmKz+mntif.XzzmGz == 0 {
		return fmt.Errorf("invalid navigationlock status: %s", "下游左闸门状态非确定")
	}

	if mntif.XyzmKyx+mntif.XyzmGyx+mntif.XyzmKz+mntif.XyzmGz > 1 {
		return fmt.Errorf("invalid navigationlock status: %s", "下游右闸门状态非唯一")
	}

	if mntif.XyzmKyx+mntif.XyzmGyx+mntif.XyzmKz+mntif.XyzmGz == 0 {
		return fmt.Errorf("invalid navigationlock status: %s", "下游右闸门状态非确定")
	}

	// 转换值到结构

	//上左右闸门
	if mntif.SzzmKyx == 1 {
		nds.TUpGates.Left.State = GateOpening
	} else if mntif.SzzmGyx == 1 {
		nds.TUpGates.Left.State = GateClosing
	} else if mntif.SzzmKz == 1 {
		nds.TUpGates.Left.State = GateOpened
	} else if mntif.SzzmGz == 1 {
		nds.TUpGates.Left.State = GateClosed
	}
	nds.TUpGates.Left.OpenAngle = mntif.SzzmKd

	if mntif.SyzmKyx == 1 {
		nds.TUpGates.Right.State = GateOpening
	} else if mntif.SyzmGyx == 1 {
		nds.TUpGates.Right.State = GateClosing
	} else if mntif.SyzmKz == 1 {
		nds.TUpGates.Right.State = GateOpened
	} else if mntif.SyzmGz == 1 {
		nds.TUpGates.Right.State = GateClosed
	}
	nds.TUpGates.Right.OpenAngle = mntif.SyzmKd

	//下左右闸门
	if mntif.XzzmKyx == 1 {
		nds.TDownGates.Left.State = GateOpening
	} else if mntif.XzzmGyx == 1 {
		nds.TDownGates.Left.State = GateClosing
	} else if mntif.XzzmKz == 1 {
		nds.TDownGates.Left.State = GateOpened
	} else if mntif.XzzmGz == 1 {
		nds.TDownGates.Left.State = GateClosed
	}
	nds.TDownGates.Left.OpenAngle = mntif.XzzmKd

	if mntif.XyzmKyx == 1 {
		nds.TDownGates.Right.State = GateOpening
	} else if mntif.XyzmGyx == 1 {
		nds.TDownGates.Right.State = GateClosing
	} else if mntif.XyzmKz == 1 {
		nds.TDownGates.Right.State = GateOpened
	} else if mntif.XyzmGz == 1 {
		nds.TDownGates.Right.State = GateClosed
	}
	nds.TDownGates.Right.OpenAngle = mntif.XyzmKd

	//3. 验证 闸门的状态是否一致 左右闸门状态实际情况都是一致的
	if nds.TUpGates.Left.State != nds.TUpGates.Right.State {
		return fmt.Errorf("invalid navigationlock status: %s", "上游左右闸门状态不一致")
	}

	if nds.TDownGates.Left.State != nds.TDownGates.Right.State {
		return fmt.Errorf("invalid navigationlock status: %s", "下游左右闸门状态不一致")
	}

	if nds.TUpGates.Left.State <= GateOpened && nds.TDownGates.Left.State <= GateOpened {
		return fmt.Errorf("invalid navigationlock status: %s", "上下游-左右闸门状态不能都为打开中或者打开")
	}

	return nil
}

// 转换船闸信号灯的状态  注意需要先重置信号灯的初始值
func (nds *NavlockDynamicStatusData) TransformLights(mntif *gzbapi_navls.NavlStatusResp) error {
	//信号灯实际情况不可能超过两个亮 有且有一个是亮的
	if mntif.SzXhd+mntif.SyXhd+mntif.XzXhd+mntif.XyXhd > 1 {
		return fmt.Errorf("invalid singallight status: %s", "信号灯状态不唯一: 有多于一个的绿灯亮")
	}

	switch mntif.SzXhd {
	case 1:
		nds.TUpSignalLights.Left = GreenLightOn
	case 0:
		nds.TUpSignalLights.Left = GreenLightOff
	}

	switch mntif.SyXhd {
	case 1:
		nds.TUpSignalLights.Right = GreenLightOn
	case 0:
		nds.TUpSignalLights.Right = GreenLightOff
	}

	switch mntif.XzXhd {
	case 1:
		nds.TDownSignalLights.Left = GreenLightOn
	case 0:
		nds.TDownSignalLights.Left = GreenLightOff
	}

	switch mntif.XyXhd {
	case 1:
		nds.TDownSignalLights.Right = GreenLightOn
	case 0:
		nds.TDownSignalLights.Right = GreenLightOff
	}

	return nil
}

// 转换船闸的状态 注意需要先重置初始值
func (nds *NavlockDynamicStatusData) TransformFrom(mntif *gzbapi_navls.NavlStatusResp) error {
	//使用协议中的时间同步 如果为零值 则使用当前时间
	nds.BeginTime = (time.Time)(mntif.Time)
	if (nds.BeginTime == time.Time{}) || (nds.BeginTime.IsZero()) {
		nds.BeginTime = time.Now()
	}

	nds.TUpWater = mntif.SySw
	nds.TDownWater = mntif.XySw
	nds.TUpInnerWater = mntif.ZssySw
	nds.TDownInnerWater = mntif.ZsxySw

	// if nds.DownInnerWaterLasttime == 0.0 {
	// 	nds.DownInnerWaterLasttime = nds.DownInnerWater
	// }

	// if nds.UpInnerWaterLasttime == 0.0 {
	// 	nds.UpInnerWaterLasttime = nds.UpInnerWater
	// }

	if err := nds.TransformGates(mntif); err != nil {
		return err
	}

	if err := nds.TransformLights(mntif); err != nil {
		return err
	}

	nds.TNavState = NavUnknown
	// todo : 葛洲坝通航状态 咨询后结论是本身转换逻辑存在问题 不能作为依据 在做船闸判断时会重置该字段
	if mntif.ThZt == 1 {
		if mntif.SxZt == 1 {
			nds.TNavState = NavUp
		}
		if mntif.XxZt == 1 {
			nds.TNavState = NavDown
		}
	}
	// todo : 做船闸判断时 是否会重置该值 需要验证整个的船闸状态存在 与正常通航下船闸状态重合
	if mntif.ThZt == 0 && mntif.SxZt == 0 && mntif.XxZt == 0 {
		nds.TNavState = NavForbidden
	}

	return nil
}
