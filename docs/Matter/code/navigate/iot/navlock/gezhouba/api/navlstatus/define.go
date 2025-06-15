package monitor

import (
	"fmt"
	"time"
)

// get 方法
// 当前解释:
//
//			1.通航标志 代表调度 比如: 调度是上行
//		    2.信号灯标志 代表正在完成或者完成了一次调度 比如: 信号灯上左绿灯 上行出闸
//			3.闸门状态代表了 当前闸室的运行情况 比如: 上下左右4闸门关终
//	     从而可能 通航标志 是通航上行 ---- 运营上调度情况; 信号灯是上行出闸 ---- 完成的一次上行船闸调度
//	     闸门状态都是关终 关闭的状态. 船闸的状态正在以通航的调度来运行,信号灯则代表了上次完成的调度状态.
//
// 信号灯：上右为下行进闸信号，下右为下行出闸信号，上左为上行出闸信号，下左为上行进闸信号
// 上行进闸:下左信号灯绿灯开启 上行出闸:上左信号灯绿灯开启
// 下行进闸:上右信号灯绿灯开启 下行出闸:下右信号的绿灯开启

type JsonTime time.Time

//const timeformat = "2006-01-02 15:04:05"

func (m *JsonTime) UnmarshalJSON(data []byte) error {
	if len(data) <= 2 {
		return nil
	}

	// Ignore null, like in the main JSON package.
	if string(data) == "null" || string(data) == `""` {
		return nil
	}

	//return cm.Json.Unmarshal(data, (*time.Time)(m))
	// // Fractional seconds are handled implicitly by Parse.
	//tt, err := time.Parse(time.DateTime, string(data))
	tt, err := time.ParseInLocation(time.DateTime, string(data[1:len(data)-1]), time.Local)
	*m = JsonTime(tt)
	return err
}

type NavlStatusResp struct {
	Time JsonTime `json:"time,omitempty"`
	//通航调度的标识
	ThZt uint `json:"TH_ZT,string"` //通航标志 Nav_OK  		   1代表通航 0代表禁航       	    ------ 整个通航过程中都有效
	SxZt uint `json:"SX_ZT,string"` //上行状态标志 Ship_UP 		1代表上行有效 0代表上行无效	      ------ 整个通航过程中都有效
	XxZt uint `json:"XX_ZT,string"` //下行状态标志 Ship_Down    1代表下行有效 0代表下行无效       ------ 整个通航过程中都有效
	//闸室的调度的标识
	SzXhd uint `json:"SZ_XHD,string"` //上左信号灯绿灯开启 取反值后 1代表开启 0代表关闭  ------ 整个通航过程中都有效
	SyXhd uint `json:"SY_XHD,string"` //上右信号灯绿灯开启 取反值后 1代表开启 0代表关闭  ------ 整个通航过程中都有效
	XzXhd uint `json:"XZ_XHD,string"` //下左信号灯绿灯开启 取反值后 1代表开启 0代表关闭  ------ 整个通航过程中都有效
	XyXhd uint `json:"XY_XHD,string"` //下右信号灯绿灯开启 取反值后 1代表开启 0代表关闭  ------ 整个通航过程中都有效
	//闸门状态  上左右2门
	SzzmKyx uint    `json:"SZZM_KYX,string"` //上左闸门开运行
	SzzmGyx uint    `json:"SZZM_GYX,string"` //上左闸门关运行
	SzzmKz  uint    `json:"SZZM_KZ,string"`  //上左闸门开终
	SzzmGz  uint    `json:"SZZM_GZ,string"`  //上左闸门关终
	SzzmKd  float32 `json:"SZZM_KD,string"`  //上左闸门开度 实际值 角度

	SyzmKyx uint    `json:"SYZM_KYX,string"` //上右闸门开运行
	SyzmGyx uint    `json:"SYZM_GYX,string"` //上右闸门关运行
	SyzmKz  uint    `json:"SYZM_KZ,string"`  //上右闸门开终
	SyzmGz  uint    `json:"SYZM_GZ,string"`  //上右闸门关终
	SyzmKd  float32 `json:"SYZM_KD,string"`  //上右闸门开度 实际值 角度
	//闸门状态 下左右2门
	XzzmKyx uint    `json:"XZZM_KYX,string"` //下左闸门开运行
	XzzmGyx uint    `json:"XZZM_GYX,string"` //下左闸门关运行
	XzzmKz  uint    `json:"XZZM_KZ,string"`  //下左闸门开终
	XzzmGz  uint    `json:"XZZM_GZ,string"`  //下左闸门关终
	XzzmKd  float32 `json:"XZZM_KD,string"`  //下左闸门开度 实际值 角度

	XyzmKyx uint    `json:"XYZM_KYX,string"` //下右闸门开运行
	XyzmGyx uint    `json:"XYZM_GYX,string"` //下右闸门关运行
	XyzmKz  uint    `json:"XYZM_KZ,string"`  //下右闸门开终
	XyzmGz  uint    `json:"XYZM_GZ,string"`  //下右闸门关终
	XyzmKd  float32 `json:"XYZM_KD,string"`  //下右闸门开度 实际值 角度

	//闸室上下游水位 闸室内侧上下游水位
	SySw   float32 `json:"SY_SW,string"`   //上游水位 实际值 米
	XySw   float32 `json:"XY_SW,string"`   //下游水位 实际值 米
	ZssySw float32 `json:"ZSSY_SW,string"` //闸室上游内侧水位 实际值 米
	ZsxySw float32 `json:"ZSXY_SW,string"` //闸室下游内侧水位 实际值 米
}

func convert01(i uint) uint {

	if i == 0 {
		return 1
	}

	if i == 1 {
		return 0
	}

	return i
}

func (m *NavlStatusResp) TransformValue() {
	m.SzXhd = convert01(m.SzXhd)
	m.SyXhd = convert01(m.SyXhd)
	m.XyXhd = convert01(m.XyXhd)
	m.XzXhd = convert01(m.XzXhd)
}

// 转换后再调用判断 是否需要怎么强的条件 需要再判断
func (m *NavlStatusResp) Valid() error {
	if m.SzXhd == 0 && m.SyXhd == 0 && m.XzXhd == 0 && m.XyXhd == 0 {
		return fmt.Errorf("signal light is all 0")
	}

	if m.SzXhd == 1 && m.SyXhd == 1 && m.XzXhd == 1 && m.XyXhd == 1 {
		return fmt.Errorf("signal light is all 1")
	}

	if (m.SzXhd + m.SyXhd + m.XzXhd + m.XyXhd) > 1 {
		return fmt.Errorf("one signal light is 1  but other signal light is also 1")
	}

	return nil
}

type XmlWrapper struct {
	String string `xml:"string"`
	Xmlns  string `xml:"xmlns,attr"`
}
