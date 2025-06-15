package schedule

const (
	PredicateMinIntervalSecond = 5    //推断有效--最小时间间隔 秒钟
	PredicateMaxIntervalMinute = 5    //推断有效--最大时间间隔 分钟
	WaterTrendUp               = 0.1  //水位上涨趋势判断值 米
	WaterTrendDown             = -0.1 //水位下降趋势判断值 米
)

// 通航状态和实际的运营相关
// 系统能根据上层接口状态判断出来的通航状态
var (
	//无状态与零值对应-0
	NavUnknown        NavStatus = 0 //初始化或者获取出现故障时使用
	NavSwitchUpToDown NavStatus = 1 //通航切换 上行切换到下行 无进闸绿灯
	NavSwitchDownToUp NavStatus = 2 //通航切换 下行切换到上行 无进闸绿灯
	NavAllow          NavStatus = 3 //在获取不到 通航--上下行的时候使用该值
	NavForbidden      NavStatus = 4 //停航 全红灯 或者通航标志设置的情况下
	NavUp             NavStatus = 5 //上行 人工开启上行进闸绿灯
	NavDown           NavStatus = 6 //下行 人工开启下行进闸绿灯
	//通航状态对照map
	NavStatusTypeName = map[NavStatus]string{
		0: "NavigationStatusUnknown",
		1: "NavigationSwitch-UpToDown",
		2: "NavigationSwitch-DownToUp",
		3: "NavigationAllow",
		4: "NavigationForbidden",
		5: "NavigationUp",
		6: "NavigationDown",
	}

	//通航状态中文对照map
	NavStatusTypeNameCN = map[NavStatus]string{
		0: "通航状态未知",
		1: "通航换向-上行换下行",
		2: "通航换向-下行换上行",
		3: "通航允许",
		4: "通航禁止",
		5: "通航上行",
		6: "通航下行",
	}
)

// 通航状态: 与运营调度系统相关 见以上定义
// NavigateOK 允许通航 NavigateForbidden 通航禁止
// NavigationStatusUnknown 通航状态不可知
// NavigateUp 上行通航 NavigateDown 下行通航
type NavStatus uint

func (n NavStatus) String() string {
	switch n {
	case NavSwitchUpToDown:
		return NavStatusTypeNameCN[NavSwitchUpToDown]
	case NavSwitchDownToUp:
		return NavStatusTypeNameCN[NavSwitchDownToUp]
	case NavAllow:
		return NavStatusTypeNameCN[NavAllow]
	case NavForbidden:
		return NavStatusTypeNameCN[NavForbidden]
	case NavUp:
		return NavStatusTypeNameCN[NavUp]
	case NavDown:
		return NavStatusTypeNameCN[NavDown]
	default:
		return NavStatusTypeNameCN[NavUnknown]
	}
}

// 信号灯:上右为下行进闸信号，下右为下行出闸信号，上左为上行出闸信号，下左为上行进闸信号
// 闸门绿灯开启事件:
// 上行:上行进闸-下游左绿灯开启,上行出闸-上游左绿灯开启
// 下行:下行进闸-上游右绿灯开启,下行出闸-下游右绿灯开启
const (
	//无状态与零值对应-0
	NavlockUnknown NavlockStatus = 0

	// 事件:上行进闸---下游闸门打开中或者打开 && 下游左绿灯开,持续过程:该闸次的船舶进入闸室-下游闸门关闭中或者关闭
	NavlockUpGoingIn NavlockStatus = 1
	// 事件:上行进闸完成---下游闸门关闭中或者关闭 || 下游左绿灯开,持续过程:该闸次的所有船舶进入闸室完毕-等待水位上升到出闸-上游闸门 打开中 之前
	NavlockUpGoInDone NavlockStatus = 2
	// 事件:全部红灯 水位上涨
	NavlockUpWaterTrendUp NavlockStatus = 3
	// 事件:上行出闸--- 上游闸门打开中或者打开 && 上游左绿灯开,持续过程:该闸次的所有船舶驶出闸室完毕的过程-- 上游左绿灯关 不判定上游闸门关闭
	NavlockUpGoingOut NavlockStatus = 4
	// 事件:上行出闸完成--- 上游左绿灯关|| 上游右绿灯开 || 上游闸门关闭中或者关闭 || 船舶出闸完毕等待到一定时间, 持续过程:直到有事件判断新通航闸次的开始
	NavlockUpGoOutDone NavlockStatus = 5
	// ******************* 上下行切换 ***************
	// 事件:下行进闸---上游闸门打开中或者打开 && 上游右绿灯开,持续过程:该闸次的船舶进入闸室-上游闸门关闭中或者关闭
	NavlockDownGoingIn NavlockStatus = 6
	// 事件:下行进闸完成---上游闸门关闭中或者关闭|| 上游右绿灯开,持续过程:该闸次的所有船舶进入闸室完毕-等待水位下降到出闸-下游闸门打开之前
	NavlockDownGoInDone NavlockStatus = 7
	// 事件: 全部红灯 绿灯关 水位下降
	NavlockDownWaterTrendDown NavlockStatus = 8
	// 事件:下行出闸---下游闸门打开中或者打开 && 下游右绿灯开,持续过程: 该闸次的所有船舶驶出闸室完毕的过程--下游右绿灯关 不判定下游闸门关闭
	NavlockDownGoingOut NavlockStatus = 9
	// 事件:下行出闸完成---下游右绿灯关|| 下游左绿灯开 || 下游闸门关闭中或者关闭 || 船舶出闸完毕等待到一定时间, 持续过程:直到有事件判断新通航闸次的开始
	NavlockDownGoOutDone NavlockStatus = 10
)

var (
	//闸室状态对照map
	NavlockStatusTypeName = map[NavlockStatus]string{
		0:  "NavlockStatusUNKNOWN",
		1:  "NavlockUpGoingIn",
		2:  "NavlockUpGoInDone",
		3:  "NavlockUpWaterTrendUp",
		4:  "NavlockUpGoingOut",
		5:  "NavlockUpGoOutDone",
		6:  "NavlockDownGoingIn",
		7:  "NavlockDownGoInDone",
		8:  "NavlockDownWaterTrendDown",
		9:  "NavlockDownGoingOut",
		10: "NavlockDownGoOutDone",
	}

	// 闸室状态中文
	NavlockStatusTypeNameCN = map[NavlockStatus]string{
		0:  "闸室运行状态未知", //闸室运行状态未知
		1:  "上行-进闸中",
		2:  "上行-进闸完毕",
		3:  "上行-闸室-升水位",
		4:  "上行-出闸中",
		5:  "上行-出闸完毕",
		6:  "下行-进闸中",
		7:  "下行-进闸完毕",
		8:  "下行-闸室-降水位",
		9:  "下行-出闸中",
		10: "下行-出闸完毕",
	}
)

// 闸室的状态: 与闸室的运行状态相关
type NavlockStatus uint

// web表的定义  0为未知
// '0: "闸室运行状态未知",1: "上行进闸中",2: "上行进闸完毕",3: "上行出闸中",4: "上行出闸完毕",5: "下行进闸中",6: "下行进闸完毕",7: "下行出闸中",8: "下行出闸完毕"',
func (s NavlockStatus) TransformToWebDefine() int {
	switch s {
	case NavlockUpGoingIn:
		return 1
	case NavlockUpGoInDone, NavlockUpWaterTrendUp:
		return 2
	case NavlockUpGoingOut:
		return 3
	case NavlockUpGoOutDone:
		return 4
	case NavlockDownGoingIn:
		return 5
	case NavlockDownGoInDone, NavlockDownWaterTrendDown:
		return 6
	case NavlockDownGoingOut:
		return 7
	case NavlockDownGoOutDone:
		return 8
	default:
		return 0
	}
}

func (s NavlockStatus) String() string {
	switch s {
	case NavlockUpGoingIn:
		return NavlockStatusTypeNameCN[NavlockUpGoingIn]
	case NavlockUpGoInDone:
		return NavlockStatusTypeNameCN[NavlockUpGoInDone]
	case NavlockUpWaterTrendUp:
		return NavlockStatusTypeNameCN[NavlockUpWaterTrendUp]
	case NavlockUpGoingOut:
		return NavlockStatusTypeNameCN[NavlockUpGoingOut]
	case NavlockUpGoOutDone:
		return NavlockStatusTypeNameCN[NavlockUpGoOutDone]
	case NavlockDownGoingIn:
		return NavlockStatusTypeNameCN[NavlockDownGoingIn]
	case NavlockDownGoInDone:
		return NavlockStatusTypeNameCN[NavlockDownGoInDone]
	case NavlockDownWaterTrendDown:
		return NavlockStatusTypeNameCN[NavlockDownWaterTrendDown]
	case NavlockDownGoingOut:
		return NavlockStatusTypeNameCN[NavlockDownGoingOut]
	case NavlockDownGoOutDone:
		return NavlockStatusTypeNameCN[NavlockDownGoOutDone]
	default:
		return NavlockStatusTypeNameCN[0]
	}
}

const (
	NavlockLocUp        NavlockLocation = 1
	NavlockLocUpLeft    NavlockLocation = 2
	NavlockLocUpRight   NavlockLocation = 3
	NavlockLocDown      NavlockLocation = 4
	NavlockLocDownLeft  NavlockLocation = 5
	NavlockLocDownRight NavlockLocation = 6
)

var (
	// 闸室位置中文
	NavlockLocationTypeNameCN = map[NavlockLocation]string{
		0: "方位未知",
		1: "上游",
		2: "上游左侧",
		3: "上游右侧",
		4: "下游",
		5: "下游左侧",
		6: "下游右侧",
	}
)

// 船闸地方的方位
type NavlockLocation uint

func (nll *NavlockLocation) String() string {
	switch *nll {
	case NavlockLocUp:
		return NavlockLocationTypeNameCN[NavlockLocUp]
	case NavlockLocUpLeft:
		return NavlockLocationTypeNameCN[NavlockLocUpLeft]
	case NavlockLocUpRight:
		return NavlockLocationTypeNameCN[NavlockLocUpRight]
	case NavlockLocDown:
		return NavlockLocationTypeNameCN[NavlockLocDown]
	case NavlockLocDownLeft:
		return NavlockLocationTypeNameCN[NavlockLocDownLeft]
	case NavlockLocDownRight:
		return NavlockLocationTypeNameCN[NavlockLocDownRight]
	default:
		return NavlockLocationTypeNameCN[0]
	}
}

// 船闸门状态:
// GateOpened 闸门开终 GateOpening 闸门开运行 GateClosed 闸门关终 GateCloseing 闸门关运行
var (
	//无状态与零值对应-0
	GateOpening GateStatus = 1
	GateOpened  GateStatus = 2
	GateClosing GateStatus = 3
	GateClosed  GateStatus = 4

	//闸室状态对照map
	GateStatusTypeName = map[GateStatus]string{
		0: "GateStatusUNKNOWN",
		1: "GateOpening",
		2: "GateOpened",
		3: "GateClosing",
		4: "GateClosed",
	}

	//闸室状态中文对照map
	GateStatusTypeNameCN = map[GateStatus]string{
		0: "闸门状态未知",
		1: "闸门开运行",
		2: "闸门开终",
		3: "闸门关运行",
		4: "闸门关终",
	}
)

type GateStatus uint

func (s GateStatus) String() string {
	switch s {
	case GateOpening:
		return GateStatusTypeNameCN[GateOpening]
	case GateOpened:
		return GateStatusTypeNameCN[GateOpened]
	case GateClosing:
		return GateStatusTypeNameCN[GateClosing]
	case GateClosed:
		return GateStatusTypeNameCN[GateClosed]
	default:
		return GateStatusTypeNameCN[0]
	}
}

type Gate struct {
	State GateStatus `json:"State"`
	//门的开度
	OpenAngle float32 `json:"OpenAngle,string"`
}

type Gates struct {
	Left  Gate `json:"Left"`
	Right Gate `json:"Right"`
}

func (gs *Gates) IsAllEqualState(gateState GateStatus) bool {
	if (gs.Left.State == gateState) && (gs.Right.State == gateState) {
		return true
	}
	return false
}

var (
	LightStatusUnknown SignalLightStatus = 0
	GreenLightOn       SignalLightStatus = 1
	GreenLightOff      SignalLightStatus = 2

	//信号灯状态对照map
	SignalLightStatusTypeName = map[SignalLightStatus]string{
		0: "LigntStatusUnknown",
		1: "GreenLigntOn",
		2: "GreenLightOff",
	}

	//信号灯状态中文对照map
	SignalLightStatusTypeNameCN = map[SignalLightStatus]string{
		0: "信号灯状态未知",
		1: "绿灯开启",
		2: "绿灯关闭",
	}
)

// signal light
// Green 绿灯开启
type SignalLightStatus uint

func (s SignalLightStatus) String() string {
	switch s {
	case GreenLightOn:
		return SignalLightStatusTypeNameCN[GreenLightOn]
	case GreenLightOff:
		return SignalLightStatusTypeNameCN[GreenLightOff]
	default:
		return SignalLightStatusTypeNameCN[0]
	}
}

type SignalLights struct {
	Left  SignalLightStatus `json:"Left"`
	Right SignalLightStatus `json:"Right"`
}

func (sls *SignalLights) IsAllEqualState(lightState SignalLightStatus) bool {
	if sls.Left == lightState && sls.Right == lightState {
		return true
	}

	return false
}
