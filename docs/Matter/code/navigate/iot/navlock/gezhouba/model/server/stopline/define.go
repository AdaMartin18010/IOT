package stopline

const (
	StopLinePlcStatusUnknown StoplinePlcStatus = 0 //禁停线PLC状态未知
	StopLinePlcStatusNormal  StoplinePlcStatus = 1 //禁停线PLC状态正常
	StopLinePlcStatusError   StoplinePlcStatus = 2 //禁停线PLC状态出错 禁停线PLC本身的系统状态出错
)

var (

	//禁停系统的状态对照map
	StopLinePlcStatusTypeName = map[StoplinePlcStatus]string{
		0: "StopLinePlcStatusUnknown",
		1: "StopLinePlcStatusNormal",
		2: "StopLinePlcStatusError",
	}

	//禁停系统的状态中文对照map
	StopLinePlcStatusTypeNameCN = map[StoplinePlcStatus]string{
		0: "禁停线PLC状态未知",
		1: "禁停线PLC状态正常",
		2: "禁停线PLC状态错误",
	}
)

// 与禁停线系统的状态判断相关
type StoplinePlcStatus int

func (s StoplinePlcStatus) String() string {
	switch s {
	case StopLinePlcStatusNormal:
		return "StopLinePlcStatusNormal"
	case StopLinePlcStatusError:
		return "StopLinePlcStatusError"
	default:
		return "StopLinePlcStatusUnknown"
	}
}

var (
	TagUnknown PlcTag = 0 //禁停线标签-未知
	TagNormal  PlcTag = 1 //禁停线标签-正常
	TagAlarm   PlcTag = 2 //禁停线标签-越线报警
	TagWarning PlcTag = 3 //禁停线标签-越线警告

	//禁停线标签对照map
	PlcTagTypeName = map[PlcTag]string{
		0: "TagUnknown",
		1: "TagNormal",
		2: "TagAlarm",
		3: "TagWarning",
	}

	//禁停线标签中文对照map
	PlcTagTypeNameCN = map[PlcTag]string{
		0: "禁停线标签-未知",
		1: "禁停线标签-正常",
		2: "禁停线标签-越线报警",
		3: "禁停线标签-越线警告",
	}

	CatchTargetUnknown CatchTarget = 0 //禁停线监测功能 未知
	CatchTargetNormal  CatchTarget = 1 //禁停线监测功能 正常
	TargetArea00000    CatchTarget = 2 //禁停线监测目标在禁停区域
	TargetArea10000    CatchTarget = 3 //禁停线监测目标在禁停区域-2
	TargetArea01000    CatchTarget = 4 //禁停线监测目标在禁停区域-1
	TargetArea00010    CatchTarget = 5 //禁停线监测目标在禁停区域+1
	TargetArea00001    CatchTarget = 6 //禁停线监测目标在禁停区域+2

	//探测目标状态对照map
	CatchTargetTypeName = map[CatchTarget]string{
		0: "CatchTargetUnknown",
		1: "CatchTargetNormal",
		2: "TargetArea-00000",
		3: "TargetArea-10000",
		4: "TargetArea-01000",
		5: "TargetArea-00010",
		6: "TargetArea-00001",
	}

	//探测目标状态中文对照map
	CatchTargetTypeNameCN = map[CatchTarget]string{
		0: "禁停线探测-未知",
		1: "禁停线探测-正常",
		2: "禁停线探测-目标在禁停区域",
		3: "禁停线探测-目标在禁停区域-2",
		4: "禁停线探测-目标在禁停区域-1",
		5: "禁停线探测-目标在禁停区域+1",
		6: "禁停线探测-目标在禁停区域+2",
	}
)

type PlcTag int

func (plc PlcTag) String() string {
	switch plc {
	case TagNormal:
		return "TagNormal"
	case TagAlarm:
		return "TagAlarm"
	case TagWarning:
		return "TagWarning"
	default:
		return "TagUnknown"
	}
}

type CatchTarget int

func (c CatchTarget) String() string {
	switch c {
	case CatchTargetNormal:
		return "CatchTargetNormal"
	case TargetArea00000:
		return "TargetArea-00000"
	case TargetArea10000:
		return "TargetArea-10000"
	case TargetArea01000:
		return "TargetArea-01000"
	case TargetArea00010:
		return "TargetArea-00010"
	case TargetArea00001:
		return "TargetArea-00001"
	default:
		return "CatchTargetUnknown"
	}

}

// 对应到禁停系统  能采集到的有效信息
type StopLinePlcDynamicData struct {
}
