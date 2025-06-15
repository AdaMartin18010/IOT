package stopline

import "bytes"

type StoplineStruct struct {
	PlcCommErr      bool //船闸运行信息检测PLC通讯故障
	LDCommErr       bool //雷达处理计算机通讯故障
	DectectedWarn   bool //越线报警
	DectectedAlarm  bool //越线警告
	LDJSJErr        bool //雷达计算机故障
	LDErr           bool //激光雷达故障
	JTXZone_N2      bool //目标在禁停线-2区域-绿色
	JTXZone_N1      bool //目标在禁停线-1区域-绿色
	JTXZone         bool //目标在禁停线区域-橙色
	JTXZone_P1      bool //目标在禁停线+1区域-红色
	JTXZone_P2      bool //目标在禁停线+2区域-红色
	Cross_Dis       int  //越线距离 单位cm  正负值
	JTX_Zone1_Width int  //禁停线绿色区域的宽度(cm)
}

func intToBoolean(v int) bool {
	switch v {
	case 0:
		return false
	case 1:
		return true
	default:
		return true
	}
}

func (sl *StoplineStruct) TransformFrom(rss *ResponseJsonStruct) {
	sl.PlcCommErr = intToBoolean(rss.JKPLCCommErr.Result)
	sl.LDCommErr = intToBoolean(rss.LDJSJCommErr.Result)
	sl.DectectedWarn = intToBoolean(rss.DectectedWarm.Result)
	sl.DectectedAlarm = intToBoolean(rss.DectectedAlarm.Result)
	sl.LDJSJErr = intToBoolean(rss.LDJSJCommErr.Result)
	sl.LDErr = intToBoolean(rss.Err.Result)
	sl.JTXZone_N2 = intToBoolean(rss.JTXZoneN2.Result)
	sl.JTXZone_N1 = intToBoolean(rss.JTXZoneN1.Result)
	sl.JTXZone = intToBoolean(rss.JTXZone.Result)
	sl.JTXZone_P1 = intToBoolean(rss.JTXZoneP1.Result)
	sl.JTXZone_P2 = intToBoolean(rss.JTXZoneP2.Result)
	sl.Cross_Dis = rss.CrossDis.Result
	sl.JTX_Zone1_Width = rss.JTXZone1Width.Result
}

// get 方法
type Params struct {
}

type CommonStructType struct {
	SearchValue   interface{} `json:"searchValue,omitempty"`
	CreateBy      string      `json:"createBy,omitempty"`
	CreateTime    interface{} `json:"createTime,omitempty"`
	UpdateBy      string      `json:"updateBy,omitempty"`
	UpdateTime    interface{} `json:"updateTime,omitempty"`
	Remark        string      `json:"remark,omitempty"`
	Params        Params      `json:"params,omitempty"`
	ID            int         `json:"id,omitempty"`
	Code          string      `json:"code,omitempty"`
	Zs            interface{} `json:"zs,omitempty"`
	ModbusAddress string      `json:"modbusAddress,omitempty"`
	PlcCode       interface{} `json:"plcCode,omitempty"`
	DataType      string      `json:"dataType,omitempty"`
	PlcAddress    interface{} `json:"plcAddress,omitempty"`
	Command       interface{} `json:"command,omitempty"`
	Result        int         `json:"result,omitempty"`
	Type          string      `json:"type,omitempty"`
	Index         string      `json:"index,omitempty"`
	DelFlag       string      `json:"delFlag,omitempty"`
}

type Heart = CommonStructType
type Xx = CommonStructType
type F4OpenCTRL = CommonStructType
type DecterDisZB = CommonStructType
type F4CloseCTRL = CommonStructType
type R3Opened = CommonStructType
type R4Opened = CommonStructType
type SzsSw = CommonStructType
type R4Kd = CommonStructType
type F1CloseCTRL = CommonStructType
type CrossDis = CommonStructType
type R2CloseCTRL = CommonStructType
type JTXWidth = CommonStructType
type AimDetected = CommonStructType
type JTXZone1Width = CommonStructType
type XxCz = CommonStructType
type XzsSw = CommonStructType
type R2OpenCTRL = CommonStructType
type F1Closed = CommonStructType
type JtxXy = CommonStructType
type LDPowerON = CommonStructType
type JTXZoneP1 = CommonStructType
type BJQNightMode = CommonStructType
type R1CloseCTRL = CommonStructType
type F3Closed = CommonStructType
type JTXZoneP2 = CommonStructType
type StopNav = CommonStructType
type JTXZone2Width = CommonStructType
type SxCz = CommonStructType
type R3Kd = CommonStructType
type LDJSJCommErr = CommonStructType
type F1OpenCTRL = CommonStructType
type JTXZone = CommonStructType
type ZPDDCtrl = CommonStructType
type ZPMotorSpeed = CommonStructType
type R1Opened = CommonStructType
type F3CloseCTRL = CommonStructType
type F2Opened = CommonStructType
type F4Opened = CommonStructType
type AutoDetect = CommonStructType
type DectectedAlarm = CommonStructType
type DetectSW = CommonStructType
type DetectStart = CommonStructType
type R3OpenCTRL = CommonStructType
type ReserveI7 = CommonStructType
type ReserveI6 = CommonStructType
type R1Closed = CommonStructType
type R2Kd = CommonStructType
type ReserveI8 = CommonStructType
type Err = CommonStructType
type R3CloseCTRL = CommonStructType
type ZSWidth = CommonStructType
type F2OpenCTRL = CommonStructType
type ManualTestB = CommonStructType
type SxJz = CommonStructType
type PICCatch = CommonStructType
type DetectDataOffset = CommonStructType
type DetectHighLmt = CommonStructType
type R3Closed = CommonStructType
type ReserveI1 = CommonStructType
type R2Closed = CommonStructType
type R4Closed = CommonStructType
type F2CloseCTRL = CommonStructType
type ReserveI3 = CommonStructType
type ReserveI2 = CommonStructType
type Calibrated = CommonStructType
type ReserveI5 = CommonStructType
type ReserveI4 = CommonStructType
type R4OpenCTRL = CommonStructType
type R1Kd = CommonStructType
type F4Closed = CommonStructType
type XySw = CommonStructType
type JTXZoneN2 = CommonStructType
type F2Closed = CommonStructType
type F3OpenCTRL = CommonStructType
type Sx = CommonStructType
type SySw = CommonStructType
type JTXZoneN1 = CommonStructType
type DetectDisWater = CommonStructType
type R4CloseCTRL = CommonStructType
type JKPLCHeart = CommonStructType
type JTXRight = CommonStructType
type XxJz = CommonStructType
type LdZ = CommonStructType
type CalibrateReq = CommonStructType
type F3Opened = CommonStructType
type LdY = CommonStructType
type LdX = CommonStructType
type JKPLCCommErr = CommonStructType
type R1OpenCTRL = CommonStructType
type DectectedWarm = CommonStructType
type F1Opened = CommonStructType
type BJQTestB = CommonStructType
type R2Opened = CommonStructType
type BJQAlarmAllow = CommonStructType
type ZMHigh = CommonStructType

type ResponseJsonStruct struct {
	Heart            Heart            `json:"Heart,omitempty"`
	Xx               Xx               `json:"XX,omitempty"`
	F4OpenCTRL       F4OpenCTRL       `json:"F4_Open_CTRL,omitempty"`
	DecterDisZB      DecterDisZB      `json:"Decter_Dis_ZB,omitempty"`
	F4CloseCTRL      F4CloseCTRL      `json:"F4_Close_CTRL,omitempty"`
	R3Opened         R3Opened         `json:"R3_Opened,omitempty"`
	R4Opened         R4Opened         `json:"R4_Opened,omitempty"`
	SzsSw            SzsSw            `json:"SZS_SW,omitempty"`
	R4Kd             R4Kd             `json:"R4_KD,omitempty"`
	F1CloseCTRL      F1CloseCTRL      `json:"F1_Close_CTRL,omitempty"`
	CrossDis         CrossDis         `json:"Cross_Dis,omitempty"`
	R2CloseCTRL      R2CloseCTRL      `json:"R2_Close_CTRL,omitempty"`
	JTXWidth         JTXWidth         `json:"JTX_Width,omitempty"`
	AimDetected      AimDetected      `json:"Aim_Detected,omitempty"`
	JTXZone1Width    JTXZone1Width    `json:"JTX_Zone1_Width,omitempty"`
	XxCz             XxCz             `json:"XX_CZ,omitempty"`
	XzsSw            XzsSw            `json:"XZS_SW,omitempty"`
	R2OpenCTRL       R2OpenCTRL       `json:"R2_Open_CTRL,omitempty"`
	F1Closed         F1Closed         `json:"F1_Closed,omitempty"`
	JtxXy            JtxXy            `json:"JTX_XY,omitempty"`
	LDPowerON        LDPowerON        `json:"LD_PowerON,omitempty"`
	JTXZoneP1        JTXZoneP1        `json:"JTX_Zone_P1,omitempty"`
	BJQNightMode     BJQNightMode     `json:"BJQ_night_mode,omitempty"`
	R1CloseCTRL      R1CloseCTRL      `json:"R1_Close_CTRL,omitempty"`
	F3Closed         F3Closed         `json:"F3_Closed,omitempty"`
	JTXZoneP2        JTXZoneP2        `json:"JTX_Zone_P2,omitempty"`
	StopNav          StopNav          `json:"STOP_NAV,omitempty"`
	JTXZone2Width    JTXZone2Width    `json:"JTX_Zone2_Width,omitempty"`
	SxCz             SxCz             `json:"SX_CZ,omitempty"`
	R3Kd             R3Kd             `json:"R3_KD,omitempty"`
	LDJSJCommErr     LDJSJCommErr     `json:"LD_JSJ_Comm_err,omitempty"`
	F1OpenCTRL       F1OpenCTRL       `json:"F1_Open_CTRL,omitempty"`
	JTXZone          JTXZone          `json:"JTX_Zone,omitempty"`
	ZPDDCtrl         ZPDDCtrl         `json:"ZP_DD_Ctrl,omitempty"`
	ZPMotorSpeed     ZPMotorSpeed     `json:"ZP_Motor_Speed,omitempty"`
	R1Opened         R1Opened         `json:"R1_Opened,omitempty"`
	F3CloseCTRL      F3CloseCTRL      `json:"F3_Close_CTRL,omitempty"`
	F2Opened         F2Opened         `json:"F2_Opened,omitempty"`
	F4Opened         F4Opened         `json:"F4_Opened,omitempty"`
	AutoDetect       AutoDetect       `json:"Auto_Detect,omitempty"`
	DectectedAlarm   DectectedAlarm   `json:"Dectected_alarm,omitempty"`
	DetectSW         DetectSW         `json:"Detect_SW,omitempty"`
	DetectStart      DetectStart      `json:"Detect_Start,omitempty"`
	R3OpenCTRL       R3OpenCTRL       `json:"R3_Open_CTRL,omitempty"`
	ReserveI7        ReserveI7        `json:"RESERVE_I7,omitempty"`
	ReserveI6        ReserveI6        `json:"RESERVE_I6,omitempty"`
	R1Closed         R1Closed         `json:"R1_Closed,omitempty"`
	R2Kd             R2Kd             `json:"R2_KD,omitempty"`
	ReserveI8        ReserveI8        `json:"RESERVE_I8,omitempty"`
	Err              Err              `json:"Err,omitempty"`
	R3CloseCTRL      R3CloseCTRL      `json:"R3_Close_CTRL,omitempty"`
	ZSWidth          ZSWidth          `json:"ZS_Width,omitempty"`
	F2OpenCTRL       F2OpenCTRL       `json:"F2_Open_CTRL,omitempty"`
	ManualTestB      ManualTestB      `json:"Manual_test_b,omitempty"`
	SxJz             SxJz             `json:"SX_JZ,omitempty"`
	PICCatch         PICCatch         `json:"PIC_Catch,omitempty"`
	DetectDataOffset DetectDataOffset `json:"DetectData_offset,omitempty"`
	DetectHighLmt    DetectHighLmt    `json:"Detect_High_lmt,omitempty"`
	R3Closed         R3Closed         `json:"R3_Closed,omitempty"`
	ReserveI1        ReserveI1        `json:"RESERVE_I1,omitempty"`
	R2Closed         R2Closed         `json:"R2_Closed,omitempty"`
	R4Closed         R4Closed         `json:"R4_Closed,omitempty"`
	F2CloseCTRL      F2CloseCTRL      `json:"F2_Close_CTRL,omitempty"`
	ReserveI3        ReserveI3        `json:"RESERVE_I3,omitempty"`
	ReserveI2        ReserveI2        `json:"RESERVE_I2,omitempty"`
	Calibrated       Calibrated       `json:"Calibrated,omitempty"`
	ReserveI5        ReserveI5        `json:"RESERVE_I5,omitempty"`
	ReserveI4        ReserveI4        `json:"RESERVE_I4,omitempty"`
	R4OpenCTRL       R4OpenCTRL       `json:"R4_Open_CTRL,omitempty"`
	R1Kd             R1Kd             `json:"R1_KD,omitempty"`
	F4Closed         F4Closed         `json:"F4_Closed,omitempty"`
	XySw             XySw             `json:"XY_SW,omitempty"`
	JTXZoneN2        JTXZoneN2        `json:"JTX_Zone_N2,omitempty"`
	F2Closed         F2Closed         `json:"F2_Closed,omitempty"`
	F3OpenCTRL       F3OpenCTRL       `json:"F3_Open_CTRL,omitempty"`
	Sx               Sx               `json:"SX,omitempty"`
	SySw             SySw             `json:"SY_SW,omitempty"`
	JTXZoneN1        JTXZoneN1        `json:"JTX_Zone_N1,omitempty"`
	DetectDisWater   DetectDisWater   `json:"Detect_Dis_water,omitempty"`
	R4CloseCTRL      R4CloseCTRL      `json:"R4_Close_CTRL,omitempty"`
	JKPLCHeart       JKPLCHeart       `json:"JK_PLC_Heart,omitempty"`
	JTXRight         JTXRight         `json:"JTX_Right,omitempty"`
	XxJz             XxJz             `json:"XX_JZ,omitempty"`
	LdZ              LdZ              `json:"LD_Z,omitempty"`
	CalibrateReq     CalibrateReq     `json:"Calibrate_Req,omitempty"`
	F3Opened         F3Opened         `json:"F3_Opened,omitempty"`
	LdY              LdY              `json:"LD_Y,omitempty"`
	LdX              LdX              `json:"LD_X,omitempty"`
	JKPLCCommErr     JKPLCCommErr     `json:"JK_PLC_Comm_err,omitempty"`
	R1OpenCTRL       R1OpenCTRL       `json:"R1_Open_CTRL,omitempty"`
	DectectedWarm    DectectedWarm    `json:"Dectected_warm,omitempty"`
	F1Opened         F1Opened         `json:"F1_Opened,omitempty"`
	BJQTestB         BJQTestB         `json:"BJQ_test_b,omitempty"`
	R2Opened         R2Opened         `json:"R2_Opened,omitempty"`
	BJQAlarmAllow    BJQAlarmAllow    `json:"BJQ_Alarm_allow,omitempty"`
	ZMHigh           ZMHigh           `json:"ZM_High,omitempty"`
}

func (rjs *ResponseJsonStruct) VerifyMaybeValid(bs []byte) bool {
	if bytes.Index(bs, []byte("Cross_Dis")) < 0 ||
		bytes.Index(bs, []byte("JTX_Zone1_Width")) < 0 {
		return false
	}
	return true
}
