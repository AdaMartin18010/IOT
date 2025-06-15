package v220828

import (
	"encoding/binary"
	"errors"
	"fmt"

	cm "navigate/common/model"
	bs "navigate/internal/bytestr"
	drs "navigate/iot/drivers"
)

//*********************************************************************
// 测距有效range : +-300m
// 测速有效range : +-2.6368m/s
// 测速和测距判断-在误差范围内认为等同: 测距误差为1.5m，测速误差为0.045m
// 静止或者停止判断标准: 速度低于0.2m/s 协议上可以认为已经开始停止
// --- 注解: 雷达速度分辨的一个单位是0.0412m/s --- ~~0.045m/s
//  		距离分辨的一个单位是1.4642m  ---- ~~1.5m
//*********************************************************************

/*
*****************************************************************************************
define radar  data (no resquest & only respond)
比如:
bd c5 80 08 00 0a
bd c5 80 08 01 01 00 93 04 41 00 02 00 0a 00 02 00 00 00 00 00 00 00 00 00 ee 95 04 00 78
bd c5 80 08 01 01 00 26 05 da ff 02 00 0b 00 02 00 00 00 00 00 00 00 00 00 d8 af 04 00 a9
*****************************************************************************************
*/
var (
	DistanceMeasuringScale uint16 = 100
	VelocityMeasuringScale uint16 = 100
)

const (
	SpeedMin    = -2.6368
	SpeedMax    = 2.6368
	DistanceMax = 350.00

	floatCompare = 0.01
)

type ShipDataUnit struct {
	SeqNum    uint8   // 第一个字节为有效值 0~255
	Distance  float32 //  0  ~ 400m
	Speed     float32 `json:",string"` //-2.6368 ~ 2.6368 m/s  离雷达越近速度为负值 离雷达远去为正值
	Position  uint16  // 0,1,2 分别代表 中 左 右 ----按区域划分 人与雷达同向的方位模糊划分
	Amplitude uint16  // 幅度大小 ---检查雷达使用
}

func (sdu *ShipDataUnit) CompareDistanceSpeed(other *ShipDataUnit) bool {
	if (sdu.Distance-other.Distance) <= floatCompare ||
		(other.Distance-sdu.Distance) <= floatCompare {
		if (sdu.Speed-other.Speed) <= floatCompare ||
			(other.Speed-sdu.Speed) <= floatCompare {
			return true
		}
	}
	return false
}

func (sdu *ShipDataUnit) decodeBE(rdb []byte) (err error) {
	//01 00 93 04 41 00 02 00 0a 00
	l := len(rdb)
	if l != ShipDataUnitLen {
		err = fmt.Errorf("radar: Shipdata len-%d != ShipDataUnitLen-%d",
			l, ShipDataUnitLen)
		return
	}

	sdu.SeqNum = uint8(rdb[0])
	if rdb[1] != 0 {
		err = fmt.Errorf("radar: Shipdata Shipdata.SeqNum Second byte != 0x00-%X",
			rdb[1])
	}

	sdu.Distance = float32(binary.BigEndian.Uint16(rdb[2:4])) / float32(DistanceMeasuringScale)
	sdu.Speed = float32(int16(binary.BigEndian.Uint16(rdb[4:6]))) / float32(VelocityMeasuringScale)
	sdu.Position = binary.BigEndian.Uint16(rdb[6:8])
	sdu.Amplitude = binary.BigEndian.Uint16(rdb[8:10])
	return
}

func (sdu *ShipDataUnit) decodeLE(rdb []byte) (err error) {
	//01 00 93 04 41 00 02 00 0a 00
	l := len(rdb)
	if l != ShipDataUnitLen {
		err = fmt.Errorf("radar: Shipdata len-%d != ShipDataUnitLen-%d",
			l, ShipDataUnitLen)
		return
	}

	sdu.SeqNum = uint8(rdb[0])
	if rdb[1] != 0 {
		err = fmt.Errorf("radar: Shipdata Shipdata.SeqNum Second byte != 0x00-%X",
			rdb[1])
	}

	sdu.Distance = float32(binary.LittleEndian.Uint16(rdb[2:4])) / float32(DistanceMeasuringScale)
	sdu.Speed = float32(int16(binary.LittleEndian.Uint16(rdb[4:6]))) / float32(VelocityMeasuringScale)
	sdu.Position = binary.LittleEndian.Uint16(rdb[6:8])
	sdu.Amplitude = binary.LittleEndian.Uint16(rdb[8:10])
	return
}

func (sdu *ShipDataUnit) String() string {
	// return fmt.Sprintf("ShipData:SeqNum-[%0X](%d),Distance-[%0X](%d),Velocity-%f,Position-[%0X](%d),Amplitude-%0X\n",
	// 	sdu.SeqNum, sdu.SeqNum, sdu.Distance,
	// 	sdu.Distance, sdu.Speed, sdu.Position,
	// 	sdu.Amplitude)
	ss, _ := cm.Json.Marshal(sdu)
	return bs.BytesToString(ss)
}

func (sdu *ShipDataUnit) Verify() (err error) {
	ok := true
	ss := ""
	if sdu.Distance-DistanceMax > 0.0001 {
		ss += fmt.Sprintf("\tDistance > %.4f:%.4f;\n", DistanceMax, sdu.Distance)
		ok = false
	}

	if sdu.Speed < SpeedMin || sdu.Speed > SpeedMax {
		ss += fmt.Sprintf("\tSpeed not in [%.4f,%.4f]: %.4f;\n",
			SpeedMin, SpeedMax, sdu.Speed)
		ok = false
	}

	if sdu.Position > 2 {
		ss += fmt.Sprintf("\tPosition not in [0,1,2]:%d;\n", sdu.Position)
		ok = false
	}

	if !ok {
		ss = fmt.Sprintf("Verify List:\n%s", ss)
		err = errors.New(ss)
	}
	return
}

// 当ShipCount为0时 其余字段全无
type RadarDataUnit struct {
	ShipCount uint8  // number of ships
	TimeMark  uint32 // time mark
	Ships     []ShipDataUnit
}

func (rdu *RadarDataUnit) Reset() {
	rdu.ShipCount = 0
	rdu.TimeMark = 0
	rdu.Ships = nil
}

func (rdu *RadarDataUnit) String() string {
	// ss := fmt.Sprintf("RadarData:\n ShipCount-%d,TimeMark-%0X\n",
	// RadarDataUnit.ShipCount, RadarDataUnit.TimeMark)
	// for _, shipdata := range RadarDataUnit.Ships {
	// 	ss += shipdata.String()
	// }
	ss, _ := cm.Json.Marshal(rdu)

	return bs.BytesToString(ss)
}

// 网络序 解析
func (rdu *RadarDataUnit) decodeBE(rdb []byte) (err error) {
	l := len(rdb)
	if l < ProtocolDataMinLen {
		err = fmt.Errorf("radar: len-%d < ProtocolDataMinLen-%d",
			l, ProtocolDataMinLen)
		return
	}
	if (rdb[0] != ProtocolHead[0]) ||
		(rdb[1] != ProtocolHead[1]) ||
		(rdb[2] != ProtocolHead[2]) ||
		(rdb[3] != ProtocolHead[3]) {
		err = fmt.Errorf("radar: head data (%v)  != ProtocolHead(%v)",
			rdb[0:4], ProtocolHead)
		return
	}

	if rdb[4] == 0x00 {
		//如果ShipCount 船的数量为0时 除了检验码后续没有数据 可以认为是心跳包
		//bd c5 80 08 00 0a
		rdu.ShipCount = uint8(rdb[4])
		rdu.Ships = nil
		rdu.TimeMark = 0
		if rdb[5] != 0x0A {
			err = fmt.Errorf("radar: head data (%+v)  != ProtocolHead(%+v)",
				rdb[0:6], ProtocolHeartBeat)
		}
		//直接退出 后续的checksum 算法适配雷达的程序bug
		return
	} else if rdb[4] >= 1 {
		rdu.ShipCount = rdb[4]
		rdu.Ships = make([]ShipDataUnit, rdu.ShipCount)
		for i := 1; i <= int(rdu.ShipCount); i++ {
			rdu.Ships[i-1].decodeBE(rdb[(10*(i-1) + 5):(10*(i-1) + 15)])
		}
	}

	rdu.TimeMark = binary.BigEndian.Uint32(rdb[(l - 5):(l - 1)])

	//最好是接收数据的时候处理
	var (
		lrc      drs.Lrc
		checkSum byte = 0
	)
	checkSum = lrc.Reset().PushBytes(rdb[0:(l - 1)]).Value()
	if checkSum != byte(rdb[(l-1)]) {
		err = fmt.Errorf(" Radar: checkSum '%X' does not match lrc Algorithm '%X'",
			rdb[(l-1)], checkSum)
	}

	return
}

// 小端解析
func (rdu *RadarDataUnit) decodeLE(rdb []byte) (err error) {
	l := len(rdb)
	if l < ProtocolDataMinLen {
		err = fmt.Errorf("radar: len-%d < ProtocolDataMinLen-%d",
			l, ProtocolDataMinLen)
		return
	}
	if (rdb[0] != ProtocolHead[0]) ||
		(rdb[1] != ProtocolHead[1]) ||
		(rdb[2] != ProtocolHead[2]) ||
		(rdb[3] != ProtocolHead[3]) {
		err = fmt.Errorf("radar: head data (%+v)  != ProtocolHead(%+v)",
			rdb[0:4], ProtocolHead)
		return
	}

	if rdb[4] == 0x00 {
		//如果ShipCount 船的数量为0时 除了检验码后续没有数据 可以认为是心跳包
		//bd c5 80 08 00 0a
		rdu.ShipCount = uint8(rdb[4])
		rdu.Ships = nil
		rdu.TimeMark = 0
		if rdb[5] != 0x0A {
			err = fmt.Errorf("radar: head data (%+v)  != ProtocolHead(%+v)",
				rdb[0:6], ProtocolHeartBeat)
		}
		//直接退出 后续的checksum 算法适配雷达的程序bug
		return
	} else if rdb[4] >= 1 {
		rdu.ShipCount = rdb[4]
		rdu.Ships = make([]ShipDataUnit, rdu.ShipCount)
		for i := 1; i <= int(rdu.ShipCount); i++ {
			rdu.Ships[i-1].decodeLE(rdb[(10*(i-1) + 5):(10*(i-1) + 15)])
		}
	}

	rdu.TimeMark = binary.LittleEndian.Uint32(rdb[(l - 5):(l - 1)])

	//最好是接收数据的时候处理
	var lrc drs.Lrc
	checkSum := lrc.Reset().PushBytes(rdb[0:(l - 1)]).Value()
	if checkSum != byte(rdb[(l-1)]) {
		err = fmt.Errorf(" Radar: checkSum '%X' does not match lrc Algorithm '%X'",
			rdb[(l-1)], checkSum)
	}

	return
}

func (rdu *RadarDataUnit) Decode(rdb []byte) (*RadarDataUnit, error) {
	//return rdu, rdu.decodeBE(rdb)
	return rdu, rdu.decodeLE(rdb)
}

func (rdu *RadarDataUnit) Verify() (err error) {
	ok := true
	ss := ""
	if rdu.ShipCount > 0 && rdu.Ships != nil {
		if int(rdu.ShipCount) != len(rdu.Ships) {
			ss += fmt.Sprintf("ShipCount != len(Ships):%d != %d;", rdu.ShipCount, len(rdu.Ships))
			ok = false
		}
		for i, ship := range rdu.Ships {
			err = ship.Verify()
			if err != nil {
				ss += fmt.Sprintf("Ship[%d]: %v", i, err)
				ok = false
			}
		}
	}
	if !ok {
		ss = fmt.Sprintf("Verify List:\n%s", ss)
		err = errors.New(ss)
	}

	return
}
