package v220828

const (
	//bd c5 80 08 00 0a 最小的包大小
	ProtocolDataMinLen = 6
	ShipDataUnitLen    = 10
)

var (
	RadarSlaveId      = 0x01 //slave device address
	ProtocolHead      = [4]byte{0xBD, 0xC5, 0x80, 0x08}
	ProtocolHeartBeat = [6]byte{0xBD, 0xC5, 0x80, 0x08, 0x00, 0x0a}
)

// RaderError implements error interface.
type RaderError struct {
	FunctionCode  byte
	ExceptionCode byte
}

// Packager specifies the communication layer function.
type Packager interface {
	Decode(adu []byte) (rdu *RadarDataUnit, err error)
}

type RadarReceiver interface {
	Recive(rdu *RadarDataUnit) (rdd []byte, err error)
}
