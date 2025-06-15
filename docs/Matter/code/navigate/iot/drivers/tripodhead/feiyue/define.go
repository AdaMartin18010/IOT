package feiyue

const (
	ProtocolPrefix            = 0xFF
	ProtocolDataLen           = 7
	CommandRebootCheckOn      = 0x0003
	CommandRebootCheckOff     = 0x0005
	CommandRebootCheckNow     = 0x0007
	CommandRebootCheckSubData = 0x0077
	AngleMeasuringScale       = 100
)

var TripodheadSlaveId = 0x01 //slave device address

// Packager specifies the communication layer function.
type Packager interface {
	Encode() (tdb []byte, err error)
	Decode(tdb []byte) (err error)
}
