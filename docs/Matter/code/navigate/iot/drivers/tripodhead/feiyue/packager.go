package feiyue

import (
	"encoding/binary"
	"fmt"

	cm "navigate/common/model"
	drs "navigate/iot/drivers"
)

/*
*********************************
#define Pelco-D-P  commond resquest & respond
FF address commond(uint16) DataH DataL checksum
************************************
*/
type PelcoDataUnit struct {
	Command uint16
	Data    uint16
}

func (pdu *PelcoDataUnit) String() string {
	ss, _ := cm.Json.Marshal(pdu)
	return string(ss)
}

// Encode adds Pelco application protocol header:
// ProtocolPrefix: 1 byte 0xFF
// Slave address : 1 byte
// Command : 2 bytes
// Sub Data: 2 bytes
// CheckSum: 1 byte
func (pdu *PelcoDataUnit) Encode() (tdb []byte, err error) {
	tdb = make([]byte, ProtocolDataLen)
	tdb[0] = uint8(ProtocolPrefix)
	tdb[1] = uint8(TripodheadSlaveId)

	binary.BigEndian.PutUint16(tdb[2:], pdu.Command)
	binary.BigEndian.PutUint16(tdb[4:], pdu.Data)

	var lrc drs.Lrc
	checkSum := lrc.Reset().PushBytes(tdb[1:]).Value()
	tdb[6] = checkSum
	return
}

// Decode extracts PelcoDataUnit from TCP frame:
// ProtocolPrefix: 1 byte 0xFF
// Slave address : 1 byte
// Command : 2 bytes
// Sub Data: 2 bytes
// CheckSum: 1 byte
func (pdu *PelcoDataUnit) Decode(tdb []byte) (err error) {
	pduLength := len(tdb)
	if pduLength != int(ProtocolDataLen) {
		err = fmt.Errorf(" Pelco: length in response '%v' does not match pdu data length '%v'",
			pduLength, ProtocolDataLen)
		return
	}
	prefix := uint8(tdb[0])
	if prefix != uint8(ProtocolPrefix) {
		err = fmt.Errorf(" Pelco: prefix '%v' does not match pdu prefix '%v'",
			prefix, ProtocolPrefix)
		return
	}

	slaveAddr := uint8(tdb[1])
	if slaveAddr != uint8(TripodheadSlaveId) {
		err = fmt.Errorf(" Pelco: slaveAddr '%v' does not match slave Address '%v'",
			slaveAddr, TripodheadSlaveId)
		return
	}

	var lrc drs.Lrc
	checkSum := lrc.Reset().PushBytes(tdb[1:6]).Value()
	if checkSum != byte(tdb[6]) {
		err = fmt.Errorf(" Pelco: checkSum '%v' does not match lrc Algorithm '%v'",
			tdb[6], checkSum)
	}

	pdu.Command = binary.BigEndian.Uint16(tdb[2:4])
	pdu.Data = binary.BigEndian.Uint16(tdb[4:6])
	return
}
