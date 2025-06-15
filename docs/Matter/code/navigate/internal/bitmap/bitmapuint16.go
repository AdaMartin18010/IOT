package bitmap

import (
	"errors"
	"unsafe"
)

// Bitmap represents a bitmap with a specific number of bytes.
// For modbus uint16 or muilt-uint16 manipulations
type BitMapUint16 struct {
	size  uint
	datas []uint16
}

var typesize uint = uint(unsafe.Sizeof(uint16(0))) * 8

// BitMapUint16 create a bitmap of bitSize
func NewBitMapUint16(bitSize uint) (*BitMapUint16, error) {
	bitMap := &BitMapUint16{
		size: bitSize,
	}
	len := bitSize / typesize
	if (bitSize % typesize) != 0 {
		len += 1
	}

	bitMap.datas = make([]uint16, len)
	return bitMap, nil
}

// SetOne set posth bit to 1
func (bmu *BitMapUint16) SetBitToOne(pos uint) error {
	if pos >= bmu.size {
		return errors.New("pos out of size")
	}
	len := pos / typesize
	position := pos % typesize
	bmu.datas[len] |= (1 << position)
	return nil
}

// SetZero set posth bit to 0
func (bmu *BitMapUint16) SetBitToZero(pos uint) error {
	if pos >= bmu.size {
		return errors.New("pos out of size")
	}
	len := pos / typesize
	position := pos % typesize
	bmu.datas[len] &= ^(1 << position)
	return nil
}

// GetPositionBit get posth bit
func (bmu *BitMapUint16) GetPositionBit(pos uint) (uint16, error) {
	if pos >= bmu.size {
		return 0, errors.New("pos out of size")
	}
	len := pos / typesize
	position := pos % typesize
	if x := bmu.datas[len] & (1 << position); x != 0 {
		return 1, nil
	} else {
		return 0, nil
	}

}

// GetPositionUint16 get posth uint16
func (bmu *BitMapUint16) GetPositionUint16(posUint16 uint) (uint16, error) {
	if int(posUint16) >= len(bmu.datas) {
		return 0, errors.New("pos out of size")
	}
	return bmu.datas[posUint16], nil
}

func (bmu *BitMapUint16) GetBytesToVale() {

}

func (bmu *BitMapUint16) ResetToValue(posUint16 uint, value uint16) error {
	if int(posUint16) >= len(bmu.datas) {
		return errors.New("pos out of size")
	}
	bmu.datas[posUint16] = value
	return nil
}

// ResetAllZero resets all positions to 0
func (bmu *BitMapUint16) ResetAllZero() {
	for i := range bmu.datas {
		bmu.datas[i] = 0
	}
}

// ResetAllOne resets all positions to 0
func (bmu *BitMapUint16) ResetAllOne() {
	for i := range bmu.datas {
		bmu.datas[i] = 0xFFFF
	}
}
