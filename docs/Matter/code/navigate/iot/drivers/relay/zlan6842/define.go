package zlan6842

import (
	cm "navigate/common/model"
	bs "navigate/internal/bytestr"
)

const (
	ZlanDOitemsMax = 8
	ZlanDOindex    = 1
	ZlanSwitchOn   = 0xFF00
	ZlanSwitchOff  = 0x0000
	ZlanDOaddr     = 0x0010
)

type RelayCommonder interface {
	SetDOsAll(onOrOff bool) error
	SetDOAny(index uint, onOrOff bool) error
	SetDOs(dos DOitems) error
	GetDOs() (dos DOitems, e error)
}

type DOitem struct {
	Id uint16
	On bool
}

func (d *DOitem) String() string {
	//return fmt.Sprintf("AddressId:%d,On:%t", d.Id, d.On)
	buf, _ := cm.Json.Marshal(d)
	return bs.BytesToString(buf)
}

type DOitems []DOitem

func (ds DOitems) Len() int { return len(ds) }

func (ds *DOitems) JsonData() []byte {
	sjson, _ := cm.Json.Marshal(ds)
	return sjson
}

func (ds *DOitems) String() string {
	sjson, _ := cm.Json.Marshal(ds)
	return bs.BytesToString(sjson)
}

// 全部on or 全部off
func (ds *DOitems) IsAll(onOrOff bool) bool {
	if onOrOff {
		allON := true
		for _, item := range *ds {
			allON = item.On && allON
		}
		return allON
	} else {
		allOff := true
		for _, item := range *ds {
			allOff = !item.On && allOff
		}
		return allOff
	}
}

func (ds *DOitems) IsOn(i uint16) bool {
	for _, item := range *ds {
		if item.Id == i {
			return item.On
		}
	}
	return false
}
