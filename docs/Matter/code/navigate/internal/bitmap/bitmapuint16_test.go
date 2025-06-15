package bitmap_test

import (
	"navigate/internal/bitmap"
	"testing"
)

func TestBasic(t *testing.T) {
	bitMap, _ := bitmap.NewBitMapUint16(33)
	val, err := bitMap.GetPositionBit(0)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
	}

	err = bitMap.SetBitToOne(4)
	if err != nil {
		t.Error(err)
	}
	err = bitMap.SetBitToOne(5)
	if err != nil {
		t.Error(err)
	}
	err = bitMap.SetBitToOne(15)
	if err != nil {
		t.Error(err)
	}
	val, err = bitMap.GetPositionBit(4)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
	}
	val, err = bitMap.GetPositionUint16(0)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
	}

	err = bitMap.SetBitToOne(28)
	if err != nil {
		t.Error(err)
	}

	err = bitMap.SetBitToOne(31)
	if err != nil {
		t.Error(err)
	}

	val, err = bitMap.GetPositionUint16(1)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
	}

	bitMap.ResetAllOne()
	val, err = bitMap.GetPositionUint16(0)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0xFFFF {
			t.Errorf("position should be 0xFFFF")
		}
		t.Logf("%016b", val)
	}
	val, err = bitMap.GetPositionUint16(1)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0xFFFF {
			t.Errorf("position should be 0xFFFF")
		}
		t.Logf("%016b", val)
	}
	val, err = bitMap.GetPositionUint16(2)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0xFFFF {
			t.Errorf("position should be 0xFFFF")
		}
		t.Logf("%016b", val)
	}

	bitMap.ResetAllZero()
	val, err = bitMap.GetPositionUint16(0)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}
	val, err = bitMap.GetPositionUint16(1)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}
	val, err = bitMap.GetPositionUint16(2)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}

	bitMap.ResetToValue(2, 0xF0F0)
	val, err = bitMap.GetPositionUint16(0)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
	}
	val, err = bitMap.GetPositionUint16(1)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}

	val, err = bitMap.GetPositionUint16(2)
	if err != nil {
		t.Error(err)
	} else {
		t.Logf("%016b", val)
		if val != 0xF0F0 {
			t.Errorf("position should be 0xF0F0")
		}
	}

	bitMap.ResetAllZero()
	val, err = bitMap.GetPositionBit(32)
	if err != nil {
		t.Error(err)
	} else {
		if val != 0x0000 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}
	bitMap.ResetAllOne()
	val, err = bitMap.GetPositionBit(32)
	if err != nil {
		t.Error(err)
	} else {
		if val != 1 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}

	val, err = bitMap.GetPositionBit(17)
	if err != nil {
		t.Error(err)
	} else {
		if val != 1 {
			t.Errorf("position should be 0x0000")
		}
		t.Logf("%016b", val)
	}
}
