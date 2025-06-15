package drivers

// Longitudinal Redundancy Checking
type Lrc struct {
	sum uint8
}

func (lrc *Lrc) Reset() *Lrc {
	lrc.sum = 0
	return lrc
}

func (lrc *Lrc) PushByte(b byte) *Lrc {
	lrc.sum += b
	return lrc
}

func (lrc *Lrc) PushBytes(data []byte) *Lrc {
	var b byte
	for _, b = range data {
		lrc.sum += b
	}
	return lrc
}

func (lrc *Lrc) Value() byte {
	// Return twos complement
	// return uint8(-int8(lrc.sum))
	// 云台协议的偶校验 私有算法
	return uint8(lrc.sum)
}

func (lrc *Lrc) RadarValue() byte {
	//雷达 适配的程序
	return uint8(lrc.sum - 1)
}
