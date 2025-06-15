package schedule

import (
	mdl "navigate/common/model"
)

type Speed struct {
	IsValid   bool
	Speed     float32
	OverSpeed bool
}

type ShipSpeeds struct {
	LeftSpeed  *Speed
	RightSpeed *Speed
}

func NewShipSpeeds() *ShipSpeeds {
	return &ShipSpeeds{
		LeftSpeed:  &Speed{},
		RightSpeed: &Speed{},
	}
}

func (ss *ShipSpeeds) String() (string, error) {
	return mdl.Json.MarshalToString(ss)
}

// if not nil  copy it value
func (ss *ShipSpeeds) AggregationWith(sso *ShipSpeeds) {
	if sso.LeftSpeed != nil && sso.LeftSpeed.IsValid {
		ss.LeftSpeed = sso.LeftSpeed
	}

	if sso.RightSpeed != nil && sso.RightSpeed.IsValid {
		ss.RightSpeed = sso.RightSpeed
	}
}

func (ss *ShipSpeeds) CompareValueTo(sso *ShipSpeeds) (valueSame bool) {
	leftValueSame := false
	rightValueSame := false
	if ss.LeftSpeed != nil {
		if sso.LeftSpeed != nil {
			if ss.LeftSpeed.IsValid == sso.LeftSpeed.IsValid &&
				(ss.LeftSpeed.Speed-ss.LeftSpeed.Speed <= 0.01) || (ss.LeftSpeed.Speed-ss.LeftSpeed.Speed <= 0.01) &&
				ss.LeftSpeed.OverSpeed == sso.LeftSpeed.OverSpeed {
				leftValueSame = true
			}
		}
	} else {
		if sso.LeftSpeed == nil {
			leftValueSame = true
		}
	}

	if ss.RightSpeed != nil {
		if sso.RightSpeed != nil {
			if ss.RightSpeed.IsValid == sso.RightSpeed.IsValid &&
				(ss.RightSpeed.Speed-ss.RightSpeed.Speed < 0.01) || (ss.RightSpeed.Speed-ss.RightSpeed.Speed < 0.01) &&
				ss.RightSpeed.OverSpeed == sso.RightSpeed.OverSpeed {
				leftValueSame = true
			}
		}
	} else {
		if sso.RightSpeed == nil {
			rightValueSame = true
		}
	}

	return (leftValueSame && rightValueSame)
}
