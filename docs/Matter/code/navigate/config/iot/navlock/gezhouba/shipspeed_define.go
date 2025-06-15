package gezhouba

import (
	"fmt"
	iotcf "navigate/config/iot/drivers"

	jsoniter "github.com/json-iterator/go"
)

type SensorsUnitSetup struct {
	RelaySetup      *iotcf.RelayConfig      `mapstructure:"relay_setup" json:"relay_setup" yaml:"relay_setup"`
	TripodheadSetup *iotcf.TripodheadConfig `mapstructure:"tripodhead_setup" json:"tripodhead_setup" yaml:"tripodhead_setup"`
	RadarSetup      *iotcf.ModbusConfig     `mapstructure:"radar_setup" json:"radar_setup" yaml:"radar_setup"`
}

func (s *SensorsUnitSetup) Validate() error {

	if s.RadarSetup != nil {
		if err := s.RadarSetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("radar_setup is not parsed")
	}

	if s.TripodheadSetup != nil {
		if err := s.TripodheadSetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("tripodhead_setup is not parsed")
	}

	if s.RelaySetup != nil {
		if err := s.RelaySetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("relay_setup is not parsed")
	}
	return nil
}

func (s *SensorsUnitSetup) AutoTest(eb bool) {
	s.RelaySetup.AutoTest(eb)
	s.TripodheadSetup.AutoTest(eb)
	s.RadarSetup.AutoTest(eb)
}

type SpeedMeasureUnitSetup struct {
	LeftSetup  *SensorsUnitSetup `mapstructure:"left_setup" json:"left_setup" yaml:"left_setup"`
	RightSetup *SensorsUnitSetup `mapstructure:"right_setup" json:"right_setup" yaml:"right_setup"`
}

func (s *SpeedMeasureUnitSetup) Validate() error {
	if s.LeftSetup != nil {
		if err := s.LeftSetup.Validate(); err != nil {
			return err
		}
	}

	if s.RightSetup != nil {
		if err := s.RightSetup.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (s *SpeedMeasureUnitSetup) AutoTest(eb bool) {
	s.LeftSetup.AutoTest(eb)
	s.RightSetup.AutoTest(eb)
}

type NavlockShipSpeedSetup struct {
	PlayScreenSetup *PlayScreenConfig      `mapstructure:"playscreen_setup" json:"playscreen_setup" yaml:"playscreen_setup"`
	UpSetup         *SpeedMeasureUnitSetup `mapstructure:"up_setup" json:"up_setup" yaml:"up_setup"`
	DownSetup       *SpeedMeasureUnitSetup `mapstructure:"down_setup" json:"down_setup" yaml:"down_setup"`
}

func (c *NavlockShipSpeedSetup) String() string {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.Marshal(c)
	return string(sbuf)
}

func (s *NavlockShipSpeedSetup) Validate() error {

	if s.PlayScreenSetup != nil {
		if err := s.PlayScreenSetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("playscreen_setup is not parsed")
	}

	if s.UpSetup != nil {
		if err := s.UpSetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("up_setup is not parsed")
	}

	if s.DownSetup != nil {
		if err := s.DownSetup.Validate(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("down_setup is not parsed")
	}

	return nil
}

func (s *NavlockShipSpeedSetup) AutoTest(eb bool) {
	s.PlayScreenSetup.AutoTest(eb)
	s.UpSetup.AutoTest(eb)
	s.DownSetup.AutoTest(eb)
}
