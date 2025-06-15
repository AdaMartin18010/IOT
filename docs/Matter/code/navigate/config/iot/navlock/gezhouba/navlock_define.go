package gezhouba

import (
	"errors"
	"fmt"

	jsoniter "github.com/json-iterator/go"
)

type NavlockConf struct {
	Enable          bool `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest  bool `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug     bool `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`
	EnableAnimation bool `mapstructure:"enable_animation" json:"enable_animation" yaml:"enable_animation"`

	Name             string  `mapstructure:"name" json:"name" yaml:"name"`
	Identifier       int     `mapstructure:"identifier" json:"identifier" yaml:"identifier"`
	NavStateInterval int     `mapstructure:"nav_state_interval" json:"nav_state_interval"`
	NavlType         int     `mapstructure:"navl_type" json:"type" yaml:"navl_type"`
	ScheduleType     int     `mapstructure:"schedule_type" json:"schedule_type"  yaml:"schedule_type"`
	SensorsLayout    int     `mapstructure:"sensors_layout" json:"sensors_layout" yaml:"sensors_layout"`
	Length           float32 `mapstructure:"length" json:"length"  yaml:"length"`
	Width            float32 `mapstructure:"width" json:"width"  yaml:"width"`

	DetectionDistanceMax  float32                `mapstructure:"detection_distance_max" json:"detection_distance_max"  yaml:"detection_distance_max"`
	DetectionSpeedMax     float32                `mapstructure:"detection_speed_max" json:"detection_speed_max"  yaml:"detection_speed_max"`
	TargetAccelerationMax float32                `mapstructure:"target_acceleration_max" json:"target_acceleration_max"  yaml:"target_acceleration_max"`
	RadarMaxRuntime       int                    `mapstructure:"radar_max_runtime" json:"radar_max_runtime"  yaml:"radar_max_runtime"`
	RadarCoolingInterval  int                    `mapstructure:"radar_cooling_interval" json:"radar_cooling_interval"  yaml:"radar_cooling_interval"`
	NavlockShipSpeedConf  *NavlockShipSpeedSetup `mapstructure:"shipspeed_setup" json:"shipspeed_setup" yaml:"shipspeed_setup"`
	NavlStatusHttp        *HttpClientConfig      `mapstructure:"navlstatus_setup" json:"navlstatus_setup" yaml:"navlstatus_setup"`
	StopLineHttp          *StopLineHttpsConf     `mapstructure:"stopline_setup" json:"stopline_setup" yaml:"stopline_setup"`
	PlatformHttp          *PlatformConfig        `mapstructure:"platform_setup" json:"platform_setup" yaml:"platform_setup"`
}

func (c *NavlockConf) String() string {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.Marshal(c)
	return string(sbuf)
}

func (c *NavlockConf) AutoTest(eb bool) {
	c.NavlockShipSpeedConf.AutoTest(eb)
	c.NavlStatusHttp.AutoTest(eb)
	c.StopLineHttp.AutoTest(eb)
	c.PlatformHttp.AutoTest(eb)
}

func (c *NavlockConf) Validate() error {
	if c.Name == "" {
		return errors.New("name : empty name")
	}

	if c.NavlType <= 0 {
		return errors.New("navl_type <= 0")
	}

	if c.SensorsLayout <= 0 {
		return errors.New("sensors_layout <= 0")
	}

	if c.Length <= 0 {
		return errors.New("length <= 0")
	}

	if c.Width <= 0 {
		return errors.New("width <= 0")
	}

	if c.NavStateInterval <= 0 {
		return errors.New("nav_state_interval <= 0")
	}

	if c.DetectionDistanceMax <= 0 {
		return errors.New("detection_distance_max <= 0")
	}

	if c.DetectionSpeedMax <= 0 {
		return errors.New("detection_speed_max <= 0")
	}

	if c.TargetAccelerationMax <= 0 {
		return errors.New("target_acceleration_max <= 0")
	}

	if c.RadarMaxRuntime <= 0 {
		return errors.New("radar_max_runtime <= 0")
	}

	if c.RadarCoolingInterval <= 0 {
		return errors.New("radar_cooling_interval <= 0")
	}

	if c.NavlockShipSpeedConf != nil {
		err := c.NavlockShipSpeedConf.Validate()
		if err != nil {
			return fmt.Errorf("NavigationlockConf.shipspeed_setup:%s", err.Error())
		}
	} else {
		return fmt.Errorf("NavigationlockConf.shipspeed_setup is not parsed")
	}

	if c.NavlStatusHttp != nil {
		err := c.NavlStatusHttp.Validate()
		if err != nil {
			return fmt.Errorf("NavigationlockConf.navlstatus_setup:%s", err.Error())
		}
	} else {
		return fmt.Errorf("NavigationlockConf.navlstatus_setup is not parsed")
	}

	if c.StopLineHttp != nil {
		err := c.StopLineHttp.UpHttp.Validate()
		if err != nil {
			return fmt.Errorf("NavigationlockConf.stopline_setup.up_http:%s", err.Error())
		}

		err = c.StopLineHttp.DownHttp.Validate()
		if err != nil {
			return fmt.Errorf("NavigationlockConf.stopline_setup.down_http:%s", err.Error())
		}

	} else {
		return fmt.Errorf("NavigationlockConf.stopline_setup is not parsed")
	}

	err := c.PlatformHttp.Validate()
	if err != nil {
		return fmt.Errorf("NavigationlockConf.platform_setup:%s", err.Error())
	}
	return nil
}
