package iot

import "errors"

type TripodheadConfig struct {
	Enable         bool `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest bool `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug    bool `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`

	//EnablePowerOnAutoCheck bool `mapstructure:"enable_poweron_auto_check" json:"enable_poweron_auto_check" yaml:"enable_poweron_auto_check"`
	DriveName string `mapstructure:"drive-name" json:"drive-name" yaml:"drive-name"`
	DeviceTag string `mapstructure:"device-tag" json:"device-tag" yaml:"device-tag"`

	Ipaddress string `mapstructure:"ip-address" json:"ip-address" yaml:"ip-address"`
	Port      int    `mapstructure:"ip-port" json:"ip-port" yaml:"ip-port"`

	ConnectReadTimeout         int `mapstructure:"connect-read-timeout" json:"connect-read-timeout"  yaml:"connect-read-timeout"`
	ReConnectRetryTimeInterval int `mapstructure:"reconnect-retry-timeinterval" json:"reconnect-retry-timeinterval" yaml:"reconnect-retry-timeinterval"`
	IdleTimeout                int `mapstructure:"idle-timeout" json:"idle-timeout" yaml:"idle-timeout"`
	ReadTimeInterval           int `mapstructure:"read-timeinterval" json:"read-timeinterval" yaml:"read-timeinterval"`
	WriteTimeInterval          int `mapstructure:"write-timeinterval" json:"write-timeinterval" yaml:"write-timeinterval"`
	SlaveId                    int `mapstructure:"slave-id" json:"slave-id" yaml:"slave-id"`

	HorizontalAngle float32 `mapstructure:"horizontal-angle,string" json:"horizontal-angle,string" yaml:"horizontal-angle,string"`
	VerticalAngle   float32 `mapstructure:"vertical-angle,string" json:"vertical-angle,string" yaml:"vertical-angle,string"`
}

func (c *TripodheadConfig) Validate() error {
	if c.DriveName == "" {
		return errors.New("drive-name is required")
	}
	if c.Ipaddress == "" {
		return errors.New("ip-address is required")
	}
	if c.Port <= 0 || c.Port > 65535 {
		return errors.New("ip-port is required > 0 and < 65535")
	}
	if c.ConnectReadTimeout < 0 {
		return errors.New("connect-read-timeout is required >= 0")
	}
	if c.IdleTimeout < 0 {
		return errors.New("idle-timeout is required >= 0")
	}
	if c.ReConnectRetryTimeInterval < 0 {
		return errors.New("reconnect-retry-timeinterval is required >0")
	} else if c.ReConnectRetryTimeInterval == 0 {
		//为0 使用默认配置
		c.ReConnectRetryTimeInterval = 5
	}

	if c.SlaveId <= 0 {
		return errors.New("slave-id is required > 0")
	}

	if c.ReadTimeInterval < 0 {
		return errors.New("read-timeinterval is required >= 0")
	}

	if c.WriteTimeInterval < 0 {
		return errors.New("write-timeinterval is required >= 0")
	}

	if c.VerticalAngle < 0.0 {
		return errors.New("vertical-angle is required >= 0.0")
	}

	if c.HorizontalAngle < 0.0 {
		return errors.New("horizontal-angle is required >= 0.0")
	}
	return nil
}

func (c *TripodheadConfig) AutoTest(eb bool) {
	c.EnableAutoTest = eb
}
