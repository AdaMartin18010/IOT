package iot

import "errors"

type RadarConfig struct {
	Enable         bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest bool   `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug    bool   `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`
	DriveName      string `mapstructure:"drive-name" json:"drive-name" yaml:"drive-name"`
	DeviceTag      string `mapstructure:"device-tag" json:"device-tag" yaml:"device-tag"`

	Ipaddress string `mapstructure:"ip-address" json:"ip-address" yaml:"ip-address"`
	Port      int    `mapstructure:"ip-port" json:"ip-port" yaml:"ip-port"`

	ConnectReadTimeout         int `mapstructure:"connect-read-timeout" json:"connect-read-timeout"  yaml:"connect-read-timeout"`
	ReConnectRetryTimeInterval int `mapstructure:"reconnect-retry-timeinterval" json:"reconnect-retry-timeinterval" yaml:"reconnect-retry-timeinterval"`
	IdleTimeout                int `mapstructure:"idle-timeout" json:"idle-timeout" yaml:"idle-timeout"`
	ReadTimeInterval           int `mapstructure:"read-timeinterval" json:"read-timeinterval" yaml:"read-timeinterval"`
	WriteTimeInterval          int `mapstructure:"write-timeinterval" json:"write-timeinterval" yaml:"write-timeinterval"`
	SlaveId                    int `mapstructure:"slave-id" json:"slave-id" yaml:"slave-id"`
}

func (r *RadarConfig) Validate() error {
	if r.DriveName == "" {
		return errors.New("drive-name is required")
	}
	if r.Ipaddress == "" {
		return errors.New("ip-address is required")
	}
	if r.Port <= 0 || r.Port > 65535 {
		return errors.New("ip-port is required > 0 and < 65535")
	}
	if r.ConnectReadTimeout < 0 {
		return errors.New("connect-read-timeout is required >= 0")
	}
	if r.IdleTimeout < 0 {
		return errors.New("idle-timeout is required >= 0")
	}
	if r.ReConnectRetryTimeInterval < 0 {
		return errors.New("reconnect-retry-timeinterval is required >0")
	} else if r.ReConnectRetryTimeInterval == 0 {
		//为0 使用默认配置
		r.ReConnectRetryTimeInterval = 5
	}

	if r.SlaveId <= 0 {
		return errors.New("slave-id is required > 0")
	}

	if r.ReadTimeInterval < 0 {
		return errors.New("read-timeinterval is required >= 0")
	}

	if r.WriteTimeInterval < 0 {
		return errors.New("write-timeinterval is required >= 0")
	}

	return nil
}

func (r *RadarConfig) AutoTest(eb bool) {
	r.EnableAutoTest = eb
}
