package gezhouba

import (
	"errors"
	"net/url"

	jsoniter "github.com/json-iterator/go"
)

type HttpClientConfig struct {
	Enable         bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest bool   `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug    bool   `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`
	Name           string `mapstructure:"name" json:"name" yaml:"name"`
	Tag            string `mapstructure:"tag" json:"tag" yaml:"tag"`

	Url                string `mapstructure:"url" json:"url" yaml:"url"`
	ConnectReadTimeout int    `mapstructure:"connect-read-timeout" json:"connect-read-timeout"  yaml:"connect-read-timeout"`
	DoTimeInterval     int    `mapstructure:"do-timeinterval" json:"do-timeinterval" yaml:"do-timeinterval"`
}

func (c *HttpClientConfig) Validate() error {
	if c.Name == "" {
		return errors.New("name is required")
	}

	if c.Url == "" {
		return errors.New("url is required")
	} else {
		_, err := url.ParseRequestURI(c.Url)
		if err != nil {
			return err
		}
	}

	if c.ConnectReadTimeout < 0 {
		return errors.New("connect-read-timeout is required >= 0")
	}

	return nil
}

func (c *HttpClientConfig) AutoTest(eb bool) {
	c.EnableAutoTest = eb
}

func (c *HttpClientConfig) String() string {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.Marshal(c)
	return string(sbuf)
}
