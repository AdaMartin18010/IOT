package gezhouba

import jsoniter "github.com/json-iterator/go"

type StopLineHttpsConf struct {
	Enable         bool `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest bool `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug    bool `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`

	UpHttp   *HttpClientConfig `mapstructure:"up_http" json:"up_http" yaml:"up_http"`
	DownHttp *HttpClientConfig `mapstructure:"down_http" json:"down_http" yaml:"down_http"`
}

func (c *StopLineHttpsConf) AutoTest(eb bool) {
	c.EnableAutoTest = eb
	c.UpHttp.AutoTest(eb)
	c.DownHttp.AutoTest(eb)
}

func (c *StopLineHttpsConf) String() string {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.Marshal(c)
	return string(sbuf)
}
