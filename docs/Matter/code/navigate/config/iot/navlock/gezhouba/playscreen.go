package gezhouba

import (
	"errors"
	"net/url"
	"strings"

	cm "navigate/common"

	jsoniter "github.com/json-iterator/go"
)

type PlayScreenConfig struct {
	Enable           bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	EnableAutoTest   bool   `mapstructure:"enable_autotest" json:"enable_autotest" yaml:"enable_autotest"`
	EnableDebug      bool   `mapstructure:"enable_debug" json:"enable_debug" yaml:"enable_debug"`
	AutoTestDirector string `mapstructure:"autotest_director" json:"autotest_director" yaml:"autotest_director"`

	EnableClearAndPublishProgram   bool   `mapstructure:"enable_clear_and_publish_program" json:"enable_clear_and_publish_program" yaml:"enable_clear_and_publish_program"`
	EnableExitActiveDefaultProgram bool   `mapstructure:"enable_exit_active_default_program" json:"enable_exit_active_default_program" yaml:"enable_exit_active_default_program"`
	DefaultProgramName             string `mapstructure:"default_program_name" json:"default_program_name" yaml:"default_program_name"`
	DefaultProgramCnf              string `mapstructure:"default_program_cnf" json:"default_program_cnf" yaml:"default_program_cnf"`
	DefaultProgramLinesCnf         string `mapstructure:"default_program_lines_cnf" json:"default_program_lines_cnf" yaml:"default_program_lines_cnf"`
	SpeedProgramName               string `mapstructure:"speed_program_name" json:"speed_program_name" yaml:"speed_program_name"`
	NormalSpeedText                string `mapstructure:"normal_speed_text" json:"normal_speed_text" yaml:"normal_speed_text"`
	OverSpeedText                  string `mapstructure:"over_speed_text" json:"over_speed_text" yaml:"over_speed_text"`
	SpeedProgramCnf                string `mapstructure:"speed_program_cnf" json:"speed_program_cnf" yaml:"speed_program_cnf"`
	SpeedProgramLinesCnf           string `mapstructure:"speed_program_lines_cnf" json:"speed_program_lines_cnf" yaml:"speed_program_lines_cnf"`

	Name string `mapstructure:"name" json:"name" yaml:"name"`
	Tag  string `mapstructure:"tag" json:"tag" yaml:"tag"`

	Url                string `mapstructure:"url" json:"url" yaml:"url"`
	ConnectReadTimeout int    `mapstructure:"connect-read-timeout" json:"connect-read-timeout"  yaml:"connect-read-timeout"`
	DoTimeInterval     int    `mapstructure:"do-timeinterval" json:"do-timeinterval" yaml:"do-timeinterval"`
}

func (c *PlayScreenConfig) Validate() error {
	if c.Name == "" {
		return errors.New("name is required")
	}

	if c.EnableAutoTest {
		if c.AutoTestDirector == "" {
			return errors.New("director is required not empty")
		}
	}

	if !c.Enable {
		return nil
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

	if strings.TrimSpace(c.DefaultProgramName) == "" {
		return errors.New("speed_program_name is required not empty")
	}

	if c.DefaultProgramCnf != "" {
		fp, err := cm.DealWithExecutedCurrentFilePath(c.DefaultProgramCnf)
		if err != nil {
			return err
		}
		c.DefaultProgramCnf = fp

		exists, err := cm.FileExists(c.DefaultProgramCnf)
		if err != nil {
			return err
		}

		if !exists {
			return errors.New(c.DefaultProgramCnf + " is not exists.")
		}

	} else {
		return errors.New("default_program_cnf is required not empty")
	}

	if c.DefaultProgramLinesCnf != "" {
		fp, err := cm.DealWithExecutedCurrentFilePath(c.DefaultProgramLinesCnf)
		if err != nil {
			return err
		}
		c.DefaultProgramLinesCnf = fp

		exists, err := cm.FileExists(c.DefaultProgramLinesCnf)
		if err != nil {
			return err
		}

		if !exists {
			return errors.New(c.DefaultProgramLinesCnf + " is not exists.")
		}

	} else {
		return errors.New("default_program_lines_cnf is required not empty")
	}

	if strings.TrimSpace(c.SpeedProgramName) == "" {
		return errors.New("speed_program_name is required not empty")
	}

	if c.SpeedProgramCnf != "" {
		fp, err := cm.DealWithExecutedCurrentFilePath(c.SpeedProgramCnf)
		if err != nil {
			return err
		}
		c.SpeedProgramCnf = fp

		exists, err := cm.FileExists(c.SpeedProgramCnf)
		if err != nil {
			return err
		}

		if !exists {
			return errors.New(c.SpeedProgramCnf + " is not exists.")
		}

	} else {
		return errors.New("speed_program_cnf is required not empty")
	}

	if c.SpeedProgramLinesCnf != "" {
		fp, err := cm.DealWithExecutedCurrentFilePath(c.SpeedProgramLinesCnf)
		if err != nil {
			return err
		}
		c.SpeedProgramLinesCnf = fp

		exists, err := cm.FileExists(c.SpeedProgramLinesCnf)
		if err != nil {
			return err
		}

		if !exists {
			return errors.New(c.SpeedProgramLinesCnf + " is not exists.")
		}

	} else {
		return errors.New("speed_program_lines_cnf is required not empty")
	}
	return nil
}

func (c *PlayScreenConfig) AutoTest(eb bool) {
	c.EnableAutoTest = eb
}

func (c *PlayScreenConfig) String() string {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.Marshal(c)
	return string(sbuf)
}
