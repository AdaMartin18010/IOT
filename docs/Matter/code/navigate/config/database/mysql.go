package database

import (
	"errors"
	cm "navigate/common"
)

type MysqlDBConfig struct {
	Enable      bool `mapstructure:"enable" json:"enable" yaml:"enable"`
	AutoMigrate bool `mapstructure:"auto-migrate" json:"auto-migrate" yaml:"auto-migrate"` // 是否自动创建数据库表 在表不存在的情况下
	//EnableTestcase  bool   `mapstructure:"enable-testcase" json:"enable-testcase" yaml:"enable-testcase"`       // 开启MySQL的自动化用例测试
	IpAddress       string `mapstructure:"ip-address" json:"ip-address" yaml:"ip-address"`                      // 服务器地址
	IpPort          string `mapstructure:"ip-port" json:"ip-port" yaml:"ip-port"`                               // 端口
	ExtraConfig     string `mapstructure:"extra-config" json:"extra-config" yaml:"extra-config"`                // 高级配置
	Username        string `mapstructure:"username" json:"username" yaml:"username"`                            // 数据库用户名
	Password        string `mapstructure:"password" json:"password" yaml:"password"`                            // 数据库密码
	Dbname          string `mapstructure:"db-name" json:"db-name" yaml:"db-name"`                               // 数据库名
	MaxIdleConns    int    `mapstructure:"max-idle-conns" json:"maxIdleConns" yaml:"max-idle-conns"`            // 空闲中的最大连接数
	MaxOpenConns    int    `mapstructure:"max-open-conns" json:"maxOpenConns" yaml:"max-open-conns"`            // 打开到数据库的最大连接数
	ConnMaxLifetime int    `mapstructure:"conn-max-lifetime" json:"conn-max-lifetime" yaml:"conn-max-lifetime"` // 连接的最大生存时间
	ConnMaxIdletime int    `mapstructure:"conn-max-idletime" json:"conn-max-idletime" yaml:"conn-max-idletime"` // 连接的最大空闲时间
}

func (m *MysqlDBConfig) Validate() error {
	if m.Enable {
		cpun := cm.GOsInfo.OsMaxProcessorCount
		if m.MaxIdleConns <= 0 {
			m.MaxIdleConns = cpun / 2
			//return errors.New("database-config.mysql8.max-idle-conns >= 1")
		}
		if m.MaxOpenConns <= 0 {
			m.MaxOpenConns = cpun*2 - 1
			//return errors.New("database-config.mysql8.max-open-conns >= 1")
		}
		if m.MaxIdleConns > m.MaxOpenConns {
			return errors.New("database-config.mysql8: max-idle-conns >= max-open-conns")
		}

		if m.Dbname == "" {
			return errors.New("database-config.mysql8: db-name=='' ")
		}

		if m.IpAddress == "" {
			return errors.New("database-config.mysql8: ip-address=='' ")
		}

		if m.IpPort == "" {
			return errors.New("database-config.mysql8: ip-port=='' ")
		}

		if m.Username == "" {
			return errors.New("database-config.mysql8: username=='' ")
		}

		if m.Password == "" {
			return errors.New("database-config.mysql8: Password=='' ")
		}
	}
	return nil
}

func (m *MysqlDBConfig) Dsn() string {
	return m.Username + ":" + m.Password + "@tcp(" + m.IpAddress + ":" + m.IpPort + ")/" + m.Dbname + "?" + m.ExtraConfig
}
