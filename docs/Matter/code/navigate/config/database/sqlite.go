package database

import (
	"errors"
	cm "navigate/common"
)

type SqliteConfig struct {
	Enable      bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	AutoMigrate bool   `mapstructure:"auto-migrate" json:"auto-migrate" yaml:"auto-migrate"` // 是否自动创建数据库表 在表不存在的情况下
	Filepath    string `mapstructure:"filepath" json:"filepath" yaml:"filepath"`
	ExtraConfig string `mapstructure:"extra-config" json:"extra-config" yaml:"extra-config"`
}

func (s *SqliteConfig) Validate() error {
	if s.Filepath == "" {
		return errors.New("database-config.sqlite: filepath=='' ")
	}

	s.Filepath = cm.DealWithExecutingCurrentFilePath(s.Filepath)
	//在启用的情况下 自动创建表
	s.AutoMigrate = true
	return nil
}

func (s *SqliteConfig) Dsn() string {
	return s.Filepath + s.ExtraConfig
}
