package database

import (
	"errors"
)

type DBConfig struct {
	DBBaseConf    DatabaseBaseConfig `mapstructure:"db-property" json:"db-property" yaml:"db-property"`
	Sqlite3Conf   SqliteConfig       `mapstructure:"sqlite" json:"sqlite" yaml:"sqlite"`
	Mysql8Conf    MysqlDBConfig      `mapstructure:"mysql8" json:"mysql8" yaml:"mysql8"`
	Mysql8WebConf MysqlDBConfig      `mapstructure:"mysql8-web" json:"mysql8-web" yaml:"mysql8-web"`
}

// 基本的GORM参数配置
type DatabaseBaseConfig struct {
	LogEnable                 bool   `mapstructure:"log-enable" json:"log-enable" yaml:"log-enable"`                                                          // 是否开启Gorm全局日志
	LogMode                   string `mapstructure:"log-mode" json:"log-mode" yaml:"log-mode"`                                                                // Gorm全局日志模式
	SlowSqlThreshold          uint   `mapstructure:"slow-sql-threshold" json:"slow-sql-threshold" yaml:"slow-sql-threshold"`                                  // 慢sql的阈值 毫秒
	SkipCallerLookup          bool   `mapstructure:"skip-caller-lookup" json:"skip-caller-lookup" yaml:"slow-sql-threshold"`                                  // 是否跳过调用函数查找
	IgnoreRecordNotFoundError bool   `mapstructure:"ignore-record-not-found-error" json:"ignore-record-not-found-error" yaml:"ignore-record-not-found-error"` // 是否忽略没有找到记录的常规错误
}

func isValidLogMode(mode string) bool {
	switch mode {
	case "silent", "error", "warn", "info":
		return true
	}
	return false
}

func (db *DatabaseBaseConfig) Validate() error {
	if db.LogEnable {
		if !isValidLogMode(db.LogMode) {
			return errors.New("database-config.db-property.log-mode: invalid Logmode- not in [silent,error,warn,info]")
		}
		if db.SlowSqlThreshold <= 200 {
			db.SlowSqlThreshold = 200
		}
	}
	return nil
}
