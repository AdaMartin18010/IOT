package dbtest

import (
	db_cf "navigate/config/database"
	gzb_init "navigate/initialize/iot/navlock/gezhouba"
)

func init() {

	//sql日志分流模式  silent=1  error=2  warn=3  info=4 数值大于日志配置的level就会输出
	// 如: info > log-property日志配置的error 则程序中数据库操作记录, 小于panic都会输出到程序的默认日志中
	dbBaseConf := db_cf.DatabaseBaseConfig{
		LogEnable: true,
		LogMode:   "silent",
		//慢sql阈值 毫秒 -- 小于200则被重置为默认200ms
		SlowSqlThreshold: 1000,
		//跳过sql调用函数的查找
		SkipCallerLookup: true,
		//忽略没有记录的常规错误
		IgnoreRecordNotFoundError: true,
	}

	gzb_init.GDatabaseConf = db_cf.DBConfig{}
	gzb_init.GDatabaseConf.DBBaseConf = dbBaseConf

	// sqliteConf := db_cf.SqliteConfig{
	// 	Enable:      true,
	// 	AutoMigrate: true,
	// 	Filepath:    "./localdb.db",
	// 	ExtraConfig: "?mode=memory&cache=shared",
	// }

	// mysql8Conf := db_cf.MysqlDBConfig{
	// 	Enable:          true,
	// 	AutoMigrate:     true,
	// 	IpAddress:       "127.0.0.1",
	// 	IpPort:          "3306",
	// 	ExtraConfig:     "charset=utf8mb4&parseTime=True&loc=Local",
	// 	Username:        "root",
	// 	Password:        "d123456b",
	// 	Dbname:          "navigation",
	// 	MaxIdleConns:    0,
	// 	MaxOpenConns:    0,
	// 	ConnMaxLifetime: 0,
	// 	ConnMaxIdletime: 0,
	// }

	// mysql8WebConf := db_cf.MysqlDBConfig{
	// 	Enable:          true,
	// 	AutoMigrate:     false,
	// 	IpAddress:       "127.0.0.1",
	// 	IpPort:          "3306",
	// 	ExtraConfig:     "charset=utf8mb4&parseTime=True&loc=Local",
	// 	Username:        "root",
	// 	Password:        "d123456b",
	// 	Dbname:          "gvaweb",
	// 	MaxIdleConns:    0,
	// 	MaxOpenConns:    0,
	// 	ConnMaxLifetime: 0,
	// 	ConnMaxIdletime: 0,
	// }

	// gzb_init.GDatabaseConf = db_cf.DBConfig{
	// 	DBBaseConf:    dbBaseConf,
	// 	Sqlite3Conf:   sqliteConf,
	// 	Mysql8Conf:    mysql8Conf,
	// 	Mysql8WebConf: mysql8WebConf,
	// }

	InitLocalDB()
	InitMysqlDB()
	InitMysqlWebDB()
	gzb_init.InitDatabase()
}

func InitLocalDB() {
	sqliteConf := db_cf.SqliteConfig{
		Enable:      true,
		AutoMigrate: true,
		Filepath:    "./localdb.db",
		ExtraConfig: "?mode=memory&cache=shared&mode=wal",
	}

	gzb_init.GDatabaseConf.Sqlite3Conf = sqliteConf
}

func InitMysqlDB() {

	mysql8Conf := db_cf.MysqlDBConfig{
		Enable:          true,
		AutoMigrate:     true,
		IpAddress:       "127.0.0.1",
		IpPort:          "3306",
		ExtraConfig:     "charset=utf8mb4&parseTime=True&loc=Local",
		Username:        "root",
		Password:        "d123456b",
		Dbname:          "navigation",
		MaxIdleConns:    0,
		MaxOpenConns:    0,
		ConnMaxLifetime: 0,
		ConnMaxIdletime: 0,
	}
	gzb_init.GDatabaseConf.Mysql8Conf = mysql8Conf
}

func InitMysqlWebDB() {
	mysql8WebConf := db_cf.MysqlDBConfig{
		Enable:          true,
		AutoMigrate:     false,
		IpAddress:       "127.0.0.1",
		IpPort:          "3306",
		ExtraConfig:     "charset=utf8mb4&parseTime=True&loc=Local",
		Username:        "root",
		Password:        "d123456b",
		Dbname:          "gvaweb",
		MaxIdleConns:    0,
		MaxOpenConns:    0,
		ConnMaxLifetime: 0,
		ConnMaxIdletime: 0,
	}
	gzb_init.GDatabaseConf.Mysql8WebConf = mysql8WebConf
}
