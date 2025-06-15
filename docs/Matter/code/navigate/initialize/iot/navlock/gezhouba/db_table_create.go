package gezhouba

import (
	"runtime"
	"time"

	db_cf "navigate/config/database"
	g "navigate/global"
	db_gzb "navigate/store/iot/navlock/gezhouba"
	db_web "navigate/store/iot/navlock/gezhouba/web"
	db_navlock "navigate/store/iot/navlock/model"

	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	gormlog "gorm.io/gorm/logger"
	zapgorm "moul.io/zapgorm2"
)

var (
	GDatabaseConf db_cf.DBConfig

	databaseConfigTag = "database-config"
)

// 自动生成 系统track表 通航业务表  web表对应调试
func RegisterSystemTrackTables(db *gorm.DB) {
	err := db.AutoMigrate(
		// 系统模块表
		db_navlock.IotNavlCpntStatus{},
		db_navlock.IotNavlCpntState{},
		db_navlock.IotNavStatus{},
		db_navlock.IotNavlStatus{},
		db_navlock.IotNavlSpeedInfo{},
		db_navlock.IotNavlDevicesStatus{},
		// TODO: Database adding
	)
	if err != nil {
		L.Sugar().Fatalf("Database initialization failed: %s", err)
	}
	L.Info("register System Track Tables success")

}

// 自动生成 通航业务相关的表
func RegisterNavigationTables(db *gorm.DB) {
	err := db.AutoMigrate(
		// 通航业务相关的表
		//禁停线告警表
		db_gzb.NavlGzbStopLineWarns{},
		//船舶超速表
		//db_gzb.NavlGzbShipSpeed{},
		//同步平台 超速表
		db_gzb.NavlGzbSpeedLimit{},
		//db_gzb.NavlGzbStopLine{},

		// TODO: Database adding
	)
	if err != nil {
		L.Sugar().Fatalf("Database initialization failed: ", err)
	}
	L.Info("register Navigation Tables success")

}

func RegisterNavigationWebTables(db *gorm.DB) {
	err := db.AutoMigrate(
		// ********* web数据表 ********
		db_web.NavAnimationRecord{},
		db_web.NavAnimationHistory{},
		// 禁停线 统计表
		db_web.NavStopLineWarn{},
		db_web.NavStopLineStatis{},
		// 设置表
		db_web.NavPlayScreenSetup{},
		db_web.NavShipSpeedSetup{},
		// 速度表 超速表  统计表
		db_web.NavSpeedCurve{},
		db_web.NavSpeedRaw{},
		db_web.NavSpeedStatis{},
		// TODO: Database adding
	)
	if err != nil {
		L.Sugar().Fatalf("Database initialization failed: ", err)
	}
	L.Info("register Navigation Tables success")

}

// RegisterTables 注册需要的数据库表专用
func RegisterTables(db *gorm.DB) {
	RegisterSystemTrackTables(db)
	RegisterNavigationTables(db)
}

// 初始化程序的数据库
func InitDatabase() {

	// if GDatabaseConf.Sqlite3Conf.Enable && GDatabaseConf.Mysql8Conf.Enable {
	// 	Glog.Sugar().Fatal("sqlite and mysql do not both enable!")
	// 	return
	// }

	if GDatabaseConf.Mysql8Conf.Enable {
		dblog := L.With(
			zap.Namespace("component"),
			zap.String("db", "mysql"),
		)
		gormconf := InitGormConf(dblog, GDatabaseConf.DBBaseConf)
		g.ProduceDB = GormMysqlByConfig(GDatabaseConf.Mysql8Conf, gormconf).WithContext(g.DBCtr.Context())
		if g.ProduceDB != nil && GDatabaseConf.Mysql8Conf.AutoMigrate {
			RegisterTables(g.ProduceDB)
		}
		L.Sugar().Debugf("Database Connected && Created --- Mysql8+")
	}

	if GDatabaseConf.Mysql8WebConf.Enable {
		dblog := L.With(
			zap.Namespace("component"),
			zap.String("db", "webdb"),
		)
		gormconf := InitGormConf(dblog, GDatabaseConf.DBBaseConf)
		g.ProduceWebDB = GormMysqlByConfig(GDatabaseConf.Mysql8WebConf, gormconf).WithContext(g.DBCtr.Context())
		// 需要注册web的表数据库 直接访问web的表
		if g.ProduceDB != nil && GDatabaseConf.Mysql8Conf.AutoMigrate {
			RegisterNavigationWebTables(g.ProduceWebDB)
		}
		L.Sugar().Infof("Database Connected  --- Mysql8 Web")
	}

	if GDatabaseConf.Sqlite3Conf.Enable {
		dblog := L.With(
			zap.Namespace("component"),
			zap.String("db", "sqlite"),
		)
		gormconf := InitGormConf(dblog, GDatabaseConf.DBBaseConf)
		g.LocalDB = GormSqliteByConfig(&GDatabaseConf.Sqlite3Conf, gormconf).WithContext(g.DBCtr.Context())
		if g.LocalDB != nil && GDatabaseConf.Sqlite3Conf.AutoMigrate {
			RegisterTables(g.LocalDB)
			RegisterNavigationWebTables(g.LocalDB)
		}
		L.Sugar().Infof("Database Connected && Created  --- Sqlite3+")
	}

	if g.LocalDB == nil && g.ProduceDB == nil && g.ProduceWebDB == nil {
		L.Fatal("No database Created --- Server Exited!")
	}

}

// 读取初始化数据库配置
func InitDBconfig() error {
	if err := gViper.UnmarshalKey(databaseConfigTag, &GDatabaseConf); err != nil {
		L.Sugar().Fatalf("Read %s Config to struct err:%+v\n", databaseConfigTag, err)
		return err
	}

	if err := GDatabaseConf.DBBaseConf.Validate(); err != nil {
		L.Sugar().Fatalf("Read %s Config to struct err:%+v\n", databaseConfigTag, err)
		return err
	}

	if err := GDatabaseConf.Mysql8Conf.Validate(); err != nil {
		L.Sugar().Fatalf("Read %s Config to struct err:%+v\n", databaseConfigTag, err)
		return err
	}

	if err := GDatabaseConf.Sqlite3Conf.Validate(); err != nil {
		L.Sugar().Fatalf("Read %s Config to struct err:%+v\n", databaseConfigTag, err)
		return err
	}

	// if gDatabaseConf.Mysql8Conf.EnableTestcase {
	// 	if !gDatabaseConf.Sqlite3Conf.Enable {
	// 		g.Glog.Sugar().Fatalf("Read %s Config : if Mysql enable_testcase is true , sqlite enable must be true!\n", databaseConfigTag)
	// 	}
	// }
	return nil
}

func gormLogLevel(mode string) gormlog.LogLevel {
	switch mode {
	case "silent":
		return gormlog.Silent
	case "error":
		return gormlog.Error
	case "warn":
		return gormlog.Warn
	case "info":
		return gormlog.Info
	}
	return gormlog.Info
}

func InitGormConf(loger *zap.Logger, conf db_cf.DatabaseBaseConfig) (config *gorm.Config) {
	config = &gorm.Config{
		DisableForeignKeyConstraintWhenMigrating: true,
	}
	if !conf.LogEnable {
		return
	} else {
		logger := zapgorm.New(loger)
		logger.SkipCallerLookup = conf.SkipCallerLookup
		logger.IgnoreRecordNotFoundError = conf.IgnoreRecordNotFoundError
		logger.SlowThreshold = time.Duration(conf.SlowSqlThreshold)
		logger.LogLevel = gormLogLevel(conf.LogMode)
		logger.SetAsDefault()
		config.Logger = logger
	}

	return
}

// GormMysqlByConfig 初始化Mysql数据库用传入配置
func GormMysqlByConfig(mysqlc db_cf.MysqlDBConfig, gconfig *gorm.Config) *gorm.DB {
	if !mysqlc.Enable {
		return nil
	}

	mysqlConfig := mysql.Config{
		DSN:                       mysqlc.Dsn(), // DSN data source name
		DefaultStringSize:         256,          // string 类型字段的默认长度
		DontSupportRenameIndex:    true,         // 重命名索引时采用删除并新建的方式，MySQL 5.7 之前的数据库和 MariaDB 不支持重命名索引
		DontSupportRenameColumn:   true,         // 用 `change` 重命名列，MySQL 8 之前的数据库和 MariaDB 不支持重命名列
		SkipInitializeWithVersion: false,        // 根据版本自动配置
	}

	if db, err := gorm.Open(mysql.New(mysqlConfig), gconfig); err != nil {
		//在开启生产数据库的情况下 如果出现错误就直接退出
		L.Sugar().Fatalf("Connecting mysql-database: %+v", err)
	} else {
		sqlDB, _ := db.DB()
		cpun := runtime.NumCPU()
		//重新检查配置情况
		sqlDB.SetMaxOpenConns(mysqlc.MaxOpenConns)
		if mysqlc.MaxOpenConns < cpun {
			sqlDB.SetMaxOpenConns(cpun)
		}
		sqlDB.SetMaxIdleConns(mysqlc.MaxIdleConns)
		if mysqlc.MaxIdleConns < cpun {
			sqlDB.SetMaxIdleConns(cpun)
		}
		sqlDB.SetConnMaxLifetime(time.Duration(mysqlc.ConnMaxLifetime) * time.Minute)
		sqlDB.SetConnMaxIdleTime(time.Duration(mysqlc.ConnMaxIdletime) * time.Minute)
		return db
	}

	return nil
}

// GormSqliteByConfig
// @Description: 初始化sqlite数据库用传入的配置
// @param sqlitec sqlite配置文件
// @param gconfig gorm.Config
func GormSqliteByConfig(sqlitec *db_cf.SqliteConfig, gconfig *gorm.Config) *gorm.DB {
	if !sqlitec.Enable {
		return nil
	}

	if db, err := gorm.Open(sqlite.Open(sqlitec.Dsn()), gconfig); err != nil {
		L.Sugar().Fatalf("error creating sqlite database: %+v", err)
	} else {
		// 启用 WAL 模式
		_ = db.Exec("PRAGMA journal_mode=WAL;")
		//_ = db.Exec("PRAGMA journal_size_limit=104857600;")
		//_ = db.Exec("PRAGMA busy_timeout=999999;")

		sqlDB, _ := db.DB()
		cpun := runtime.NumCPU()
		sqlDB.SetMaxIdleConns(cpun)
		sqlDB.SetMaxOpenConns(cpun)
		return db
	}

	return nil
}
