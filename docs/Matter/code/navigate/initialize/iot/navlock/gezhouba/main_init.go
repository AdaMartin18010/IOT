package gezhouba

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	cm "navigate/common"
	mdl "navigate/common/model"
	gzbcf "navigate/config/iot/navlock/gezhouba"
	g "navigate/global"

	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var (
	gViper              *viper.Viper
	GNavConfig_gezhouba gzbcf.Navigationlocks
)

func init() {
	//创建监听退出chan
	gchan := make(chan os.Signal, 1)
	//监听系统信号
	signal.Notify(gchan, syscall.SIGHUP, syscall.SIGINT,
		syscall.SIGTERM, syscall.SIGQUIT, syscall.SIGPIPE, syscall.SIGABRT,
		syscall.SIGALRM, syscall.SIGTRAP, syscall.SIGBUS, syscall.SIGSEGV,
		syscall.SIGINT, syscall.SIGILL)

	if g.MainCtr == nil {
		g.MainCtr = mdl.NewCtrlSt(context.Background())
	}

	if g.DBCtr == nil {
		g.DBCtr = mdl.NewCtrlSt(context.Background())
	}

	//创建系统信号监听 用户主动取消调用全局取消函数
	go func() {
		for s := range gchan {
			L.Sugar().Infof("Catched system signal:%s", s)
			switch s {
			//signal ctrl+c kill
			case syscall.SIGINT, syscall.SIGTERM:
				g.MainCtr.Cancel()
				return
			}
		}
	}()

	InitServerMaster(g.MainCtr.ForkCtxWg())

	g.ExecutingCurrentFilePath = cm.ExecutingCurrentFilePath()
	L.Sugar().Infof("ExecutingCurrentFilePath: %+v", g.ExecutingCurrentFilePath)
	g.CompiledExectionFilePath = cm.CompiledExectionFilePath()
	L.Sugar().Infof("CompiledExectionFilePath: %+v", g.CompiledExectionFilePath)
	var err error
	g.ExecutedCurrentFilePath, err = cm.ExecutedCurrentFilePath()
	if err != nil {
		L.Sugar().Fatalf("ExecutedCurrentFilePath: err: %+v\n", err)
	}
	L.Sugar().Infof("ExecutedCurrentFilePath: %+v\n", g.ExecutedCurrentFilePath)
}

// 系统读取命令和配置文件  初始化数据库
func SystemInitialization() {
	err := SetUpConfig(GVerbose)
	if err != nil {
		os.Exit(0)
	}
	InitDatabase()
}

func CloseDBs() {
	if g.LocalDB != nil {
		defer func() {
			if db, err := g.LocalDB.DB(); err != nil {
				L.Sugar().Error(err)
			} else {
				if err = db.Close(); err != nil {
					L.Sugar().Error(err)
				}
			}
			L.Sugar().Info("local db exited")
		}()
	}

	if g.ProduceDB != nil {
		defer func() {
			if db, err := g.ProduceDB.DB(); err != nil {
				L.Sugar().Error(err)
			} else {
				if err = db.Close(); err != nil {
					L.Sugar().Error(err)
				}
			}
			L.Sugar().Info("produce db exited")
		}()
	}

	if g.ProduceWebDB != nil {
		defer func() {
			if db, err := g.ProduceWebDB.DB(); err != nil {
				L.Sugar().Error(err)
			} else {
				if err = db.Close(); err != nil {
					L.Sugar().Error(err)
				}
			}
			L.Sugar().Info("produce web db exited")
		}()
	}
}

// 所有的服务最终结束 由系统主动调用该函数来处理最后的工作
func ServerFinalization() {
	L.Info(".....Navlock Main  exiting .....")
	defer L.Info(".....Navlock Main  exited .....")

	timeWait := time.Duration(300) * time.Millisecond
	timer := time.NewTimer(timeWait)
	defer timer.Stop()
	mainExited := true
	ServerMasterExited := false
	DBExited := false
	for {
		select {
		case <-g.MainCtr.Context().Done():
			{
				if !mainExited {
					continue
				}

				mainExited = false
				if err := g.MainCtr.Context().Err(); err != context.Canceled {
					L.Sugar().Infof("main Exit get context err: %+v", err)
				}

				L.Info("----- ServerMaster  Stopping -----")
				GServerMaster.Ctrl().Cancel()
				ServerMasterExited = true
			}
		case <-GServerMaster.Ctrl().Context().Done():
			{
				if !ServerMasterExited {
					continue
				}

				if GServerMaster.IsRunning() {
					if err := GServerMaster.Stop(); err != nil {
						L.Sugar().Infof("ServerMaster  Stopping get err: %+v", err)
					}
				}

				L.Info("----- ServerMaster  waiting -----")
				//time.Sleep(1 * time.Second)
				GServerMaster.Ctrl().WaitGroup().WaitAsync()
				ServerMasterExited = false
				L.Info("----- ServerMaster  Stopped -----")

				L.Info("---- Base component Stopping ----")
				//最后再取消数据库的上下文
				g.DBCtr.Cancel()
				DBExited = true
			}
		case <-g.DBCtr.Context().Done():
			{
				if !DBExited {
					continue
				}
				if err := g.DBCtr.Context().Err(); err != context.Canceled && err != context.DeadlineExceeded {
					L.Sugar().Warnf("DB cancel: %+v", err)
				}
				g.DBCtr.WaitGroup().WaitAsync()
				CloseDBs()
				L.Info("---- Base component Stopped ----")
				DBExited = false
				goto exit
			}
		case <-timer.C:
			timer.Reset(timeWait)
			L.Info("..... main  exiting ..... ..... .....")
		}
	}

exit:
	g.MainCtr.WaitGroup().WaitAsync()
}

// 读取配置文件 检查基本的有效性
func SetUpConfig(verbose bool) error {
	InitLogConfig()

	l := InitLogger()
	zap.ReplaceGlobals(l)
	mdl.L = l

	if err := gViper.UnmarshalKey("navigationlock-config", &GNavConfig_gezhouba); err != nil {
		L.Sugar().Panicf("Read Config to struct err:%+v", err)
		return err
	}

	if verbose {
		L.Sugar().Infof("config loaded: %#v\n", GNavConfig_gezhouba)
	}

	if verbose {
		L.Sugar().Infof("NavigationlockConf01: %+v", GNavConfig_gezhouba.NavlockConf01)
	}
	err := GNavConfig_gezhouba.NavlockConf01.Validate()
	if err != nil {
		L.Sugar().Infof("NavigationlockConf01: not Valid: %+v", err.Error())
		return err
	}

	if verbose {
		L.Sugar().Infof("NavigationlockConf02: %+v", GNavConfig_gezhouba.NavlockConf02)
	}

	err = GNavConfig_gezhouba.NavlockConf02.Validate()
	if err != nil {
		L.Sugar().Infof("NavigationlockConf02: not Valid: %+v", err.Error())
		return err
	}

	if verbose {
		L.Sugar().Infof("NavigationlockConf03: %+v", GNavConfig_gezhouba.NavlockConf03)
	}

	err = GNavConfig_gezhouba.NavlockConf03.Validate()
	if err != nil {
		L.Sugar().Infof("NavigationlockConf03: not Valid: %+v", err.Error())
		return err
	}

	InitDBconfig()
	if verbose {
		L.Sugar().Infof("database-config: %+v", GDatabaseConf)
	}
	return nil
}
