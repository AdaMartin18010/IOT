package gezhouba

import (
	"fmt"
	"os"

	cm "navigate/common"
	mdl "navigate/common/model"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

const defaultConfigFile = "navigate-model"

var (
	gConfigFileName = "navigate-model"
	gConfigFileType = "yaml"
	gConfigPath     = "./conf/"
	GVerbose        = false
	gVersionShow    = false
)

func InitViper() {
	pflag.StringP("configFile", "c", "", "choose config file. navigate by default.")
	pflag.BoolVarP(&GVerbose, "Verbose enable", "e", true, "Verbose help message for enable")
	pflag.BoolVarP(&gVersionShow, "Version", "v", false, "Version Show version information for app")
	pflag.Parse()
	if gVersionShow {
		fmt.Println(Version())
		os.Exit(0)
	}
	// 优先级: 命令行 > 环境变量 > 默认值
	gViper = viper.New()
	gViper.BindPFlags(pflag.CommandLine)
	// Gviper.SetEnvPrefix("navigate")
	// Gviper.BindEnv("configFile")

	gConfigFileName = gViper.GetString("configFile")
	if gConfigFileName == "" {
		gConfigFileName = defaultConfigFile
	}
	gViper.SetConfigName(gConfigFileName) // name of config file (without extension)
	gViper.SetConfigType(gConfigFileType) // REQUIRED if the config file does not have the extension in the name

	fs, err := cm.DealWithExecutedCurrentFilePath(gConfigPath)
	if err != nil {
		mdl.L.Sugar().Fatalf("Fatal error config file: %+v \n", err)
	} else {
		gConfigPath = fs
		mdl.L.Sugar().Infof("configure path:%s", gConfigPath)
	}

	gViper.AddConfigPath(gConfigPath) // optionally look for config in the working directory
	if err := gViper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			mdl.L.Sugar().Fatalf("Fatal error config file: %+v \n", err)
		} else {
			mdl.L.Sugar().Fatalf("readconfig err:%+v\n", err)
		}
	}
	mdl.L.Sugar().Infoln("read configure OK!")
}
