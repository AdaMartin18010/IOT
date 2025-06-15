package gezhouba

import (
	"testing"

	"github.com/spf13/viper"
)

const testConfigPath = "./"

func TestReadconfig(t *testing.T) {

	viper.SetConfigName(gConfigFileName) // name of config file (without extension)
	viper.SetConfigType(gConfigFileType) // REQUIRED if the config file does not have the extension in the name
	viper.AddConfigPath(testConfigPath)  // optionally look for config in the working directory
	if err := viper.ReadInConfig(); err != nil {
		// Find and read the config file
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// Config file not found; ignore error if desired
			t.Errorf("Fatal error config file: %+v \n", err)
		} else {
			// Config file was found but another error was produced
			t.Errorf("readconfig err:%+v\n", err)
		}
	}

	var glogconf LogConf
	viper.UnmarshalKey("log-property", &glogconf)
	t.Logf("config: %+v\n", glogconf)
}
