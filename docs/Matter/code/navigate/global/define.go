package global

import (
	mdl "navigate/common/model"

	"gorm.io/gorm"
)

var (
	// 项目编译后的可执行文件所在目录
	ExecutedCurrentFilePath string
	// 可执行文件实际执行时所在的目录 ./ 路径就是基于该目录
	ExecutingCurrentFilePath string
	// 项目源文件main.go的文件目录
	CompiledExectionFilePath string

	MainCtr      *mdl.CtrlSt = nil
	DBCtr        *mdl.CtrlSt = nil
	MQCtr        *mdl.CtrlSt = nil
	LocalDB      *gorm.DB    = nil
	ProduceDB    *gorm.DB    = nil
	ProduceWebDB *gorm.DB    = nil
)
