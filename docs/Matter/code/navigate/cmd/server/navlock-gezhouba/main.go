package main

import (
	mdl "navigate/common/model"
	gzb_init "navigate/initialize/iot/navlock/gezhouba"
)

func main() {
	defer mdl.L.Sync()
	gzb_init.GServerMaster.Start()
	gzb_init.GServerMaster.Finalize()
}
