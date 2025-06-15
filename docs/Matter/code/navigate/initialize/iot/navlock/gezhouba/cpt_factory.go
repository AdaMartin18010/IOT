package gezhouba

import (
	gzb_md "navigate/iot/navlock/gezhouba/model/factory"
)

const (
	NavlockShipSpeedMeasurementSystem = "船闸测速"
)

func CreateAndAddComponents() {
	gzb_md.CreateNavLockModel(GServerMaster, &GNavConfig_gezhouba)
}
