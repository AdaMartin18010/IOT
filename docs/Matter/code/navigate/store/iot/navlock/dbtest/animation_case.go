package dbtest

import (
	gzb_db_web_wrapper "navigate/store/iot/navlock/gezhouba/web/wrapper"
)

func AnimationRecordSave(navAnmWrap *gzb_db_web_wrapper.NavAnimationWrap) {
	navAnmWrap.SetScheduleIdAndNavlockStatus("TestScheduID", 2)
	navAnmWrap.SetGatesStatus(0, 0, 0, 1)
	navAnmWrap.SetUpStoplineInfo(0, 0, 0, 0)
	navAnmWrap.SetDownStoplineInfo(0, 1, 0, 1)
	navAnmWrap.SetUpSensorsInfo(1, 1)
	navAnmWrap.SetDownSensorsInfo(2, 2)
	navAnmWrap.SetLeftRadarInfo(1, 200, 1.7)
	navAnmWrap.SetRightRadarInfo(1, 240, 1.82)
}

func AnimationRecordSave0(navAnmWrap *gzb_db_web_wrapper.NavAnimationWrap) {
	navAnmWrap.SetScheduleIdAndNavlockStatus("TestScheduID", 2)
	navAnmWrap.SetGatesStatus(0, 0, 0, 0)

	navAnmWrap.SetUpStoplineInfo(0, 0, 0, 0)
	navAnmWrap.SetDownStoplineInfo(0, 0, 0, 0)

	navAnmWrap.SetUpSensorsInfo(1, 1)
	navAnmWrap.SetDownSensorsInfo(2, 2)

	navAnmWrap.SetLeftRadarInfo(1, 200, 1.7)
	navAnmWrap.SetRightRadarInfo(1, 240, 1.82)
}
