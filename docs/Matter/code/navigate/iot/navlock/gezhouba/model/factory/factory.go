package factory

import (
	"fmt"
	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	"navigate/iot/navlock/model"
	"time"
)

var (
	L = cm.L
)

const (
	SystemName  = "船闸测速"
	NavlockKind = cmpt.KindName("Navlock")
)

/*
ServerMaster:
	--- navlock:{
	    id:[#1,#2,#3]
		components:{
			stopline:{
				id:[stopline]
				component: httpdr,
			},
			navlstatus:{
				id:[navlstatus]
				component: httpdr,
			},
			platform:{
				id:[platform]
				component: httpdr,
			},
			palyscreen:{
				id:[palyscreen]
				component: httpdr,
			},
			shipspeed:{
				id:[shipspeed,shipspeed-up,shipspeed-down]
				component: shipspeed,
				components:{
					up:{
						id:[shipspeed-up]
						component: sensors,
						components:{
							left:sensors,
							right:sensors
						}
					},
				 	down:{
						id:[shipspeed-down]
						component: sensors,
						components:{
							left:sensors,
							right:sensors
						}
					},
				}
			}
		}
	}
*/

func CreateNavLockModel(root cmpt.CptComposite, conf *gzb_cf.Navigationlocks) {
	L.Sugar().Infof("Creating NavLockModel ---------------------------begin-------------------------------------")
	if conf.NavlockConf01 != nil && conf.NavlockConf01.Enable {
		NavLockModelCreateComponentsFromConfig(root, conf.NavlockConf01)
	}

	if conf.NavlockConf02 != nil && conf.NavlockConf02.Enable {
		NavLockModelCreateComponentsFromConfig(root, conf.NavlockConf02)
	}

	if conf.NavlockConf03 != nil && conf.NavlockConf03.Enable {
		NavLockModelCreateComponentsFromConfig(root, conf.NavlockConf03)
	}
	L.Sugar().Infof("Created NavLockModel ---------------------------End-------------------------------------")
}

func NavLockModelCreateComponentsFromConfig(root cmpt.CptComposite, conf *gzb_cf.NavlockConf) (navlock *Navlock) {
	if conf == nil {
		L.Sugar().Infof("gzb_cf.NavlockConf is nil")
		return
	}

	if conf.Enable {
		//根据全局设置EnableAutoTest
		if !conf.EnableAutoTest {
			conf.NavlockShipSpeedConf.AutoTest(false)
			conf.NavlStatusHttp.AutoTest(false)
			conf.StopLineHttp.AutoTest(false)
			conf.PlatformHttp.AutoTest(false)
		}
	}
	navlId := fmt.Sprintf("#%d", conf.Identifier)
	mcs := model.NewCptSt(NavlockKind, cmpt.IdName(navlId), 1, root.Ctrl().ForkCtxWg())
	mss := model.NewModelSysSt(mcs.CmptInfo(), SystemName+navlId,
		"", string(NavlockKind), "")

	trks := gzb_sche.NewTrackingSt(conf.Identifier, time.Duration(conf.NavStateInterval), conf.EnableAnimation)
	trks.EnableDebug = conf.EnableDebug
	trks.NavlockName = conf.Name
	trks.NavlockType = conf.NavlType
	trks.NavlockLength = conf.Length
	trks.NavlockWidth = conf.Width

	trks.SensorsLayout = conf.SensorsLayout
	trks.DetectionDistance = conf.DetectionDistanceMax
	trks.DetectionSpeedMax = conf.DetectionSpeedMax
	trks.TargetAccelerationMax = conf.TargetAccelerationMax

	navlock = NewNavlockSvr(mcs, mss, trks)
	navlock.NavlockConf = conf
	navlock.InitComponentsFromConfig(conf)
	root.AddCpts(navlock)
	L.Sugar().Infof("Created NavLockModel#%d", navlock.NavlockIdentifier)

	//--------------- 创建基础的事件发布消息队列 -------------------------
	L.Sugar().Infof("-------- EventTopics: %s ", navlock.EventChans.Topics())
	//----------------------------------------------------------------

	return
}
