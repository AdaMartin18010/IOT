package playscreen

import (
	"fmt"
	"time"

	mdl "navigate/common/model"
	cmd_token "navigate/common/model/token"
	gzb_schedule "navigate/iot/navlock/gezhouba/schedule"
)

func (ps *PlayScreen) startProgramDefaultActive() {
	if !ps.httpActiveProgramDefault.IsRunning() {
		if ps.Cnf.EnableDebug {
			ps.httpActiveProgramDefault.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpActiveProgramDefault.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpActiveProgramDefault.Ctrl().Context())
		}

		ps.httpActiveProgramDefault.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

		if ps.Cnf.EnableDebug {
			ps.httpActiveProgramDefault.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpActiveProgramDefault.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpActiveProgramDefault.Ctrl().Context())
		}

		if err := ps.httpActiveProgramDefault.Start(); err != nil {
			ps.httpActiveProgramDefault.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpActiveProgramDefault.Info())
		}
	}
}

func (ps *PlayScreen) startProgramSpeedActive() {
	if !ps.httpActiveProgramSpeed.IsRunning() {
		if ps.Cnf.EnableDebug {
			ps.httpActiveProgramSpeed.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpActiveProgramSpeed.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpActiveProgramSpeed.Ctrl().Context())
		}

		ps.httpActiveProgramSpeed.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

		if ps.Cnf.EnableDebug {
			ps.httpActiveProgramSpeed.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpActiveProgramSpeed.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpActiveProgramSpeed.Ctrl().Context())
		}

		if err := ps.httpActiveProgramSpeed.Start(); err != nil {
			ps.httpActiveProgramSpeed.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpActiveProgramSpeed.Info())
		}
	}
}

func (ps *PlayScreen) checkingIfCreateDir() {

	navs := ps.TrackingSt.PredicateSchedule.GetNavStatus()
	navls := ps.TrackingSt.PredicateSchedule.GetNavlockStatus()
	switch navs {
	case gzb_schedule.NavUp, gzb_schedule.NavDown:
		{
			switch navls {
			case gzb_schedule.NavlockUpGoingIn, gzb_schedule.NavlockUpGoingOut, gzb_schedule.NavlockDownGoingIn, gzb_schedule.NavlockDownGoingOut:
				if ps.TrackingSt.ScheduleId.GetScheduleId() != "" {
					// 创建目录
					if err := ps.createFilePath(); err != nil {
						ps.Logger.Sugar().Errorf(`Create Dir err: %+v.%s`, err, ps.Info())
					}
					ps.createScreenShotFilePathName(fmt.Sprintf("Actived_%s_%d", ps.activeProgramType.String(), ps.rand.Intn(1000)))
					ps.ScreenShot()
				}
			}
		}
	}

}

// 需要函数可重入 可持续的检查 判断中继续余下的工作
func (ps *PlayScreen) checkingSwitchToProgramDefault() {
	if !ps.httpDoActiveProgramDefault {
		ps.startProgramDefaultActive()
		ps.httpDoActiveProgramDefault = true
		ps.httpDoActiveProgramDefaultReturnOK = false
	} else {
		// 开启了http 激活 然后判断返回值
		// 1. 如果返回值 设置成功了 就判断命令是否执行成功
		if ps.httpDoActiveProgramDefaultReturnOK {
			cmdDefaultOK := ps.CmdProgramWork(ps.programDefaultSetInfo.CmdSetOn, ps.wkProgramDefaultCmdChan)
			ps.programDefaultSetInfo.CurrentOn = ps.programDefaultStatus.IsActiced()
			ps.programDefaultSetInfo.NeedCmdSet = ps.programDefaultSetInfo.CurrentOn != ps.programDefaultSetInfo.CmdSetOn

			if cmdDefaultOK && !ps.programDefaultSetInfo.NeedCmdSet {
				if ps.Cnf.EnableDebug {
					ps.httpActiveProgramDefault.Logger.Sugar().Debugf(`Active default program : ok.%s`, ps.Info())
				}
				// 3. 命令执行成功了 就把返回值设置回初始值 http可再启动 节目当前状态设置成 默认节目
				ps.httpDoActiveProgramDefaultReturnOK = false
				ps.httpDoActiveProgramDefault = false
				// 都设置成功了 就设置当前激活的节目类型
				ps.activeProgramType = PrograDefault
				if ps.Cnf.EnableAutoTest {
					ps.checkingIfCreateDir()
				}
			}
		} else {
			// 2. 如果返回值 未设置成功 就设置http重复可再启动
			ps.httpDoActiveProgramDefault = false
		}
	}
}

// 需要函数可重入 可持续的检查 判断中继续余下的工作
func (ps *PlayScreen) checkingSwitchToProgramSpeed() {
	if !ps.httpDoActiveProgramSpeed {
		ps.startProgramSpeedActive()
		ps.httpDoActiveProgramSpeed = true
		ps.httpDoActiveProgramSpeedReturnOk = false
	} else {
		// 开启了http 激活 然后判断返回值
		// 1. 如果返回值 设置成功了 就判断命令是否执行成功
		if ps.httpDoActiveProgramSpeedReturnOk {
			cmdSpeedOK := ps.CmdProgramWork(ps.programSpeedSetInfo.CmdSetOn, ps.wkProgramSpeedCmdChan)
			ps.programSpeedSetInfo.CurrentOn = ps.programSpeedStatus.IsActiced()
			ps.programSpeedSetInfo.NeedCmdSet = ps.programSpeedSetInfo.CurrentOn != ps.programSpeedSetInfo.CmdSetOn

			if cmdSpeedOK && !ps.programSpeedSetInfo.NeedCmdSet {
				if ps.Cnf.EnableDebug {
					ps.httpActiveProgramSpeed.Logger.Sugar().Debugf(`Active Speed program : ok.%s`, ps.Info())
				}
				// 3. 命令执行成功了 就把返回值设置回初始值 http可再启动 节目当前状态设置成 显速节目
				ps.httpDoActiveProgramSpeedReturnOk = false
				ps.httpDoActiveProgramSpeed = false
				// 都设置成功了 就设置当前激活的节目类型
				ps.activeProgramType = ProgramSpeed
				if ps.Cnf.EnableAutoTest {
					ps.checkingIfCreateDir()
				}
			}
		} else {
			// 2.如果返回值 未设置成功 就设置http重复可再启动
			ps.httpDoActiveProgramSpeed = false
		}
	}
}

// 判断节目的切换 设置节目的切换状态
func (ps *PlayScreen) PredicateProgramSwitch() {
	programNormalOnOrOff := false
	programSpeedOnOrOff := false

	// 1. 如果为雷达测速阶段- 上下行进出闸 切换到 SpeedProgram
	// 2. 如果为非雷达测速阶段- 上下行进闸完成等 切换到NormalProgram
	navs := ps.TrackingSt.PredicateSchedule.GetNavStatus()
	navls := ps.TrackingSt.PredicateSchedule.GetNavlockStatus()

	if ps.Cnf.EnableDebug {
		ps.Logger.Sugar().Debugf(`PredicateSwitchInterval navl: [%s],[%s].%s`, navs.String(), navls.String(), ps.Info())
	}
	switch navs {
	case gzb_schedule.NavUp:
		switch navls {
		case gzb_schedule.NavlockUpGoingIn, gzb_schedule.NavlockUpGoingOut:
			{
				programNormalOnOrOff = false
				programSpeedOnOrOff = true
			}
		default:
			{
				programNormalOnOrOff = true
				programSpeedOnOrOff = false
			}
		}
	case gzb_schedule.NavDown:
		switch navls {
		case gzb_schedule.NavlockDownGoingIn, gzb_schedule.NavlockDownGoingOut:
			{
				programNormalOnOrOff = false
				programSpeedOnOrOff = true
			}
		default:
			{
				programNormalOnOrOff = true
				programSpeedOnOrOff = false
			}
		}
	default:
		{
			programNormalOnOrOff = true
			programSpeedOnOrOff = false
		}
	}

	//推断需要开启的时候 只判断当前应该设置的值 而不关注其他的状态
	ps.programDefaultSetInfo.CmdSetOn = programNormalOnOrOff
	ps.programSpeedSetInfo.CmdSetOn = programSpeedOnOrOff
}

func (ps *PlayScreen) CmdProgramWork(setActived bool, cmdChan chan<- cmd_token.TokenCompletor) bool {
	Cmd := NewProgramCmd()
	Cmd.SetONOrOFF = setActived
	if !ps.Commander(Cmd, cmdChan) {
		ps.Logger.Sugar().Warnf(`Commander(),false,setActived:%+v.%s"`, setActived, ps.Info())
		return false
	}
	ok := Cmd.WaitTimeout(cmdWaitInterval)
	if ok {
		return true
	} else {
		ps.Logger.Sugar().Errorf("Cmd.WaitTimeout() err:%+v,setActived:%s.%+v", Cmd.Err(), setActived, ps.Info())
		return false
	}
}

func (ps *PlayScreen) Commander(cmd cmd_token.TokenCompletor, cmdChan chan<- cmd_token.TokenCompletor) bool {
	if !ps.State.Load().(bool) {
		cmd.SetErr(fmt.Errorf("%s is exited", ps.Info()))
		return false
	}

	timer := mdl.TimerPool.Get(cmdWaitInterval)
	defer mdl.TimerPool.Put(timer)
	for {
		select {
		case <-ps.Ctrl().Context().Done():
			return false
		case cmdChan <- cmd:
			if ps.Cnf.EnableDebug {
				ps.Logger.Sugar().Debugf("get cmd_token - ok.cmd chan:%#v, cap:%d,len:%d.%s", cmdChan, cap(cmdChan), len(cmdChan), ps.Info())
			}
			return true
		case <-timer.C:
			if ps.Cnf.EnableDebug {
				ps.Logger.Sugar().Debugf("get cmd_token - time out: cmd chan:%#v, cap:%d,len:%d.%s", cmdChan, cap(cmdChan), len(cmdChan), ps.Info())
			}
			return false
		}
	}
}

// 根据节目的切换状态 检查状态 并执行命令
func (ps *PlayScreen) setProgramCmds(setProgramNormalActived, setProgramSpeedActived bool) bool {
	//获取节目的状态
	ps.programDefaultSetInfo.CurrentOn = ps.programDefaultStatus.IsActiced()
	ps.programSpeedSetInfo.CurrentOn = ps.programSpeedStatus.IsActiced()

	cmdNormalOK := false
	if ps.programDefaultSetInfo.CurrentOn != setProgramNormalActived {
		cmdNormalOK = ps.CmdProgramWork(setProgramNormalActived, ps.wkProgramDefaultCmdChan)
	}

	cmdSpeedOK := false
	if ps.programSpeedSetInfo.CurrentOn != setProgramSpeedActived {
		cmdSpeedOK = ps.CmdProgramWork(setProgramSpeedActived, ps.wkProgramSpeedCmdChan)
	}

	if cmdNormalOK {
		ps.programDefaultSetInfo.CurrentOn = ps.programDefaultStatus.IsActiced()
	}

	if cmdSpeedOK {
		ps.programSpeedSetInfo.CurrentOn = ps.programSpeedStatus.IsActiced()
	}

	return (ps.programDefaultSetInfo.CurrentOn == setProgramNormalActived) && (ps.programSpeedSetInfo.CurrentOn == setProgramSpeedActived)
}

// 检查节目的切换状态 并执行动作
func (ps *PlayScreen) CheckingSwitchPrograms() {
	//获取节目的状态
	ps.programDefaultSetInfo.CurrentOn = ps.programDefaultStatus.IsActiced()
	ps.programSpeedSetInfo.CurrentOn = ps.programSpeedStatus.IsActiced()
	ps.programDefaultSetInfo.NeedCmdSet = ps.programDefaultSetInfo.CurrentOn != ps.programDefaultSetInfo.CmdSetOn
	ps.programSpeedSetInfo.NeedCmdSet = ps.programSpeedSetInfo.CurrentOn != ps.programSpeedSetInfo.CmdSetOn
	// 只要有一个节目需要设置 就开始检查
	if ps.programDefaultSetInfo.NeedCmdSet || ps.programSpeedSetInfo.NeedCmdSet {
		//1. 先设置所有节目工作线程 都停止发布文本节目 否则什么都不做 等待下一次设置成功
		if ps.setProgramCmds(false, false) {
			if ps.Cnf.EnableDebug {
				ps.Logger.Sugar().Debugf(`deActive all program worker : ok.%s`, ps.Info())
			}

			//都设置成功的情况下 重置节目 和激活对应的工作线程
			switch ps.activeProgramType {
			case ProgramUnknown:
				{
					// 初始化的情况下
					if ps.programDefaultSetInfo.CmdSetOn {
						ps.checkingSwitchToProgramDefault()
					}

					if ps.programSpeedSetInfo.CmdSetOn {
						ps.checkingSwitchToProgramSpeed()
					}
				}
			case PrograDefault:
				{
					// 当前激活的节目为 默认节目的情况下
					if ps.programSpeedSetInfo.CmdSetOn {
						ps.checkingSwitchToProgramSpeed()
					}
				}
			case ProgramSpeed:
				{
					// 当前激活的节目为 显速节目的情况下
					if ps.programDefaultSetInfo.CmdSetOn {
						ps.checkingSwitchToProgramDefault()
					}
				}
			}
		} else {
			// 所有节目停止发送文本节目 设置失败
			if ps.Cnf.EnableDebug {
				ps.Logger.Sugar().Debugf(`deActive all program worker : failed.%s`, ps.Info())
			}
		}
	}
}
