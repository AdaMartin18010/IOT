package playscreen

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"strings"

	cm "navigate/common"
	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	navl_md "navigate/iot/navlock/model"
)

func (ps *PlayScreen) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		ps.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (ps *PlayScreen) Start() (err error) {
	if ps.CptSt == nil || ps.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", ps.CmptInfo())
	} else {
		if err = ps.CptSt.Validate(); err != nil {
			return err
		}
	}

	if ps.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", ps.CmptInfo())
	}
	ps.Logger.Sugar().Infof("%s is Starting", ps.CmptInfo())

	// 添加组件 注意ID 命名冲突 会导致创建和获取错误
	//------------- 初始化 program-default -----------------
	if ps.Cpts.Cpt(cmpt.IdName("program-default")) == nil {
		cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
			cmpt.IdName("program-default"),
			0,
			ps.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
				ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"worker",
				"program-default"))

		workdr := drs_wd.NewWorkDvr(dr, ps.TrackingSt)
		workdr.Fn = ps.ProgramDefaultWork
		ps.wkProgramDefault = workdr
		ps.wkProgramDefault.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.wkProgramDefault)
	}
	// --------------------------------------------

	//------------- 初始化 Program Speed -----------------
	if ps.Cpts.Cpt(cmpt.IdName("program-speed")) == nil {
		cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
			cmpt.IdName("program-speed"),
			0,
			ps.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
				ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"woker",
				"program-speed"))

		workdr := drs_wd.NewWorkDvr(dr, ps.TrackingSt)
		workdr.Fn = ps.ProgramSpeedWork
		ps.wkProgramSpeed = workdr
		ps.wkProgramSpeed.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.wkProgramSpeed)
	}
	// --------------------------------------------

	if ps.Cnf.EnableAutoTest {
		//------------- 初始化worker screen-shot -----------------
		if ps.Cpts.Cpt(cmpt.IdName("screen-shot")) == nil {
			cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
				cmpt.IdName("screen-shot"),
				0,
				ps.Ctrl().ForkCtxWg())

			dr := navl_md.NewBaseCptCtx(cs,
				navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
					ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"worker",
					"screen-shot"))

			workdr := drs_wd.NewWorkDvr(dr, ps.TrackingSt)
			workdr.Fn = ps.ScreenShotWork
			ps.wkScreenShot = workdr
			ps.wkScreenShot.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.wkScreenShot)
		}
		// --------------------------------------------

		//------------- 初始化httpdr http-screen-shot-----------------
		if ps.Cpts.Cpt(cmpt.IdName("http-screen-shot")) == nil {
			cs := navl_md.NewCptSt(drs_once.OnceKind,
				cmpt.IdName("http-screen-shot"),
				0,
				ps.Ctrl().ForkCtxWg())

			dr := navl_md.NewBaseCptCtx(cs,
				navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"httponce",
					"http-screen-shot"))

			ps.httpGetScreenShot = drs_once.NewOnceDvr(dr, ps.TrackingSt)
			ps.httpGetScreenShot.Fn = ps.doHttpScreenShot
			ps.httpGetScreenShot.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.httpGetScreenShot)
		}
		// --------------------------------------------

	}

	if ps.Cnf.EnableClearAndPublishProgram {
		//------------- 初始化worker program-init -----------------
		if ps.Cpts.Cpt(cmpt.IdName("program-init")) == nil {
			cs := navl_md.NewCptSt(drs_wd.WorkDriverKind,
				cmpt.IdName("program-init"),
				0,
				ps.Ctrl().ForkCtxWg())

			dr := navl_md.NewBaseCptCtx(cs,
				navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
					ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"worker",
					"program-init"))

			ps.wkProgramInit = drs_wd.NewWorkDvr(dr, ps.TrackingSt)
			ps.wkProgramInit.Fn = ps.ProgramInitWork
			ps.wkProgramInit.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.wkProgramInit)
		}
		// --------------------------------------------

		//------------- 初始化httpdr clear Program-----------------
		if ps.Cpts.Cpt(cmpt.IdName("http-clear-program")) == nil {
			tmpMndCtrl := navl_md.NewCptSt(drs_once.OnceKind,
				cmpt.IdName("http-clear-program"),
				0,
				ps.Ctrl().ForkCtxWg())

			tmpdr := navl_md.NewBaseCptCtx(tmpMndCtrl,
				navl_md.NewModelSysSt(tmpMndCtrl.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"httponce",
					"http-clear-program"))

			ps.httpClearLanProgram = drs_once.NewOnceDvr(tmpdr, ps.TrackingSt)
			ps.httpClearLanProgram.Fn = ps.doClearLanProgram
			ps.httpClearLanProgram.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.httpClearLanProgram)
		}
		// --------------------------------------------

		//------------- 初始化httpdr publish Program default-----------------
		if ps.Cpts.Cpt(cmpt.IdName("publish-program-default")) == nil {
			tmpMndCtrl := navl_md.NewCptSt(drs_once.OnceKind,
				cmpt.IdName("publish-program-default"),
				0,
				ps.Ctrl().ForkCtxWg())

			tmpdr := navl_md.NewBaseCptCtx(tmpMndCtrl,
				navl_md.NewModelSysSt(tmpMndCtrl.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"httponce",
					"publish-program-default"))

			ps.httpPublishProgramDefault = drs_once.NewOnceDvr(tmpdr, ps.TrackingSt)
			ps.httpPublishProgramDefault.Fn = ps.doPublishProgramDefault
			ps.httpPublishProgramDefault.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.httpPublishProgramDefault)
		}
		// --------------------------------------------

		//------------- 初始化httpdr publish Program Speed-----------------
		if ps.Cpts.Cpt(cmpt.IdName("publish-program-speed")) == nil {
			tmpMndCtrl := navl_md.NewCptSt(drs_once.OnceKind,
				cmpt.IdName("publish-program-speed"),
				0,
				ps.Ctrl().ForkCtxWg())

			tmpdr := navl_md.NewBaseCptCtx(tmpMndCtrl,
				navl_md.NewModelSysSt(tmpMndCtrl.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
					"playscreen",
					"httponce",
					"publish-program-speed"))

			ps.httpPublishProgramSpeed = drs_once.NewOnceDvr(tmpdr, ps.TrackingSt)
			ps.httpPublishProgramSpeed.Fn = ps.doPublishProgramSpeed
			ps.httpPublishProgramSpeed.EnableDebug = ps.Cnf.EnableDebug
			ps.AddCpts(ps.httpPublishProgramSpeed)
		}
		// --------------------------------------------

	}

	//------------- 初始化httpdr active Program default-----------------
	if ps.Cpts.Cpt(cmpt.IdName("active-program-default")) == nil {
		tmpMndCtrl := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("active-program-default"),
			0,
			ps.Ctrl().ForkCtxWg())

		tmpdr := navl_md.NewBaseCptCtx(tmpMndCtrl,
			navl_md.NewModelSysSt(tmpMndCtrl.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"httponce",
				"active-program-default"))

		ps.httpActiveProgramDefault = drs_once.NewOnceDvr(tmpdr, ps.TrackingSt)
		ps.httpActiveProgramDefault.Fn = ps.doActiveProgramDefault
		ps.httpActiveProgramDefault.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.httpActiveProgramDefault)
	}
	// --------------------------------------------

	//------------- 初始化httpdr active Program speed-----------------
	if ps.Cpts.Cpt(cmpt.IdName("active-program-speed")) == nil {
		tmpMndCtrl := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("active-program-speed"),
			0,
			ps.Ctrl().ForkCtxWg())

		tmpdr := navl_md.NewBaseCptCtx(tmpMndCtrl,
			navl_md.NewModelSysSt(tmpMndCtrl.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"httponce",
				"active-program-speed"))

		ps.httpActiveProgramSpeed = drs_once.NewOnceDvr(tmpdr, ps.TrackingSt)
		ps.httpActiveProgramSpeed.Fn = ps.doActiveProgramSpeed
		ps.httpActiveProgramSpeed.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.httpActiveProgramSpeed)
	}
	// --------------------------------------------

	//------------- 初始化httpdr update program default lines-----------------
	if ps.Cpts.Cpt(cmpt.IdName("update-program-default-lines")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("update-program-default-lines"),
			0,
			ps.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"httponce",
				"update-program-default-lines"))

		ps.httpUpdateProgramDefaultMultiLines = drs_once.NewOnceDvr(dr, ps.TrackingSt)
		ps.httpUpdateProgramDefaultMultiLines.Fn = ps.doUpdateProgramDefaultMultiLines
		ps.httpUpdateProgramDefaultMultiLines.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.httpUpdateProgramDefaultMultiLines)
	}
	// --------------------------------------------

	//------------- 初始化httpdr update program speed lines-----------------
	if ps.Cpts.Cpt(cmpt.IdName("update-program-speed-lines")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("update-program-speed-lines"),
			0,
			ps.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", ps.TrackingSt.NavlockIdentifier),
				"playscreen",
				"httponce",
				"update-program-speed-lines"))

		ps.httpUpdateProgramSpeedMultiLines = drs_once.NewOnceDvr(dr, ps.TrackingSt)
		ps.httpUpdateProgramSpeedMultiLines.Fn = ps.doUpdateProgramSpeedMultiLines
		ps.httpUpdateProgramSpeedMultiLines.EnableDebug = ps.Cnf.EnableDebug
		ps.AddCpts(ps.httpUpdateProgramSpeedMultiLines)
	}
	// --------------------------------------------

	//所有初始化工作完成后启动
	ps.Ctrl().WaitGroup().StartingWait(ps.WorkerRecover)

	// 只创建 和添加组件 不启动 由程序自己控制启停
	// if ps.Cpts != nil && ps.Cpts.Len() > 0 {
	// 	err = ps.Cpts.Start()
	// 	if err != nil {
	// 		ps.Logger.Sugar().Errorf("Start() error : %+v", err)
	// 	}
	// }

	ps.State.Store(true)
	ps.Logger.Sugar().Infof("%s is Started", ps.CmptInfo())
	return nil
}

func (ps *PlayScreen) Stop() (err error) {
	if !ps.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", ps.CmptInfo())
	}

	ps.Ctrl().Cancel()
	ps.Logger.Sugar().Infoln("--- Stopping ---")
	<-ps.Ctrl().Context().Done()

	// 退出时全部停止
	if ps.Cpts != nil && ps.Cpts.Len() > 0 {
		err = ps.Cpts.Stop()
		if err != nil {
			ps.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}
	ps.State.Store(false)
	ps.Logger.Sugar().Infoln("--- Stopped ---")
	return err
}

// 读取和检查json配置文件 包括节目名称的默认值 处理这些异常和初始值
func (ps *PlayScreen) ReadConfigJson() error {
	data, err := os.ReadFile(ps.Cnf.DefaultProgramCnf)
	if err == nil {
		data = cm.EraseControlChar(data)
		// 擦除掉两个空格
		data = bytes.ReplaceAll(data, []byte("  "), []byte(""))
		if mdl.Json.Valid(data) {
			ps.programDefaultJsonContent = string(data)
		} else {
			ps.Logger.Sugar().Fatalf("Cnf.DefaultProgramCnf content:%s is not valid json format.",
				ps.programDefaultJsonContent)
		}
		ps.Logger.Sugar().Infof("Cnf.DefaultProgramCnf content:%s;",
			ps.programDefaultJsonContent)
	} else {
		return fmt.Errorf("read %s file err:%+v", ps.Cnf.DefaultProgramCnf, err)
	}

	data, err = os.ReadFile(ps.Cnf.DefaultProgramLinesCnf)
	if err == nil {
		data = cm.EraseControlChar(data)
		data = bytes.ReplaceAll(data, []byte("  "), []byte(""))
		if mdl.Json.Valid(data) {
			ps.programDefaultLinesJsonContent = string(data)
		} else {
			ps.Logger.Sugar().Fatalf("Cnf.DefaultProgramLinesCnf content:%s is not valid json format.",
				ps.programDefaultLinesJsonContent)
		}

		ps.Logger.Sugar().Infof("Cnf.DefaultProgramLinesCnf content:%s;",
			ps.programDefaultLinesJsonContent)

	} else {
		return fmt.Errorf("read %s file err:%+v", ps.Cnf.DefaultProgramLinesCnf, err)
	}

	ps.Cnf.DefaultProgramName = strings.TrimSpace(ps.Cnf.DefaultProgramName)
	if ps.Cnf.DefaultProgramName == "" {
		ps.programDefaultName = "default"
	} else {
		ps.programDefaultName = ps.Cnf.DefaultProgramName
	}

	data, err = os.ReadFile(ps.Cnf.SpeedProgramCnf)
	if err == nil {
		data = cm.EraseControlChar(data)
		data = bytes.ReplaceAll(data, []byte("  "), []byte(""))
		if mdl.Json.Valid(data) {
			ps.programSpeedJsonContent = string(data)
		} else {
			ps.Logger.Sugar().Fatalf("Cnf.SpeedProgramCnf content:%s is not valid json format.",
				ps.programSpeedJsonContent)
		}
		ps.Logger.Sugar().Infof("Cnf.SpeedProgramCnf content:%s;", ps.programSpeedJsonContent)
	} else {
		return fmt.Errorf("read %s file err:%+v", ps.Cnf.SpeedProgramCnf, err)
	}

	data, err = os.ReadFile(ps.Cnf.SpeedProgramLinesCnf)
	if err == nil {
		data = cm.EraseControlChar(data)
		data = bytes.ReplaceAll(data, []byte("  "), []byte(""))
		if mdl.Json.Valid(data) {
			ps.programSpeedLinesJsonContent = string(data)
		} else {
			ps.Logger.Sugar().Fatalf("Cnf.SpeedProgramLinesCnf content:%s is not valid json format.",
				ps.programSpeedLinesJsonContent)
		}
		ps.Logger.Sugar().Infof("Cnf.SpeedProgramLinesCnf content:%s;", ps.programSpeedLinesJsonContent)
	} else {
		return fmt.Errorf("read %s file err:%+v", ps.Cnf.SpeedProgramLinesCnf, err)
	}

	ps.Cnf.SpeedProgramName = strings.TrimSpace(ps.Cnf.SpeedProgramName)
	if ps.Cnf.SpeedProgramName == "" {
		ps.programSpeedName = "speed"
	} else {
		ps.programSpeedName = ps.Cnf.SpeedProgramName
	}

	return nil
}
