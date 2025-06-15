package playscreen

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

	cm "navigate/common"
	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	cmd_token "navigate/common/model/token"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	drs_wd "navigate/iot/navlock/gezhouba/model/driver/work"
	gzb_schedule "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

var (
	_ mdl.WorkerRecover = (*PlayScreen)(nil)
	_ cmpt.Cpt          = (*PlayScreen)(nil)
	_ cmpt.CptsOperator = (*PlayScreen)(nil)
	_ cmpt.CptComposite = (*PlayScreen)(nil)
	_ cmpt.CptRoot      = (*PlayScreen)(nil)
)

const (
	Kind            = cmpt.KindName("playscreen")
	cmdbufsize      = 10240
	cmdWaitInterval = 1000 * time.Millisecond
)

type cmdSetInfo struct {
	RunningOk  bool // 是否是处于运行状态
	CurrentOn  bool // 当前的状态是否开启
	CmdSetOn   bool // 命令设置的状态
	NeedCmdSet bool // 是否需要执行命令
	Info       string
}

func (cm cmdSetInfo) FmtStr() string {
	return fmt.Sprintf("CmdSetInfo:%+v", cm)
}

type ProgramCmd struct {
	*cmd_token.BaseToken
	SetONOrOFF bool
	Info       string
}

func NewProgramCmd() *ProgramCmd {
	return &ProgramCmd{
		BaseToken:  cmd_token.NewBaseToken(),
		SetONOrOFF: false,
		Info:       "",
	}
}

type ProgramStatus struct {
	mu      *sync.Mutex
	actived bool
}

func NewProgramStatus() *ProgramStatus {
	return &ProgramStatus{
		mu:      &sync.Mutex{},
		actived: false,
	}
}

func (ps *ProgramStatus) IsActiced() bool {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return ps.actived
}

func (ps *ProgramStatus) SetStatus(actived bool) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.actived = actived
}

type FilePathWrap struct {
	mu   *sync.Mutex
	path string
}

func NewFilePathWrap() *FilePathWrap {
	return &FilePathWrap{
		mu:   &sync.Mutex{},
		path: "",
	}
}

func (fp *FilePathWrap) Get() string {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	return fp.path
}

func (fp *FilePathWrap) Set(p string) {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.path = p
}

// 需要进一步确认 驱动的状态
type PlayScreen struct {
	*navl_md.BaseCptCtx
	*gzb_schedule.TrackingSt
	Cnf *gzb_cf.PlayScreenConfig

	activeProgramType                  ProgramType
	httpDoActiveProgramDefault         bool
	httpDoActiveProgramDefaultReturnOK bool
	httpDoActiveProgramSpeed           bool
	httpDoActiveProgramSpeedReturnOk   bool

	programDefaultName                   string
	programDefaultJsonContent            string
	programDefaultLinesJsonContent       string
	programDefaultUpdateLinesJsonContent string

	programSpeedName                   string
	programSpeedJsonContent            string
	programSpeedLinesJsonContent       string
	programSpeedUpdateLinesJsonContent string

	msgChanShipSpeeds <-chan any
	msgChanNavState   <-chan any
	msgChanNavlState  <-chan any

	msgChanProgramInitNotifyOk chan bool

	programDefaultSetInfo   *cmdSetInfo
	programSpeedSetInfo     *cmdSetInfo
	screenShotInfo          *cmdSetInfo
	programDefaultStatus    *ProgramStatus
	programSpeedStatus      *ProgramStatus
	screenShotStatus        *ProgramStatus
	wkProgramDefaultCmdChan chan cmd_token.TokenCompletor
	wkProgramSpeedCmdChan   chan cmd_token.TokenCompletor
	wkScreenShotCmdChan     chan cmd_token.TokenCompletor
	wkProgramDefault        *drs_wd.Work
	wkProgramSpeed          *drs_wd.Work
	wkScreenShot            *drs_wd.Work
	wkProgramInit           *drs_wd.Work

	httpClearLanProgram                *drs_once.Once
	httpPublishProgramDefault          *drs_once.Once
	httpPublishProgramSpeed            *drs_once.Once
	httpActiveProgramDefault           *drs_once.Once
	httpActiveProgramSpeed             *drs_once.Once
	httpGetScreenShot                  *drs_once.Once
	httpUpdateProgramDefaultMultiLines *drs_once.Once
	httpUpdateProgramSpeedMultiLines   *drs_once.Once

	currentScheduleId string
	currentNavStatus  gzb_schedule.NavStatus
	currentNavlStatus gzb_schedule.NavlockStatus
	screenShotPath    string
	filePathSetMutex  *sync.Mutex

	rand               *rand.Rand
	screenShotFileName *FilePathWrap
}

func NewPlayScreenSvr(dr *navl_md.BaseCptCtx, trs *gzb_schedule.TrackingSt) *PlayScreen {
	tmp := &PlayScreen{
		BaseCptCtx:                 dr,
		TrackingSt:                 trs,
		rand:                       rand.New(rand.NewSource(time.Now().UnixNano())),
		activeProgramType:          ProgramUnknown,
		filePathSetMutex:           &sync.Mutex{},
		programDefaultSetInfo:      &cmdSetInfo{},
		programSpeedSetInfo:        &cmdSetInfo{},
		screenShotInfo:             &cmdSetInfo{},
		screenShotFileName:         NewFilePathWrap(),
		programDefaultStatus:       NewProgramStatus(),
		programSpeedStatus:         NewProgramStatus(),
		screenShotStatus:           NewProgramStatus(),
		wkProgramDefaultCmdChan:    make(chan cmd_token.TokenCompletor, cmdbufsize),
		wkProgramSpeedCmdChan:      make(chan cmd_token.TokenCompletor, cmdbufsize),
		wkScreenShotCmdChan:        make(chan cmd_token.TokenCompletor, cmdbufsize),
		msgChanProgramInitNotifyOk: make(chan bool, 1),
	}
	tmp.WorkerRecover = tmp
	return tmp
}

// 根据船闸状态 闸次号  创建新的文件夹目录
// todo : 默认是nav01路径下的 目录
func (ps *PlayScreen) createFilePath() error {
	ps.filePathSetMutex.Lock()
	defer ps.filePathSetMutex.Unlock()
	path := ps.Cnf.AutoTestDirector
	notSameAsCurrentScheduleId := false
	if ps.currentScheduleId != ps.TrackingSt.ScheduleId.GetScheduleId() {
		ps.currentScheduleId = ps.TrackingSt.ScheduleId.GetScheduleId()
		notSameAsCurrentScheduleId = true
	}

	notSameAsCurrentNavStatus := false
	if ps.currentNavStatus != ps.TrackingSt.PredicateSchedule.GetNavStatus() {
		ps.currentNavStatus = ps.TrackingSt.PredicateSchedule.GetNavStatus()
		notSameAsCurrentNavStatus = true
	}

	notSameAsCurrentNavlStatus := false
	if ps.currentNavlStatus != ps.TrackingSt.PredicateSchedule.GetNavlockStatus() {
		ps.currentNavlStatus = ps.TrackingSt.PredicateSchedule.GetNavlockStatus()
		notSameAsCurrentNavlStatus = true
	}

	if notSameAsCurrentScheduleId || notSameAsCurrentNavStatus || notSameAsCurrentNavlStatus {
		tempTopPathName := fmt.Sprintf("%s_%s", ps.currentScheduleId, ps.currentNavStatus.String())
		path := cm.PathJoin(path, tempTopPathName)
		tempLevelPathName := fmt.Sprintf("%s_%s", ps.currentNavlStatus.String(), ps.activeProgramType.String())
		path = cm.PathJoin(path, tempLevelPathName)
		if err := cm.CreatePathDir(path); err != nil {
			ps.Logger.Sugar().Warnf(" Create directory:%s, err: %+v.%s", path, err, ps.Info())
			return err
		}
	}
	ps.screenShotPath = path
	return nil
}

// 根据字符串组合成 文件路径 并设置
func (ps *PlayScreen) createScreenShotFilePathName(filename string) {
	ps.filePathSetMutex.Lock()
	defer ps.filePathSetMutex.Unlock()
	filePath := cm.PathJoin(ps.screenShotPath, filename)
	ps.screenShotFileName.Set(filePath)
}

// 退出时 多次检查是否切换回默认节目
func (ps *PlayScreen) exitingSwitchToProgramDefault() bool {
	ps.httpDoActiveProgramDefault = false
	ps.httpDoActiveProgramDefaultReturnOK = false
	tickerCheckInterval := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer tickerCheckInterval.Stop()
	i := 1
	tryNum := 3
	for {
		runtime.Gosched()
		ps.httpActiveProgramDefault.Logger.Sugar().Infof(`Exiting -- Checking Switch to Default Program ....... continue :%d .%s`, i, ps.Info())
		//todo : 退出时 是否需要重新设置上下文
		ps.httpActiveProgramDefault.WithCtx(context.Background())
		select {
		case <-tickerCheckInterval.C:
			{
				// drain the chan
				select {
				case <-tickerCheckInterval.C:
				default:
				}

				if !ps.httpActiveProgramDefault.IsRunning() {
					ps.startProgramDefaultActive()

					i = i + 1

					if i > tryNum {
						return false
					}
				}

			}
		case result, ok := <-ps.httpActiveProgramDefault.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpActiveProgramDefault.Stop(); err != nil {
						ps.httpActiveProgramDefault.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpActiveProgramDefault.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpActiveProgramDefault, result) {
					return true
				}
			}

		}
	}
}

func (ps *PlayScreen) Work() error {
	defer ps.State.Store(false)

	if ps.Cnf.EnableDebug {
		ps.Logger.Sugar().Debugf(`Work Starting.%s`, ps.TrackingSt.Info())
		defer ps.Logger.Sugar().Debugf(`Work Exited.%s`, ps.TrackingSt.Info())
	}

	ps.Logger.Sugar().Infof(`Work Starting. Cnf :%s.%s`, ps.Cnf, ps.Info())

	// 启动自动测试情况下 启动获取播放屏的抓取图像工作者
	if ps.Cnf.EnableAutoTest {
		ps.wkScreenShot.Start()
		ps.screenShotInfo.RunningOk = ps.wkScreenShot.IsRunning()
	}

	if ps.Cnf.EnableClearAndPublishProgram {

		tickerCheckInterval5min := time.NewTicker(time.Duration(5*60) * time.Second)
		defer tickerCheckInterval5min.Stop()

		ps.wkProgramInit.Start()
		for {
			select {
			case <-ps.Ctrl().Context().Done():
				{
					ps.wkProgramInit.Stop()
					return nil
				}
			case initOk, ok := <-ps.msgChanProgramInitNotifyOk:
				{
					if !ok {
						ps.Logger.Sugar().Infof(`ps.wkProgramInit chan close And exit.%s`, ps.Info())
					}
					ps.Logger.Sugar().Infof(`Program init:%t.%s`, initOk, ps.Info())
					if initOk {
						// 关闭抓屏 定时器
						tickerCheckInterval5min.Stop()
						// 成功的情况下 结束初始化 并跳转到 主流程
						ps.wkProgramInit.Stop()
						goto mainloop
					}
				}
			case <-tickerCheckInterval5min.C:
				{
					// drain the chan
					select {
					case <-tickerCheckInterval5min.C:
					default:
					}

					if ps.Cnf.EnableAutoTest {
						ps.createScreenShotFilePathName(fmt.Sprintf("SomeMinShot_%d", ps.rand.Intn(1000)))
						ps.ScreenShot()
					}
				}
			}
		}
	}

mainloop:

	tickerCheckInterval := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer tickerCheckInterval.Stop()

	// 主流程
	// 1. 判断当前通航状态和船闸状态 然后做出节目切换指令
	// 2. 先把所有节目工作暂停
	// 3. 再http去激活对应的节目
	// 4. 再把对应的节目工作激活
	ps.msgChanNavState = ps.EventChans.Subscribe(gzb_schedule.MsgTopic_NavStatus)
	ps.msgChanNavlState = ps.EventChans.Subscribe(gzb_schedule.MsgTopic_NavlockStatus)
	defer func() {
		ps.EventChans.UnSubscribe(gzb_schedule.MsgTopic_NavStatus, ps.msgChanNavState)
		ps.EventChans.UnSubscribe(gzb_schedule.MsgTopic_NavlockStatus, ps.msgChanNavlState)
	}()

	ps.wkProgramDefault.Start()
	ps.programDefaultSetInfo.RunningOk = ps.wkProgramDefault.IsRunning()
	ps.wkProgramSpeed.Start()
	ps.programSpeedSetInfo.RunningOk = ps.wkProgramSpeed.IsRunning()

	for {
		runtime.Gosched()
		select {
		case <-ps.Ctrl().Context().Done():
			{
				ps.Logger.Sugar().Debugf(`works exit.%s`, ps.Info())
				if ps.screenShotInfo.RunningOk {
					ps.wkScreenShot.Stop()
				}
				ps.wkProgramDefault.Stop()
				ps.wkProgramSpeed.Stop()
				// 说明没有初始化成功或者没有出现有效的切换
				if ps.activeProgramType == ProgramUnknown {
					return nil
				}
				// 退出时 切换到默认的节目
				if ps.Cnf.EnableExitActiveDefaultProgram {
					//todo : 退出时 激活默认节目
					if ps.activeProgramType != PrograDefault {
						if !ps.exitingSwitchToProgramDefault() {
							ps.Logger.Sugar().Warnf(`Program Switch to Default Program failed ---- User have to check out!!!.%s`, ps.Info())
						}
					}
				}
				return nil
			}
		case _, ok := <-ps.msgChanNavlState:
			{
				if !ok {
					ps.Logger.Sugar().Infof(`Event chan closed.%s`, ps.Info())
					continue
				}
				ps.PredicateProgramSwitch()
				ps.CheckingSwitchPrograms()
				if ps.Cnf.EnableDebug {
					ps.Logger.Sugar().Debugf(`receive navls. programNormalSetInfo:%s,programSpeedSetInfo:%s.%s`,
						ps.programDefaultSetInfo.FmtStr(),
						ps.programSpeedSetInfo.FmtStr(),
						ps.Info())
				}
			}
		case _, ok := <-ps.msgChanNavState:
			{
				if !ok {
					ps.Logger.Sugar().Infof(`Event chan closed.%s`, ps.Info())
					continue
				}

				ps.PredicateProgramSwitch()
				ps.CheckingSwitchPrograms()
				if ps.Cnf.EnableDebug {
					ps.Logger.Sugar().Debugf(`receive navls. programNormalSetInfo:%s,programSpeedSetInfo:%s.%s`,
						ps.programDefaultSetInfo.FmtStr(),
						ps.programSpeedSetInfo.FmtStr(),
						ps.Info())
				}
			}
		case <-tickerCheckInterval.C:
			{
				// drain the chan
				select {
				case <-tickerCheckInterval.C:
				default:
				}

				ps.CheckingSwitchPrograms()
			}
		case result, ok := <-ps.httpActiveProgramDefault.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpActiveProgramDefault.Stop(); err != nil {
						ps.httpActiveProgramDefault.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpActiveProgramDefault.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpActiveProgramDefault, result) {
					if ps.Cnf.EnableDebug {
						ps.httpActiveProgramDefault.Logger.Sugar().Debugf(`httpActiveProgramDefault: ok.%s`, ps.Info())
					}
					ps.httpDoActiveProgramDefaultReturnOK = true
				} else {
					ps.httpActiveProgramDefault.Logger.Sugar().Errorf(`httpActiveProgramDefault: failed.%s`, ps.Info())
				}
			}
		case result, ok := <-ps.httpActiveProgramSpeed.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpActiveProgramSpeed.Stop(); err != nil {
						ps.httpActiveProgramSpeed.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpActiveProgramSpeed.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpActiveProgramSpeed, result) {
					if ps.Cnf.EnableDebug {
						ps.httpActiveProgramSpeed.Logger.Sugar().Debugf(`httpActiveProgramSpeed: ok.%s`, ps.Info())
					}
					ps.httpDoActiveProgramSpeedReturnOk = true
				} else {
					ps.httpActiveProgramSpeed.Logger.Sugar().Errorf(`httpActiveProgramSpeed: failed.%s`, ps.Info())
				}
			}
		}
	}

}
