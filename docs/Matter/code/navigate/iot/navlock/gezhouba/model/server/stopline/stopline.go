package stopline

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	gzb_api_stpl "navigate/iot/navlock/gezhouba/api/stopline"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
	gzb_db "navigate/store/iot/navlock/gezhouba"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"
)

var (
	_ mdl.WorkerRecover = (*Stopline)(nil)
	_ cmpt.Cpt          = (*Stopline)(nil)
	_ cmpt.CptsOperator = (*Stopline)(nil)
	_ cmpt.CptComposite = (*Stopline)(nil)
	_ cmpt.CptRoot      = (*Stopline)(nil)
)

const (
	StoplineKind = cmpt.KindName("stopline")
)

type Stopline struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	UpCnf   *gzb_cf.HttpClientConfig
	DownCnf *gzb_cf.HttpClientConfig
	wkUp    *drs_once.Once
	wkDown  *drs_once.Once

	ConnectReadTimeout int
	DoTimeInterval     int
}

func NewStoplineSvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Stopline {
	tmp := &Stopline{
		BaseCptCtx: dr,
		TrackingSt: trs,
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (sl *Stopline) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		sl.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (sl *Stopline) Start() (err error) {
	if sl.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", sl.Info())
	}

	sl.DoTimeInterval = sl.UpCnf.DoTimeInterval
	if sl.UpCnf.DoTimeInterval > sl.DownCnf.DoTimeInterval {
		sl.DoTimeInterval = sl.DownCnf.DoTimeInterval
	}

	if sl.TrackingSt.EnableDebug {
		sl.Logger.Sugar().Debugf("%s is Starting", sl.Info())
		defer sl.Logger.Sugar().Debugf("%s is Started", sl.Info())
	}

	// 只有Stopline初始化完成后 才能取到合适的参数 所以在Start()里面实现
	//创建一个新的控制结构独立控制 httpdr 的运行
	//------------- 初始化httpdr Up-----------------
	if sl.Cpts.Cpt(cmpt.IdName("stopline-up")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("stopline-up"),
			0,
			sl.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", sl.TrackingSt.NavlockIdentifier),
				"",
				"",
				"stopline-up"))

		sl.wkUp = drs_once.NewOnceDvr(dr, sl.TrackingSt)
		sl.wkUp.Fn = sl.doHttpWorkUp
		sl.AddCpts(sl.wkUp)
	}
	// --------------------------------------------

	//------------- 初始化httpdr Down-----------------
	if sl.Cpts.Cpt(cmpt.IdName("stopline-down")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("stopline-down"),
			0,
			sl.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", sl.TrackingSt.NavlockIdentifier),
				"",
				"",
				"stopline-down"))

		sl.wkDown = drs_once.NewOnceDvr(dr, sl.TrackingSt)
		sl.wkDown.Fn = sl.doHttpWorkDown
		sl.AddCpts(sl.wkDown)
		//添加不启动
	}
	// --------------------------------------------

	//所有初始化工作完成后启动
	sl.Ctrl().WaitGroup().StartingWait(sl.WorkerRecover)

	// if sl.Cpts != nil && sl.Cpts.Len() > 0 {
	// 	err = sl.Cpts.Start()
	// 	if err != nil {
	// 		sl.Logger.Sugar().Errorf("Start() error : %+v", err)
	// 	}
	// }

	sl.State.Store(true)
	return
}

func (sl *Stopline) Stop() (err error) {
	if !sl.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", sl.Info())
	}

	sl.Ctrl().Cancel()

	if sl.TrackingSt.EnableDebug {
		sl.Logger.Sugar().Debugln("--- Stopping ---")
		defer sl.Logger.Sugar().Debugln("--- Stopped ---")
	}

	<-sl.Ctrl().Context().Done()

	if sl.Cpts != nil && sl.Cpts.Len() > 0 {
		err = sl.Cpts.Stop()
		if err != nil {
			sl.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	sl.State.Store(false)

	return
}

func (sl *Stopline) doHttpWorkUp(ctx context.Context) *drs_once.Result {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sl.UpCnf.Url, nil)
	if err != nil {
		sl.Logger.Sugar().Errorf(`%s[Dr:%s,%s]:"err : %+v"`,
			sl.CmptInfo(), sl.UpCnf.Name, sl.Info(), err)
	}
	resp, err := http.DefaultClient.Do(req)
	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (sl *Stopline) doHttpWorkDown(ctx context.Context) *drs_once.Result {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sl.DownCnf.Url, nil)
	if err != nil {
		sl.Logger.Sugar().Errorf(`%s[Dr:%s,%s]:"err : %+v"`,
			sl.CmptInfo(), sl.UpCnf.Name, sl.Info(), err)
	}
	resp, err := http.DefaultClient.Do(req)
	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (sl *Stopline) Work() error {
	if sl.TrackingSt.EnableDebug {
		sl.Logger.Sugar().Debugf(`[Dr:(%s,%s),%s]):"Work Starting,config: %+v,%+v"`,
			sl.UpCnf.Name, sl.DownCnf.Name,
			sl.Info(),
			sl.UpCnf, sl.DownCnf)
	}

	ticker := time.NewTicker(time.Duration(sl.DoTimeInterval) * time.Millisecond)
	defer ticker.Stop()

	wkUpWorkDo := sl.UpCnf.Enable
	wkDownWorkDo := sl.DownCnf.Enable
	for {
		runtime.Gosched()
		//todo: 检查通航状态 调度上下游的禁停线请求 来完成工作
		select {
		case <-sl.Ctrl().Context().Done():
			{
				if err := sl.Ctrl().Context().Err(); err != context.Canceled && err != context.DeadlineExceeded {
					sl.Logger.Sugar().Warnf(`%s,(%s):"err : %+v"`,
						sl.Info(), sl.UpCnf.Name, err)
				}
				return nil
			}
		case result, ok := <-sl.wkUp.Chan():
			{
				if !ok {
					if err := sl.wkUp.Stop(); err != nil {
						sl.wkUp.Logger.Sugar().Errorf(`%s - Stop() error: %+v`, sl.wkUp.Info(), err)
					}

					if !wkUpWorkDo {
						wkUpWorkDo = true
					}
				} else {
					sl.DealWith(sl.wkUp, result, sl.UpCnf, true)
				}
				result = nil
			}
		case result, ok := <-sl.wkDown.Chan():
			{
				if !ok {
					if err := sl.wkDown.Stop(); err != nil {
						sl.wkDown.Logger.Sugar().Errorf(`%s - Stop() error: %+v`, sl.wkDown.Info(), err)
					}
					if !wkDownWorkDo {
						wkDownWorkDo = true
					}
				} else {
					sl.DealWith(sl.wkDown, result, sl.DownCnf, false)
				}
				result = nil
			}
		case <-ticker.C:
			{
				// drain event
				select {
				case <-ticker.C:
				default:
				}

				if wkUpWorkDo {
					if !sl.wkUp.IsRunning() {
						sl.wkUp.Ctrl().WithTimeout(sl.Ctrl().Context(),
							time.Duration(sl.UpCnf.ConnectReadTimeout)*time.Millisecond)
						if err := sl.wkUp.Start(); err != nil {
							sl.wkUp.Logger.Sugar().Errorf(`%s - Start() error: %+v`, sl.wkUp.Info(), err)
						}
						wkUpWorkDo = false
					}
				}

				if wkDownWorkDo {
					if !sl.wkDown.IsRunning() {
						sl.wkDown.Ctrl().WithTimeout(sl.Ctrl().Context(),
							time.Duration(sl.DownCnf.ConnectReadTimeout)*time.Millisecond)
						if err := sl.wkDown.Start(); err != nil {
							sl.wkDown.Logger.Sugar().Errorf(`%s - Start() error: %+v`, sl.wkDown.Info(), err)
						}
						wkDownWorkDo = false
					}
				}
			}
		}
	}

}

func (sl *Stopline) DealWith(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.HttpClientConfig, isUpOrDown bool) error {
	defer func() {
		if result != nil && result.Val != nil {
			resp, ok := result.Val.(*http.Response)
			if ok && resp != nil {
				resp.Body.Close()
			}
		}
		result = nil
	}()

	if result.Err != nil {
		if errors.Is(result.Err, context.DeadlineExceeded) {
			if isUpOrDown {
				if sl.TrackingSt.Ref.AnimationWrap != nil {
					sl.TrackingSt.Ref.AnimationWrap.SetUpStoplineInfo(0, 0, 0, 0.0)
				}

			} else {
				if sl.TrackingSt.Ref.AnimationWrap != nil {
					sl.TrackingSt.Ref.AnimationWrap.SetDownStoplineInfo(0, 0, 0, 0.0)
				}

			}

			wk.Logger.Sugar().Errorf(`%s,(%s):"http-请求超时,err : %+v"`,
				sl.Info(), cnf.Name, result.Err)
			return nil
		}
		if errors.Is(result.Err, context.Canceled) {
			wk.Logger.Sugar().Debugf(`%s,(%s):"http  worker 退出,err : %+v"`,
				sl.Info(), cnf.Name, result.Err)
			return nil
		}

		wk.Logger.Sugar().Warnf(`%s,(%s):"http-请求,err : %+v"`,
			sl.Info(), cnf.Name, result.Err)
		return result.Err
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			sl.Logger.Sugar().Errorf(`%s :"tranform from interface failed"`, sl.Info())
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			wk.Logger.Sugar().Errorf(`%s,(%s):"Error reading response body: %+v"`,
				sl.Info(), cnf.Name, err)
			return err
		}

		if resp.StatusCode == 200 {
			sl.tranformData(body, cnf, isUpOrDown)
		}
		body = nil
		return nil
	}

}

// 禁停线数据的转换和更新
func (sl *Stopline) tranformData(ss []byte, cnf *gzb_cf.HttpClientConfig, isUpOrDown bool) {
	var slstruct gzb_api_stpl.ResponseJsonStruct
	if !slstruct.VerifyMaybeValid(ss) {
		sl.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"response body: not Valid"`,
			cnf.Tag, sl.ScheduleId.GetScheduleId())
		return
	}
	err := mdl.Json.Unmarshal(ss, &slstruct)
	if err != nil {
		sl.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"Error reading response body: %+v"`,
			cnf.Tag, sl.ScheduleId.GetScheduleId(), err)
		return
	}

	if (isUpOrDown && sl.UpCnf.EnableDebug) || (!isUpOrDown && sl.DownCnf.EnableDebug) {
		sl.Logger.Sugar().Debugf(`ResponseJsonStruct:%+v.%s`, slstruct, sl.Info())
	}
	var slst gzb_api_stpl.StoplineStruct
	slst.TransformFrom(&slstruct)
	if (isUpOrDown && sl.UpCnf.EnableDebug) || (!isUpOrDown && sl.DownCnf.EnableDebug) {
		sl.Logger.Sugar().Debugf(`StoplineStruct:%+v.%s`, slst, sl.Info())
	}
	sl.PublishStoplineInfos(&slst, cnf, isUpOrDown)
}

// 写入数据库
func (sl *Stopline) PublishStoplineInfos(st *gzb_api_stpl.StoplineStruct, cnf *gzb_cf.HttpClientConfig, isUpOrDown bool) {
	// 在实际运行的过程打开 过滤掉非报警和警告的记录
	// todo: 过滤掉重复的数据
	if !st.DectectedWarn && !st.DectectedAlarm {
		return
	}

	// todo: 过滤掉 不符合通航运营的状态 和 船闸状态
	//判断通航状态是否是运营有效的 并获取
	scheduleId := sl.TrackingSt.ScheduleId.GetScheduleId()
	navs, navsValid := sl.PredicateSchedule.GetIsValidNavState()
	if !navsValid {
		if sl.TrackingSt.EnableDebug {
			sl.Logger.Sugar().Debugf("navs,navsValid:%s,%t return.%s", navs.String(), navsValid, sl.Info())
		}
		return
	}

	//判断船闸状态是否是 船舶测速系统有效的 并获取
	navls, navlsValid := sl.PredicateSchedule.GetIsValidStopLineNavlState()
	if !navlsValid {
		if sl.TrackingSt.EnableDebug {
			sl.Logger.Sugar().Debugf("navls,navlsShipSpeedValid:%s,%t return.%s", navls.String(), navlsValid, sl.Info())
		}
		return
	}

	autoTest := sl.UpCnf.EnableAutoTest || sl.DownCnf.EnableAutoTest
	var info gzb_db.NavlGzbStoplineWarn

	stateAnimation := 0
	level := ""
	if st.DectectedWarn {
		level = "报警"
		stateAnimation = 1
	}

	if st.DectectedAlarm {
		level = "警告"
		stateAnimation = 2
	}

	info.NavlockId = fmt.Sprintf("%d", sl.TrackingSt.NavlockIdentifier)
	info.ScheduleId = scheduleId
	info.ScheduleStatus = navls.String()
	info.DeviceTag = cnf.Tag
	info.CrossLocation = sl.NavlockLocation.String()
	info.CrossLevel = level
	info.StoplineWidth = st.JTX_Zone1_Width
	info.CrossDistance = st.Cross_Dis

	{
		if sl.TrackingSt.Ref.AnimationWrap != nil {
			//设置给动画的状态
			if isUpOrDown {
				sl.TrackingSt.Ref.AnimationWrap.SetUpStoplineInfo(1, st.JTX_Zone1_Width, stateAnimation, float64(st.Cross_Dis))
			} else {
				sl.TrackingSt.Ref.AnimationWrap.SetDownStoplineInfo(1, st.JTX_Zone1_Width, stateAnimation, float64(st.Cross_Dis))
			}
		}

	}

	{
		if info.ScheduleId != "" {
			if autoTest && sl.LDB != nil {
				// 同步给平台的表
				var syncStopLine gzb_db.NavlGzbStopLineWarns
				syncStopLine.CreatedAt = time.Now()
				syncStopLine.NavigationlockId = fmt.Sprintf("葛洲坝%d#", sl.NavlockIdentifier)
				syncStopLine.ScheduleId = info.ScheduleId
				syncStopLine.ScheduleStatus = info.ScheduleStatus
				syncStopLine.DeviceTag = cnf.Tag
				if isUpOrDown {
					syncStopLine.CrossLocation = "上游"
				} else {
					syncStopLine.CrossLocation = "下游"
				}
				syncStopLine.CrossLevel = info.CrossLevel
				syncStopLine.StoplineWidth = info.StoplineWidth
				syncStopLine.CrossDistance = info.CrossDistance
				if sl.LDB != nil {
					syncStopLine.SaveToDB(sl.LDB, sl.Logger)
				}
			}
		}
	}

	var webinfo gzb_db_web.NavStopLineWarn
	webinfo.Createdat = time.Now()
	webinfo.NavlockId = sl.NavlockIdentifier
	webinfo.ScheduleId = info.ScheduleId
	webinfo.ScheduleStatus = info.ScheduleStatus
	webinfo.DeviceTag = cnf.Tag
	if isUpOrDown {
		webinfo.CrossLocation = "上游"
	} else {
		webinfo.CrossLocation = "下游"
	}
	webinfo.CrossLevel = info.CrossLevel
	webinfo.StoplineWidth = info.StoplineWidth
	webinfo.CrossDistance = info.CrossDistance
	//插入web数据库 需要做过滤的设置
	if sl.PwebDB != nil {
		webinfo.SaveToDB(sl.PwebDB, sl.Logger)
	}
	if sl.LDB != nil {
		webinfo.SaveToDB(sl.LDB, sl.Logger)
	}

}
