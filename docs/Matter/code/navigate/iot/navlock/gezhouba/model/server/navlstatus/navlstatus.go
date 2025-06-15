package navlstatus

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"runtime"
	"strings"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	gzb_api_mn "navigate/iot/navlock/gezhouba/api/navlstatus"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
	db_md "navigate/store/iot/navlock/model"
)

var (
	_ mdl.WorkerRecover = (*NavlStatus)(nil)
	_ cmpt.Cpt          = (*NavlStatus)(nil)
	_ cmpt.CptsOperator = (*NavlStatus)(nil)
	_ cmpt.CptComposite = (*NavlStatus)(nil)
	_ cmpt.CptRoot      = (*NavlStatus)(nil)
)

const (
	NavlStatusKind = cmpt.KindName("navlstatus")
	HttpDriverId   = cmpt.IdName("navlstatus-http")
)

type NavlStatus struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	Cnf             *gzb_cf.HttpClientConfig
	navlDyndata     *gzb_sche.NavlockDynamicStatusData
	tmpNavState     gzb_sche.NavStatus     //通航调度状态 --判断完整的一个航程-上行或者下行
	tmpNavlockState gzb_sche.NavlockStatus //船闸调度状态 --判断完整的一个闸室状态--进闸到出闸

	msgChanNavlockState <-chan any
	//msgChanNavState     <-chan any
}

func NewNavlStatusSvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *NavlStatus {
	tmpMnd := &NavlStatus{
		BaseCptCtx: dr,
		TrackingSt: trs,
	}
	tmpMnd.navlDyndata = &gzb_sche.NavlockDynamicStatusData{}
	//初始化设置时间
	tmpMnd.navlDyndata.BeginTime = time.Now()
	//设置worker
	tmpMnd.CptMetaSt.WorkerRecover = tmpMnd
	//设置获取其他完成的状态
	return tmpMnd
}

func (mhd *NavlStatus) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		mhd.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (mhd *NavlStatus) Start() (err error) {
	if mhd.CptSt == nil || mhd.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", mhd.CmptInfo())
	} else {
		if err = mhd.CptSt.Validate(); err != nil {
			return err
		}
	}

	if mhd.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", mhd.CmptInfo())
	}

	if mhd.Cnf.EnableDebug {
		mhd.Logger.Sugar().Debugln("--- Starting ---")
		defer mhd.Logger.Sugar().Debugln("--- Started ---")
	}

	//------------- 初始化httpdr -----------------
	if mhd.Cpts.Cpt(HttpDriverId) == nil {
		// 只有NavlStatus初始化完成后 才能取到合适的参数 所以在Start()里面实现
		//创建一个新的控制结构独立控制 httpdr 的运行
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			HttpDriverId,
			0,
			mhd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d",
				mhd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"http"))

		httpdr := drs_once.NewOnceDvr(dr, mhd.TrackingSt)
		httpdr.Fn = mhd.doHttpWork
		mhd.AddCpts(httpdr)
		//只添加组件不启动
	}
	// --------------------------------------------

	mhd.Ctrl().WaitGroup().StartingWait(mhd.WorkerRecover)
	if mhd.Cpts != nil && mhd.Cpts.Len() > 0 {
		err = mhd.Cpts.Start()
		if err != nil {
			mhd.Logger.Sugar().Errorf("Start() error : %+v", err)
		}
	}
	mhd.State.Store(true)
	return nil
}

func (mhd *NavlStatus) Stop() (err error) {
	if !mhd.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", mhd.Info())
	}
	mhd.Ctrl().Cancel()

	if mhd.Cnf.EnableDebug {
		mhd.Logger.Sugar().Debugln("--- Stopping ---")
		defer mhd.Logger.Sugar().Debugln("--- Stopped ---")
	}

	if mhd.Cpts != nil && mhd.Cpts.Len() > 0 {
		err = mhd.Cpts.Stop()
		if err != nil {
			mhd.Logger.Sugar().Errorf("Stop() error : %+v", err)
		}
	}

	<-mhd.Ctrl().Context().Done()
	mhd.State.Store(false)
	return
}

func (mhd *NavlStatus) doHttpWork(ctx context.Context) *drs_once.Result {
	urlValues := url.Values{}
	urlValues.Set("zs", fmt.Sprintf("%d", mhd.NavlockIdentifier))
	reqBody := urlValues.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, mhd.Cnf.Url, strings.NewReader(reqBody))
	if err != nil {
		mhd.Logger.Sugar().Fatalf(`http.NewRequestWithContext error: %+v`, err)
	}
	// 1. 强制使用短链接
	req.Close = true
	// 2. 添加链接关闭头
	req.Header.Add("Connection", "close")

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	// 3. 创建非长链接的tcp请求
	tr := &http.Transport{
		DisableKeepAlives: true,
	}
	client := &http.Client{
		Transport: tr,
	}
	resp, err := client.Do(req)

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (mhd *NavlStatus) Work() error {

	if mhd.Cnf.EnableDebug {
		mhd.Logger.Sugar().Debugf(`[%s]: "Work Starting,autotest:%t"`,
			mhd.Info(), mhd.Cnf.EnableAutoTest)
	}

	wkCanCreate := mhd.Cnf.Enable

	HttpWork, ok := mhd.Cpts.Cpt(HttpDriverId).(*drs_once.Once)
	if !ok {
		mhd.Logger.Sugar().Fatalf(`%s - tranform (*httpdr.HttpClientDriver) from interface failed"`,
			mhd.Info())
	}

	//发布船闸状态变更
	mhd.msgChanNavlockState = mhd.EventChans.Subscribe(gzb_sche.MsgTopic_NavlockStatus)
	//发布通航状态变更
	//mhd.msgChanNavState = mhd.EventChans.Subscribe(gzb_sche.MsgTopic_NavStatus)
	defer func() {
		mhd.EventChans.UnSubscribe(gzb_sche.MsgTopic_NavlockStatus, mhd.msgChanNavlockState)
		//mhd.EventChans.UnSubscribe(gzb_sche.MsgTopic_NavStatus, mhd.msgChanNavState)
	}()

	ticker := time.NewTicker(time.Duration(mhd.Cnf.DoTimeInterval) * time.Millisecond)
	defer ticker.Stop()

	// 在2小时内 : 通航状态,闸次号,船闸状态 均未发生变化 则自动重置调度系统
	StatesResetTicker := time.NewTicker(time.Duration(2 * time.Hour))
	defer StatesResetTicker.Stop()
	NavStateReset := gzb_sche.NavUnknown
	NavlStateReset := gzb_sche.NavlockUnknown
	ScheduleIdReset := mhd.TrackingSt.ScheduleId.GetScheduleId()
	for {
		runtime.Gosched()
		select {
		case <-mhd.Ctrl().Context().Done():
			{
				wkCanCreate = false
				if err := mhd.Ctrl().Context().Err(); err != context.Canceled && err != context.DeadlineExceeded {
					mhd.Logger.Sugar().Warnf(`[Dr:%s,%s]:" err: %+v"`,
						mhd.Cnf.Name, mhd.Info(), err)
				}
				return nil
			}
		case result, ok := <-HttpWork.Chan():
			{
				if !ok {
					if err := HttpWork.Stop(); err != nil {
						HttpWork.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}

					// 只有chan 关闭了 才算上次的工作完成了
					if !wkCanCreate {
						wkCanCreate = true
					}
				} else {
					//正常处理的情况下
					mhd.DealWith(HttpWork, result, mhd.Cnf)
				}
				result = nil
			}
		case _, ok := <-mhd.msgChanNavlockState:
			{
				if !ok {
					mhd.Logger.Sugar().Errorf(`[%s]:"Event chan closed"`, mhd.Info())
					continue
				}

				NavStateReset = mhd.PredicateSchedule.GetNavStatus()
				NavlStateReset = mhd.PredicateSchedule.GetNavlockStatus()
				ScheduleIdReset = mhd.TrackingSt.ScheduleId.GetScheduleId()
				//响应 外部事件获取到的船闸状态的发布
				switch mhd.PredicateSchedule.GetNavlockStatus() {
				// 响应重置船闸状态
				case gzb_sche.NavlockUnknown:
				case gzb_sche.NavlockUpGoOutDone:
				case gzb_sche.NavlockDownGoOutDone:
					{
						//重置通航和船闸状态缓存 为未知
						//以便进入下一次完整的判断
						mhd.tmpNavState = gzb_sche.NavUnknown
						mhd.tmpNavlockState = gzb_sche.NavlockUnknown

					}
				}
			}
		case <-ticker.C:
			{
				// drain event
				select {
				case <-ticker.C:
				default:
				}

				//创建各种drs_once
				if wkCanCreate {
					if !HttpWork.IsRunning() {
						HttpWork.Ctrl().WithTimeout(mhd.Ctrl().Context(), time.Duration(mhd.Cnf.ConnectReadTimeout)*time.Millisecond)
						if err := HttpWork.Start(); err != nil {
							HttpWork.Logger.Sugar().Errorf(`Start() error: %+v`, err)
						}
						//创建重启后设置标签 --- 1. 超时取消 2.主动取消
						wkCanCreate = false
					}
				}
			}
		case <-StatesResetTicker.C:
			{
				// drain event
				select {
				case <-StatesResetTicker.C:
				default:
				}
				// 通航 船闸 或者 闸次号 未知的情况下 不处理
				if NavStateReset == gzb_sche.NavUnknown ||
					NavlStateReset == gzb_sche.NavlockUnknown ||
					ScheduleIdReset == "" {
					continue
				}

				// 如果间隔两个小时 通航 船闸状态 和闸次号都未发生改变 则重置
				if NavStateReset == mhd.PredicateSchedule.GetNavStatus() &&
					NavlStateReset == mhd.PredicateSchedule.GetNavlockStatus() &&
					ScheduleIdReset == mhd.TrackingSt.ScheduleId.GetScheduleId() {
					mhd.PredicateSchedule.SetNavStatus(gzb_sche.NavUnknown)
					mhd.PredicateSchedule.SetNavlockStatus(gzb_sche.NavlockUnknown)
					// 通知船闸状态重置
					mhd.EventChans.Publish(gzb_sche.MsgTopic_NavlockStatus, struct{}{})
				}
			}
		}
	}

}

func (mhd *NavlStatus) DealWith(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.HttpClientConfig) error {
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
		if errors.Is(result.Err, context.Canceled) {
			mhd.Logger.Sugar().Errorf(`[Dr:%s,%s]:"http worker - exit,err : %+v"`,
				mhd.Cnf.Name, mhd.Info(), result.Err)
			return nil
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			mhd.Logger.Sugar().Warnf(`[Dr:%s,%s]:"http worker - timeout,err : %+v"`,
				mhd.Cnf.Name, mhd.Info(), result.Err)
			return nil
		}

		mhd.Logger.Sugar().Warnf(`[Dr:%s,%s]:"err : %+v"`,
			mhd.Cnf.Name, mhd.Info(), result.Err)

		return result.Err
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			mhd.Logger.Sugar().Errorf(`%s :"tranform from interface failed"`, mhd.CmptInfo())
			return nil
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			mhd.Logger.Sugar().Errorf(`[Dr:%s,%s]):"Error reading response body: %+v"`,
				mhd.Cnf.Name, mhd.Info(), err)
			return err
		}
		// mhd.Logger.Sugar().Debugf(`[Dr:%s,%s]:"Response StatusCode: %d,Responsebody: %s"`,
		// 	mhd.Cnf.DriveName, mhd.TrackInfo(), resp.StatusCode, body)
		if resp.StatusCode == 200 {
			mhd.tranformData(body)
		}
		body = nil
		return nil
	}
}

func (mhd *NavlStatus) writeRuntimeHistorytoLDB(ss string) {
	if mhd.Cnf.EnableAutoTest || mhd.LDB == nil {
		return
	}

	var info db_md.IotNavlCpntState
	ScheduleId := mhd.ScheduleId.GetScheduleId()
	info.NavlockId = fmt.Sprintf("%d", mhd.TrackingSt.NavlockIdentifier)
	info.ScheduleId = ScheduleId
	info.NavlockStatus = mhd.PredicateSchedule.GetNavlockStatus().String()
	info.NavStatus = mhd.PredicateSchedule.GetNavStatus().String()
	info.ServerName = mhd.ServerName
	info.SystemName = mhd.SystemName
	info.DriverName = mhd.DriverName

	info.FormatVersion = db_md.FormatVersionTest + 0
	info.InfoLevel = uint(db_md.LevelInfo)

	info.Info = []byte(ss)
	if err := mhd.LDB.Create(&info).Error; err != nil {
		mhd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"insert to db err: %+v"`, ScheduleId, err)
	}
}

// 船闸数据的更新
func (mhd *NavlStatus) tranformData(bys []byte) bool {
	xmlss := string(bys)
	xmlss = strings.ReplaceAll(xmlss, "\n", "")
	xmlss = strings.ReplaceAll(xmlss, "\t", "")

	// 会造成时间字符串解析出错
	//xmlss = strings.ReplaceAll(xmlss, " ", "")
	begin := strings.Index(xmlss, ">{")
	end := strings.Index(xmlss, "}<")

	var MonitorInfoStruct gzb_api_mn.NavlStatusResp
	if begin >= 0 && end >= 0 && end >= begin {
		jsonss := ""
		jsonss = xmlss[(begin + 1):(end + 1)]
		err := mdl.Json.UnmarshalFromString(jsonss, &MonitorInfoStruct)
		if err != nil {
			// program transform string error
			mhd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"tranform from xmlstring: %+v"`, mhd.ScheduleId.GetScheduleId(), err)
		}
		//mhd.Logger.Sugar().Debugf(`ScheduleId-[%s]:"解析http后的数据对应字段: MonitorInfoStruct: %+v "`, mhd.ScheduleId.GetScheduleId(), MonitorInfoStruct)
		MonitorInfoStruct.TransformValue()
		//mhd.Logger.Sugar().Debugf(`ScheduleId-[%s]:"解析后转换信号灯的数据对应字段: MonitorInfoStruct: %+v"`, mhd.ScheduleId.GetScheduleId(), MonitorInfoStruct)
	} else {
		mhd.Logger.Sugar().Errorf(`%s:"xmlss parse errno: begin:%d,end:%d"`, mhd.CmptInfo(), begin, end)
		return false
	}
	//解析船闸状态的 数据字段

	//转换前先设置转换字段为初始值
	mhd.navlDyndata.ReSet(false)
	if err := mhd.navlDyndata.TransformFrom(&MonitorInfoStruct); err != nil {
		mhd.Logger.Sugar().Errorf(`%s: error: %+v,TransformData:"%+v", SourceData:"%s"`,
			mhd.Info(), err, MonitorInfoStruct, string(bys))
		return false
	}

	mhd.writeRuntimeHistorytoLDB(fmt.Sprintf(`{"Info":"NavigationlockDynamicData: %+v"}`, mhd.navlDyndata))
	//mhd.Logger.Sugar().Debugf(`ScheduleId-[%s]:"NavigationlockDynamicData: %+v"`, mhd.ScheduleId.GetScheduleId(), mhd.navlDyndata)
	// 使用新的判断结构
	returnCode, err := mhd.navlDyndata.PredicateBaseOnPhysicalLightStates()
	mhd.writeRuntimeHistorytoLDB(fmt.Sprintf(`{"Info":"PredicateBaseOnPhysicalLightStates:{ returncode:%s,err:%+v,NavigationlockDynamicData: %+v}"}`,
		returnCode.String(), err, mhd.navlDyndata))

	//判断获取的状态
	switch returnCode {
	case gzb_sche.NavGetOK:
		{
			//直接判断出了状态 设置最近一次判断的状态和时间 水位
			// 使用协议时间
			//mhd.navlDyndata.BeginTime = time.Now()
			mhd.navlDyndata.NavStateLast = mhd.navlDyndata.NavState
			mhd.navlDyndata.NavlockStateLast = mhd.navlDyndata.NavlockState
			mhd.navlDyndata.UpInnerWaterLast = mhd.navlDyndata.TUpInnerWater
			mhd.navlDyndata.DownInnerWaterLast = mhd.navlDyndata.TDownInnerWater
			{

				if mhd.TrackingSt.Ref.AnimationWrap != nil {
					//设置动画需要的状态数据
					// 设置闸门状态上下左右 0:"闸门状态未知",1:"闸门开运行",2:"闸门开终",3:"闸门关运行",4:"闸门关终"
					// 		0: "闸门状态未知",1: "闸门开运行",2: "闸门开终",3: "闸门关运行",4: "闸门关终",
					mhd.TrackingSt.Ref.AnimationWrap.SetGatesStatus(int(mhd.navlDyndata.TUpGates.Left.State),
						int(mhd.navlDyndata.TUpGates.Right.State),
						int(mhd.navlDyndata.TDownGates.Left.State),
						int(mhd.navlDyndata.TDownGates.Right.State))
				}

			}
			mhd.navlDyndata.EndTime = time.Now()
			//copy 一份记录 插入到综合判断结构
			navlDyndata := *mhd.navlDyndata
			isNavUpDownSwitch, isNavlSwitch := mhd.PredicateSchedule.PushNavlockAutoStatus(&navlDyndata, mhd.TrackingSt, mhd.LDB, mhd.PDB, mhd.Logger)
			if isNavlSwitch || isNavUpDownSwitch {
				mhd.tmpNavlockState = mhd.PredicateSchedule.GetNavlockStatus()
				mhd.tmpNavState = mhd.PredicateSchedule.GetNavStatus()
				//switch mhd.tmpNavState {
				//case gzb_sche.NavUp, gzb_sche.NavDown:
				if isNavUpDownSwitch {
					//发布当前通航状态 只是通知下 不需要获取消息解析
					mhd.EventChans.Publish(gzb_sche.MsgTopic_NavStatus, struct{}{})
				}
				//}

				if isNavlSwitch {
					//发布当前船闸状态 只是通知下 不需要获取消息解析
					mhd.EventChans.Publish(gzb_sche.MsgTopic_NavlockStatus, struct{}{})
				}
				mhd.AssignState()
			}
		}
	case gzb_sche.NavGetNeed:
		{
			//需要再次获取后判断
		}
	case gzb_sche.NavGetNeedReset:
		{
			//需要重置后再获取判断 --- 每次获取都会重置字段 所以这个错误会避免
		}
	case gzb_sche.NavGetNeedResetTime:
		{
			//时间上大于 设置的最大5分钟 则数据以当前的为准
			mhd.navlDyndata.BeginTime = time.Now()
			mhd.navlDyndata.UpInnerWaterLast = mhd.navlDyndata.TUpInnerWater
			mhd.navlDyndata.DownInnerWaterLast = mhd.navlDyndata.TDownInnerWater
		}
	case gzb_sche.NavGetStoreAndCompare:
		{
			//需要存储内侧水位下来 再获取一次对比判断
			mhd.navlDyndata.UpInnerWaterLast = mhd.navlDyndata.TUpInnerWater
			mhd.navlDyndata.DownInnerWaterLast = mhd.navlDyndata.TDownInnerWater
		}
	}

	//mhd.Logger.Sugar().Debugf("PredicateBaseOnPhysicalStates: %s, error: %+v\n", returnCode.String(), err)
	return true
}

// 只有需要更新的时候 才使用更新状态
func (mhd *NavlStatus) AssignState() {
	navs := mhd.PredicateSchedule.GetNavStatus()
	navls := mhd.PredicateSchedule.GetNavlockStatus()

	// 只有正确符合实际情况的状态才会设置动画
	switch navs {
	case gzb_sche.NavDown:
		{
			if mhd.TrackingSt.Ref.AnimationWrap != nil {
				mhd.TrackingSt.Ref.AnimationWrap.SetNavStatus(int(navs))
			}

			switch navls {
			case gzb_sche.NavlockDownGoingIn, gzb_sche.NavlockDownGoInDone, gzb_sche.NavlockDownGoingOut, gzb_sche.NavlockDownGoOutDone,
				gzb_sche.NavlockDownWaterTrendDown:
				{
					if mhd.TrackingSt.Ref.AnimationWrap != nil {
						//1. 设置当前判断的船闸状态
						//设置动画需要的状态转换数据
						mhd.TrackingSt.Ref.AnimationWrap.SetScheduleIdAndNavlockStatus(mhd.ScheduleId.GetScheduleId(),
							mhd.PredicateSchedule.GetNavlockStatus().TransformToWebDefine())
					}

				}
			}
		}
	case gzb_sche.NavUp:
		{
			if mhd.TrackingSt.Ref.AnimationWrap != nil {
				mhd.TrackingSt.Ref.AnimationWrap.SetNavStatus(int(navs))
			}

			switch navls {
			case gzb_sche.NavlockUpGoingIn, gzb_sche.NavlockUpGoInDone, gzb_sche.NavlockUpGoingOut, gzb_sche.NavlockDownGoOutDone,
				gzb_sche.NavlockUpWaterTrendUp:
				{

					if mhd.TrackingSt.Ref.AnimationWrap != nil {
						//2. 设置当前判断的船闸状态
						//设置动画需要的状态转换数据
						mhd.TrackingSt.Ref.AnimationWrap.SetScheduleIdAndNavlockStatus(mhd.ScheduleId.GetScheduleId(),
							mhd.PredicateSchedule.GetNavlockStatus().TransformToWebDefine())
					}
				}
			}
		}
	}
}
