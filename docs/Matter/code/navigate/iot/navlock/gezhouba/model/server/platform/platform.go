package platform

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"runtime"
	"time"

	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	bs "navigate/internal/bytestr"
	gzb_api_pf "navigate/iot/navlock/gezhouba/api/platform"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	navl_md "navigate/iot/navlock/model"
)

var (
	_ mdl.WorkerRecover = (*Platform)(nil)
	_ cmpt.Cpt          = (*Platform)(nil)
	_ cmpt.CptsOperator = (*Platform)(nil)
	_ cmpt.CptComposite = (*Platform)(nil)
	_ cmpt.CptRoot      = (*Platform)(nil)
)

const (
	PlatformKind = cmpt.KindName("platform")
)

type Platform struct {
	*navl_md.BaseCptCtx
	*gzb_sche.TrackingSt

	Cnf *gzb_cf.PlatformConfig

	msgChanSpeedlimit <-chan any
	msgSpeedlimit     *gzb_api_pf.SpeedlimitRecord
	msgChanStopline   <-chan any
	msgStopline       *gzb_api_pf.StoplineRecord

	hTmsg      *gzb_api_pf.HttpTokenMsg
	initWorkOK bool
	// mCtrl                   *mdl.CtrlSt
	wkGetToken              *drs_once.Once
	wkCreateTableStopLine   *drs_once.Once
	wkCreateTableSpeedLimit *drs_once.Once
	wkStopLineInsert        *drs_once.Once
	wkSpeedLimitInsert      *drs_once.Once
}

func NewPlatformSvr(dr *navl_md.BaseCptCtx, trs *gzb_sche.TrackingSt) *Platform {
	tmp := &Platform{
		BaseCptCtx: dr,
		TrackingSt: trs,
		hTmsg:      &gzb_api_pf.HttpTokenMsg{},
	}
	tmp.WorkerRecover = tmp
	// tmp.msgChanSpeedlimit = tmp.EventChans.Subscribe(gzb_sche.MsgTopic_Platform_Speedlimit)
	// tmp.msgChanStopline = tmp.EventChans.Subscribe(gzb_sche.MsgTopic_Platform_Stopline)

	return tmp
}

func (plfd *Platform) Recover() {
	if rc := recover(); rc != nil {
		var buf [8192]byte
		n := runtime.Stack(buf[:], false)
		plfd.Logger.Sugar().Warnf(`Work() -- Recover() :%+v,Stack Trace: %s`, rc, buf[:n])
	}
}

func (plfd *Platform) Start() (err error) {
	if plfd.CptSt == nil || plfd.ModelSysSt == nil {
		return fmt.Errorf("%s is not initialized", plfd.CmptInfo())
	} else {
		if err = plfd.CptSt.Validate(); err != nil {
			return err
		}
	}

	if plfd.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", plfd.CmptInfo())
	}
	plfd.Logger.Sugar().Infof("%s is Starting", plfd.CmptInfo())

	// PlatformDriver 才能取到合适的参数 所以在Start()里面实现
	//创建一个新的控制结构独立控制 httpdr 的运行
	// plfd.mCtrl = mdl.NewCtrlSt(context.Background())
	//------------- 初始化httpdr GetToken-----------------
	if plfd.Cpts.Cpt(cmpt.IdName("platform-GetToken")) == nil {

		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("platform-GetToken"),
			0,
			plfd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", plfd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"platform-GetToken"))

		plfd.wkGetToken = drs_once.NewOnceDvr(dr, plfd.TrackingSt)
		plfd.wkGetToken.Fn = plfd.doGetTokenWork
		plfd.wkGetToken.WorkerRecover = plfd.wkGetToken
		//实现独立的创建和取消控制 不放入主流程
		//plfd.AddComponents(plfd.wkGetToken)
	}
	// --------------------------------------------

	//------------- 初始化httpdr CreateTableStopLine-----------------
	if plfd.Cpts.Cpt(cmpt.IdName("platform-CreateTableStopLine")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("platform-CreateTableStopLine"),
			0,
			plfd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", plfd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"platform-CreateTableStopLine"))

		plfd.wkCreateTableStopLine = drs_once.NewOnceDvr(dr, plfd.TrackingSt)
		plfd.wkCreateTableStopLine.Fn = plfd.doCreateTableStopLineWork
		//plfd.AddComponents(plfd.wkCreateTableStopLine)
	}
	// --------------------------------------------

	//------------- 初始化httpdr CreateTableSpeedLimit-----------------
	if plfd.Cpts.Cpt(cmpt.IdName("platform-CreateTableSpeedLimit")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("platform-CreateTableSpeedLimit"),
			0,
			plfd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", plfd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"platform-CreateTableSpeedLimit"))

		plfd.wkCreateTableSpeedLimit = drs_once.NewOnceDvr(dr, plfd.TrackingSt)
		plfd.wkCreateTableSpeedLimit.Fn = plfd.doCreateTableSpeedLimitWork
		//plfd.AddComponents(plfd.wkCreateTableSpeedLimit)
	}
	// --------------------------------------------

	//------------- 初始化httpdr StopLineInsert-----------------
	if plfd.Cpts.Cpt(cmpt.IdName("platform-StopLineInsert")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("platform-StopLineInsert"),
			0,
			plfd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", plfd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"platform-StopLineInsert"))

		plfd.wkStopLineInsert = drs_once.NewOnceDvr(dr, plfd.TrackingSt)
		plfd.wkStopLineInsert.Fn = plfd.doStopLineInsertWork
		//plfd.AddComponents(plfd.wkStopLineInsert)
	}
	// --------------------------------------------

	//------------- 初始化httpdr SpeedLimitInsert-----------------
	if plfd.Cpts.Cpt(cmpt.IdName("platform-SpeedLimitInsert")) == nil {
		cs := navl_md.NewCptSt(drs_once.OnceKind,
			cmpt.IdName("platform-SpeedLimitInsert"),
			0,
			plfd.Ctrl().ForkCtxWg())

		dr := navl_md.NewBaseCptCtx(cs,
			navl_md.NewModelSysSt(cs.CmptInfo(), fmt.Sprintf("#%d", plfd.TrackingSt.NavlockIdentifier),
				"",
				"",
				"platform-SpeedLimitInsert"))

		plfd.wkSpeedLimitInsert = drs_once.NewOnceDvr(dr, plfd.TrackingSt)
		plfd.wkSpeedLimitInsert.Fn = plfd.doSpeedLimitInsertWork
		//plfd.AddComponents(plfd.wkSpeedLimitInsert)
	}
	// --------------------------------------------

	//所有初始化工作完成后启动
	plfd.Ctrl().WaitGroup().StartingWait(plfd.WorkerRecover)
	plfd.State.Store(true)
	plfd.Logger.Sugar().Infof("%s is Started", plfd.CmptInfo())
	return nil
}

func (plfd *Platform) Stop() (err error) {
	if !plfd.State.Load().(bool) {
		return fmt.Errorf("%s is already Stopped", plfd.CmptInfo())
	}

	plfd.Ctrl().Cancel()
	plfd.Logger.Sugar().Infoln("--- Stopping ---")

	{
		if plfd.Cpts != nil && plfd.Cpts.Len() > 0 {
			err = plfd.Cpts.Stop()
			if err != nil {
				plfd.Logger.Sugar().Errorf("Stop() error : %+v", err)
			}
		}
	}
	<-plfd.Ctrl().Context().Done()
	plfd.State.Store(false)
	plfd.Logger.Sugar().Infoln("--- Stopped ---")
	return err
}

func (plfd *Platform) doGetTokenWork(ctx context.Context) *drs_once.Result {
	url := plfd.Cnf.Url + "/Token/GetToken/" + gzb_api_pf.TokenID + "/" + gzb_api_pf.TokenKey
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "apifox/1.0.0 (https://www.apifox.cn)")
	resp, err := http.DefaultClient.Do(req)
	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (plfd *Platform) doCreateTableStopLineWork(ctx context.Context) *drs_once.Result {
	url := plfd.Cnf.Url + "/DataCenter/CreateTable"
	var table gzb_api_pf.Table
	bb, _ := gzb_api_pf.EncryptEncodeWraper(bs.StringToBytes(table.StopLineStr()))
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("data", bs.BytesToString(bb))
	err := writer.Close()
	if err != nil {
		plfd.Logger.Sugar().Errorf(`writer.Close() err: %+v`, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		plfd.Logger.Sugar().Errorf(`http.NewRequestWithContext err: %+v`, err)
	}

	req.Header.Set("Authorization", "Bearer"+" "+plfd.hTmsg.Token)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("User-Agent", "apifox/1.0.0 (https://www.apifox.cn)")
	resp, err := http.DefaultClient.Do(req)
	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (plfd *Platform) doCreateTableSpeedLimitWork(ctx context.Context) *drs_once.Result {
	url := plfd.Cnf.Url + "/DataCenter/CreateTable"
	var table gzb_api_pf.Table
	bb, _ := gzb_api_pf.EncryptEncodeWraper(bs.StringToBytes(table.SpeedlimitStr()))
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("data", bs.BytesToString(bb))
	err := writer.Close()
	if err != nil {
		plfd.Logger.Sugar().Errorf(`writer.Close() err: %+v`, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		plfd.Logger.Sugar().Errorf(`http.NewRequestWithContext err: %+v`, err)
	}
	req.Header.Set("Authorization", "Bearer"+" "+plfd.hTmsg.Token)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("User-Agent", "apifox/1.0.0 (https://www.apifox.cn)")
	resp, err := http.DefaultClient.Do(req)
	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (plfd *Platform) doStopLineInsertWork(ctx context.Context) *drs_once.Result {
	url := plfd.Cnf.Url + "/DataCenter/Insert"

	var table gzb_api_pf.Table
	rs, err := table.TransformFromStopline(plfd.msgStopline)
	if err != nil {
		plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]:"tranform stopline err : %+v"`,
			plfd.Cnf.Name, plfd.Info(), err)

		return &drs_once.Result{
			Val: nil,
			Err: err,
		}
	} else {
		plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]:"tranform stopline: %+v"`,
			plfd.Cnf.Name, plfd.Info(), bs.BytesToString(rs))
	}

	bb, _ := gzb_api_pf.EncryptEncodeWraper(rs)
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("data", bs.BytesToString(bb))
	err = writer.Close()
	if err != nil {
		plfd.Logger.Sugar().Errorf(`writer.Close() err: %+v`, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		plfd.Logger.Sugar().Errorf(`http.NewRequestWithContext err: %+v`, err)
	}
	req.Header.Set("Authorization", "Bearer"+" "+plfd.hTmsg.Token)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("User-Agent", "apifox/1.0.0 (https://www.apifox.cn)")
	resp, err := http.DefaultClient.Do(req)

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (plfd *Platform) doSpeedLimitInsertWork(ctx context.Context) *drs_once.Result {
	url := plfd.Cnf.Url + "/DataCenter/Insert"

	var table gzb_api_pf.Table
	rs, err := table.TransformFromSpeedlimit(plfd.msgSpeedlimit)
	if err != nil {
		plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]:"tranform Speedlimit err : %+v"`,
			plfd.Cnf.Name, plfd.Info(), err)

		return &drs_once.Result{
			Val: nil,
			Err: err,
		}

	} else {
		plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]:"tranform Speedlimit: %+v"`,
			plfd.Cnf.Name, plfd.Info(), bs.BytesToString(rs))
	}

	bb, _ := gzb_api_pf.EncryptEncodeWraper(rs)
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("data", bs.BytesToString(bb))
	err = writer.Close()
	if err != nil {
		plfd.Logger.Sugar().Errorf(`writer.Close() err: %+v`, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		plfd.Logger.Sugar().Errorf(`http.NewRequestWithContext err: %+v`, err)
	}
	req.Header.Set("Authorization", "Bearer"+" "+plfd.hTmsg.Token)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("User-Agent", "apifox/1.0.0 (https://www.apifox.cn)")
	resp, err := http.DefaultClient.Do(req)

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (plfd *Platform) Work() error {
	plfd.Logger.Sugar().Debugf(`"Work Starting: Autotest:%t,CreateTable:%t" %s-%s`,
		plfd.Cnf.EnableAutoTest, plfd.Cnf.CreateTable,
		plfd.CptMetaSt.CmptInfo(), plfd.TrackingSt.Info())

	ticker := time.NewTicker(time.Duration(plfd.Cnf.DoTimeInterval) * time.Millisecond)
	defer ticker.Stop()

	// ConnectReadTimeout := slhd.Cnf.ConnectReadTimeout
	// if time.Duration(ConnectReadTimeout) < 2*time.Millisecond {
	// 	ConnectReadTimeout = int(5 * time.Millisecond)
	// }

	// 1. 首先获取token
	// 2. 1.成功的情况下 创建表
	// 3. 2.创建表成功的情况下 insert 不同的表
	// 4. 需要获取token的情况下 并发执行1,3
	// 5. event 数据必须不停的接收 不能执行平台数据insert的情况下 不上传

	//运行时状态 由程序自己设置
	plfd.initWorkOK = false
	wkCreatedTableSpeedLimitOK := false
	wkCreatedTableStopLineOK := false
	NotifyToExit := false

	//组件创建判断 由程序自己设置
	wkGetToken_WorkCreate := true
	wkCreateTableSpeedLimit_WorkCreate := false
	wkCreateTableStopLine_WorkCreate := false
	wkStopLineInsert_WorkCreate := false
	wkSpeedLimitInsert_WorkCreate := false

	//可以由外界设置 默认为false
	wkNeedTableCreate := plfd.Cnf.CreateTable

	for {
		runtime.Gosched()
		select {
		case <-plfd.Ctrl().Context().Done():
			{
				wkGetToken_WorkCreate = false
				wkCreateTableSpeedLimit_WorkCreate = false
				wkCreateTableStopLine_WorkCreate = false
				wkStopLineInsert_WorkCreate = false
				wkSpeedLimitInsert_WorkCreate = false
				// plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]):"cancel works"`, plfd.Cnf.DriveName, plfd.Info())
				// plfd.mCtrl.Cancel()
				// <-plfd.mCtrl.Context().Done()
				// plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]):"WaitAsync"`, plfd.Cnf.DriveName, plfd.Info())
				// plfd.mCtrl.WaitGroup().WaitAsync()
				return nil
			}

		case result, ok := <-plfd.wkGetToken.Chan():
			{
				if !ok {
					plfd.wkGetToken.Logger.Sugar().Debugf(`[Dr:%s,%s]:"work chan closed"`,
						plfd.Cnf.Name, plfd.Info())

					if err := plfd.wkGetToken.Stop(); err != nil {
						plfd.wkGetToken.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}
				} else {
					if plfd.DealWithGetToken(plfd.wkGetToken, result, plfd.Cnf) {
						// token 获取成功后 就直接关掉 由后续的token超时判断打开
						wkGetToken_WorkCreate = false

						//同步平台的初始化工作未完成的情况下
						if !plfd.initWorkOK {
							//如果是需要创建table的工作 就按照顺序设置
							if wkNeedTableCreate {
								//打开创建表的工作 由创建完后设置 plfd.initWorkOK = true
								wkCreateTableStopLine_WorkCreate = true
								wkCreateTableSpeedLimit_WorkCreate = true
							} else {
								//设置初始化状态已经完成
								plfd.initWorkOK = true
								//创建上传数据工作
								wkStopLineInsert_WorkCreate = true
								wkSpeedLimitInsert_WorkCreate = true
							}
						}

						plfd.Logger.Sugar().Debugf(`[Dr:%s,%s]):"Get set token OK"`,
							plfd.Cnf.Name, plfd.Info())
					}
				}
				result = nil
			}
		case result, ok := <-plfd.wkCreateTableStopLine.Chan():
			{
				if !ok {
					plfd.wkCreateTableStopLine.Logger.Sugar().Debugf(`[Dr:%s,%s]:"work chan closed"`,
						plfd.Cnf.Name, plfd.Info())

					if err := plfd.wkCreateTableStopLine.Stop(); err != nil {
						plfd.wkCreateTableStopLine.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}

					if !plfd.initWorkOK && !wkCreatedTableStopLineOK && !wkCreateTableStopLine_WorkCreate {
						wkCreateTableStopLine_WorkCreate = true
					}
				} else {
					if plfd.DealWithCreateTableStopline(plfd.wkCreateTableStopLine, result, plfd.Cnf) {
						// 如果创建成功 取消 并不再创建
						wkCreateTableStopLine_WorkCreate = false

						//设置创建表成功
						wkCreatedTableStopLineOK = true
						//如果都创建成功
						if wkCreatedTableSpeedLimitOK {
							//设置初始化状态已经完成
							plfd.initWorkOK = true
						}

						// 开启stopline 的插入数据工作
						wkStopLineInsert_WorkCreate = true
						plfd.Logger.Sugar().Debugf(`%s[Dr:%s,%s]:"Create Table Stopline : OK"`,
							plfd.CmptInfo(), plfd.Cnf.Name, plfd.Info())
					}
				}
				result = nil
			}
		case result, ok := <-plfd.wkCreateTableSpeedLimit.Chan():
			{
				if !ok {
					plfd.wkCreateTableSpeedLimit.Logger.Sugar().Debugf(`[Dr:%s,%s]:"work chan closed"`,
						plfd.Cnf.Name, plfd.Info())
					if err := plfd.wkCreateTableSpeedLimit.Stop(); err != nil {
						plfd.wkCreateTableSpeedLimit.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}

					if !plfd.initWorkOK && !wkCreatedTableSpeedLimitOK && !wkCreateTableSpeedLimit_WorkCreate {
						wkCreateTableSpeedLimit_WorkCreate = true
					}
				} else {
					if !NotifyToExit && plfd.DealWithCreateTableSpeedLimit(plfd.wkCreateTableSpeedLimit, result, plfd.Cnf) {
						// 如果创建成功 取消 并不再创建
						wkCreateTableSpeedLimit_WorkCreate = false

						//设置创建表成功
						wkCreatedTableSpeedLimitOK = true
						//如果都创建成功
						if wkCreatedTableStopLineOK {
							//设置初始化状态已经完成
							plfd.initWorkOK = true
						}

						// 开启SpeedLimit 的插入数据工作
						wkSpeedLimitInsert_WorkCreate = true
						plfd.Logger.Sugar().Infof(`%s[Dr:%s,%s]:"Create Table SpeedLimit : OK"`,
							plfd.CmptInfo(), plfd.Cnf.Name, plfd.Info())
					}
				}
				result = nil
			}
		case result, ok := <-plfd.wkStopLineInsert.Chan():
			{
				if !ok {
					plfd.wkStopLineInsert.Logger.Sugar().Debugf(`[Dr:%s,%s]:"work chan closed"`,
						plfd.Cnf.Name, plfd.Info())
					if err := plfd.wkStopLineInsert.Stop(); err != nil {
						plfd.wkStopLineInsert.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}
					//持续创建插入数据的工作
					if !wkStopLineInsert_WorkCreate {
						wkStopLineInsert_WorkCreate = true
					}
				} else {
					plfd.DealWithStopLineInsert(plfd.wkStopLineInsert, result, plfd.Cnf)
				}
				result = nil
			}
		case result, ok := <-plfd.wkSpeedLimitInsert.Chan():
			{
				if !ok {
					plfd.wkSpeedLimitInsert.Logger.Sugar().Debugf(`[Dr:%s,%s]:"work chan closed"`,
						plfd.Cnf.Name, plfd.Info())
					if err := plfd.wkSpeedLimitInsert.Stop(); err != nil {
						plfd.wkSpeedLimitInsert.Logger.Sugar().Errorf(`Stop() error: %+v`, err)
					}

					if !wkSpeedLimitInsert_WorkCreate {
						wkSpeedLimitInsert_WorkCreate = true
					}
				} else {
					plfd.DealWithSpeedLimitInsert(plfd.wkSpeedLimitInsert, result, plfd.Cnf)
				}
				result = nil
			}
		case result, ok := <-plfd.msgChanStopline:
			{
				if !ok {
					plfd.wkSpeedLimitInsert.Logger.Sugar().Debugf(`[Dr:%s,%s]:"Event chan closed"`,
						plfd.Cnf.Name, plfd.Info())
					NotifyToExit = true
					continue
				}

				msg, transformOK := result.(*gzb_api_pf.StoplineRecord)
				if !transformOK {
					plfd.wkSpeedLimitInsert.Logger.Sugar().Fatalf(`[Dr:%s,%s]:"Event chan msg transform gzb_api_pf.StoplineRecord failed"`,
						plfd.Cnf.Name, plfd.Info())
				} else {
					plfd.msgStopline = msg
				}

				result = nil
			}
		case result, ok := <-plfd.msgChanSpeedlimit:
			{
				if !ok {
					plfd.wkSpeedLimitInsert.Logger.Sugar().Debugf(`[Dr:%s,%s]:"Event chan closed"`,
						plfd.Cnf.Name, plfd.Info())
					NotifyToExit = true
					continue
				}

				msg, transformOK := result.(*gzb_api_pf.SpeedlimitRecord)
				if !transformOK {
					plfd.wkSpeedLimitInsert.Logger.Sugar().Fatalf(`[Dr:%s,%s]:"Event chan msg transform gzb_api_pf.StoplineRecord failed"`,
						plfd.Cnf.Name, plfd.Info())
				} else {
					plfd.msgSpeedlimit = msg
				}

				result = nil
			}
		case <-ticker.C:
			{

				//在token过期之前 判断是否要启动获取token
				if plfd.hTmsg.IsTimeToGetToken() {
					wkGetToken_WorkCreate = true
					plfd.wkGetToken.Logger.Sugar().Debugf(`[Dr:%s,%s]):"time to get token "`,
						plfd.Cnf.Name, plfd.Info())
				}

				if wkGetToken_WorkCreate {
					if !plfd.wkGetToken.IsRunning() {
						plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkGetToken.Ctrl() info:%s",
							plfd.Ctrl().DebugInfo(), plfd.wkGetToken.Ctrl().DebugInfo())
						plfd.wkGetToken.Ctrl().WithTimeout(plfd.Ctrl().Context(), time.Duration(plfd.Cnf.ConnectReadTimeout)*time.Millisecond)
						plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkGetToken.Ctrl() info:%s",
							plfd.Ctrl().DebugInfo(), plfd.wkGetToken.Ctrl().DebugInfo())
						if err := plfd.wkGetToken.Start(); err != nil {
							plfd.wkGetToken.Logger.Sugar().Errorf(`%s - Start() error: %+v`, plfd.wkGetToken.CmptInfo(), err)
						}
						wkGetToken_WorkCreate = false
					}
				}

				if !plfd.initWorkOK {
					if wkCreateTableStopLine_WorkCreate {
						if !plfd.wkCreateTableStopLine.IsRunning() {
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkCreateTableStopLine.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkCreateTableStopLine.Ctrl().DebugInfo())
							plfd.wkCreateTableStopLine.Ctrl().WithTimeout(plfd.Ctrl().Context(), time.Duration(plfd.Cnf.ConnectReadTimeout)*time.Millisecond)
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkCreateTableStopLine.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkCreateTableStopLine.Ctrl().DebugInfo())
							if err := plfd.wkCreateTableStopLine.Start(); err != nil {
								plfd.wkCreateTableStopLine.Logger.Sugar().Errorf(`%s - Start() error: %+v`, plfd.wkCreateTableStopLine.CmptInfo(), err)
							}
							wkCreateTableStopLine_WorkCreate = false
						}
					}

					if wkCreateTableSpeedLimit_WorkCreate {
						if !plfd.wkCreateTableSpeedLimit.IsRunning() {
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkCreateTableSpeedLimit.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkCreateTableSpeedLimit.Ctrl().DebugInfo())
							plfd.wkCreateTableSpeedLimit.Ctrl().WithTimeout(plfd.Ctrl().Context(), time.Duration(plfd.Cnf.ConnectReadTimeout)*time.Millisecond)
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkCreateTableSpeedLimit.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkCreateTableSpeedLimit.Ctrl().DebugInfo())
							if err := plfd.wkCreateTableSpeedLimit.Start(); err != nil {
								plfd.wkCreateTableSpeedLimit.Logger.Sugar().Errorf(`%s - Start() error: %+v`, plfd.wkCreateTableSpeedLimit.CmptInfo(), err)
							}
							wkCreateTableSpeedLimit_WorkCreate = false
						}
					}

				} else {
					if wkStopLineInsert_WorkCreate && plfd.msgStopline != nil {
						if !plfd.wkStopLineInsert.IsRunning() {
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkStopLineInsert.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkStopLineInsert.Ctrl().DebugInfo())
							plfd.wkStopLineInsert.Ctrl().WithTimeout(plfd.Ctrl().Context(), time.Duration(plfd.Cnf.ConnectReadTimeout)*time.Millisecond)
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkStopLineInsert.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkStopLineInsert.Ctrl().DebugInfo())
							if err := plfd.wkStopLineInsert.Start(); err != nil {
								plfd.wkStopLineInsert.Logger.Sugar().Errorf(`%s - Start() error: %+v`, plfd.wkStopLineInsert.CmptInfo(), err)
							}
							wkStopLineInsert_WorkCreate = false
						}
					}

					if wkSpeedLimitInsert_WorkCreate && plfd.msgSpeedlimit != nil {
						if !plfd.wkSpeedLimitInsert.IsRunning() {
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkSpeedLimitInsert.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkSpeedLimitInsert.Ctrl().DebugInfo())
							plfd.wkSpeedLimitInsert.Ctrl().WithTimeout(plfd.Ctrl().Context(), time.Duration(plfd.Cnf.ConnectReadTimeout)*time.Millisecond)
							plfd.Logger.Sugar().Debugf("plfd.mCtrl_info:%s;plfd.wkSpeedLimitInsert.Ctrl() info:%s",
								plfd.Ctrl().DebugInfo(), plfd.wkSpeedLimitInsert.Ctrl().DebugInfo())
							if err := plfd.wkSpeedLimitInsert.Start(); err != nil {
								plfd.wkSpeedLimitInsert.Logger.Sugar().Errorf(`%s - Start() error: %+v`, plfd.wkSpeedLimitInsert.CmptInfo(), err)
							}
							wkSpeedLimitInsert_WorkCreate = false
						}
					}
				}

			}
		}
	}

}

func (plfd *Platform) DealWithGetToken(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.PlatformConfig) bool {
	defer func() {
		//防止所有未释放的资源
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
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]: "http-请求超时,err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
			return false
		}

		plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		return false
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			plfd.Logger.Sugar().Errorf(`%s :"tranform from interface failed - get token"`, plfd.CmptInfo())
			return false
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			plfd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"Error reading response body: %+v"`, plfd.ScheduleId.GetScheduleId(), err)
			return false
		}

		if resp.StatusCode == 200 {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
			if len(body) == 0 {
				return false
			}
			var (
				temp gzb_api_pf.HttpTokenMsg
				Resp gzb_api_pf.Response
			)

			//2 .解码获取响应包体
			err := mdl.Json.Unmarshal(body, &Resp)
			if err != nil {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response  err: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
				return false
			} else {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), Resp)
				//3 . 解析token
				err := Resp.ParseTokenMsg(&temp)
				if err != nil {
					plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"ParseTokenMsg  err: %+v"`,
						cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
					return false
				} else {
					plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"token msg: %+v"`,
						cnf.Tag, plfd.ScheduleId.GetScheduleId(), temp)
					//成功后直接替换
					plfd.hTmsg = &temp
					return true
				}
			}

		} else {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
		}
		body = nil
		result = nil
	}

	return false
}

func (plfd *Platform) DealWithCreateTableStopline(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.PlatformConfig) bool {
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
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]: "http-请求超时,err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
			return false
		}

		plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		return false

	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			plfd.Logger.Sugar().Errorf(`%s :"tranform from interface failed - Create Table Stopline"`, plfd.CmptInfo())
			return false
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			if resp.Body != nil {
				resp.Body.Close()
			}
			plfd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"Error reading response body: %+v"`, plfd.ScheduleId.GetScheduleId(), err)
			return false
		}

		if resp.StatusCode == 200 {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
			var (
				Resp gzb_api_pf.Response
			)

			//2 .获取响应包体
			err := mdl.Json.Unmarshal(body, &Resp)
			if err != nil {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response  err: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
				return false
			} else {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), Resp)
				// 3 . 解析返回的消息
				if Resp.Flag {
					// 服务端返回成功
					return true
				}
			}

		} else {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
		}
		body = nil
		result = nil
	}

	return false
}

func (plfd *Platform) DealWithCreateTableSpeedLimit(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.PlatformConfig) bool {
	defer func() {
		//防止所有未释放的资源
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
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]: "http-请求超时,err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
			return false
		}

		plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		return false
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			plfd.Logger.Sugar().Errorf(`%s :"tranform from interface failed - get token"`, plfd.CmptInfo())
			return false
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			plfd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"Error reading response body: %+v"`, plfd.ScheduleId.GetScheduleId(), err)
			return false
		}

		if resp.StatusCode == 200 {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)

			var (
				Resp gzb_api_pf.Response
			)

			//2 .获取响应包体
			err := mdl.Json.Unmarshal(body, &Resp)
			if err != nil {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response  err: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
				return false
			} else {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), Resp)
				// 3 . 解析返回的消息
				if Resp.Flag {
					// 服务端返回成功
					return true
				}
			}

		} else {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
		}
		body = nil
		result = nil
	}

	return false
}

func (plfd *Platform) DealWithStopLineInsert(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.PlatformConfig) bool {
	defer func() {
		//防止所有未释放的资源
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
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]: "http-请求超时,err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		} else {
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		}

		return false
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			plfd.Logger.Sugar().Errorf(`%s :"tranform from interface failed - get token"`, plfd.CmptInfo())
			return false
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			plfd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"Error reading response body: %+v"`, plfd.ScheduleId.GetScheduleId(), err)
			return false
		}

		if resp.StatusCode == 200 {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
			var (
				Resp gzb_api_pf.Response
			)

			//2 .获取响应包体
			err := mdl.Json.Unmarshal(body, &Resp)
			if err != nil {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response  err: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
				return false
			} else {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), Resp)
				// 3 . 解析返回的消息
				if Resp.Flag {
					// 服务端返回成功
					plfd.msgStopline = nil
					return true
				}
			}

		} else {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
		}
		body = nil
		result = nil
	}

	return false
}

func (plfd *Platform) DealWithSpeedLimitInsert(wk *drs_once.Once, result *drs_once.Result, cnf *gzb_cf.PlatformConfig) bool {
	defer func() {
		//防止所有未释放的资源
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
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]: "http-请求超时,err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
			return false
		}

		plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"err : %+v"`, cnf.Tag, plfd.ScheduleId.GetScheduleId(), result.Err)
		if result.Val != nil {
			resp := result.Val.(*http.Response)
			if resp != nil {
				resp.Body.Close()
			}
			return false
		}
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			plfd.Logger.Sugar().Errorf(`%s :"tranform from interface failed - get token"`, plfd.CmptInfo())
			return false
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			if resp.Body != nil {
				resp.Body.Close()
			}
			plfd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"Error reading response body: %+v"`, plfd.ScheduleId.GetScheduleId(), err)
			return false
		}
		if resp.StatusCode == 200 {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
			var (
				Resp gzb_api_pf.Response
			)

			//2 .获取响应包体
			err := mdl.Json.Unmarshal(body, &Resp)
			if err != nil {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response  err: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), err)
				return false
			} else {
				plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Json.Unmarshal get token Response: %+v"`,
					cnf.Tag, plfd.ScheduleId.GetScheduleId(), Resp)
				// 3 . 解析返回的消息
				if Resp.Flag {
					//成功返回后 设置消息
					plfd.msgSpeedlimit = nil
					// 服务端返回成功
					return true
				}
			}
		} else {
			plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"Response StatusCode: %d,Responsebody: %s"`,
				cnf.Tag, plfd.ScheduleId.GetScheduleId(), resp.StatusCode, body)
		}
		body = nil
		result = nil
	}
	return false
}

// // 禁停线数据的转换和更新
// func (plfd *PlatformDriver) tranformData(ss []byte, cnf *gzb_cf.HttpClientConfig) {

// 	var slstruct gzbapi_stpl.ResponseJsonStruct
// 	err := cm.Json.UnmarshalFromString(string(ss), &slstruct)
// 	if err != nil {
// 		plfd.Logger.Sugar().Errorf(`driver:%s,ScheduleId-[%s]:"Error reading response body: %+v"`,
// 			cnf.DeviceTag, plfd.ScheduleId.GetScheduleId(), err)
// 		return
// 	}

// 	plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"ResponseJsonStruct:%+v"`, cnf.DeviceTag, plfd.ScheduleId.GetScheduleId(), slstruct)
// 	var slst gzbapi_stpl.StoplineStruct
// 	slst.TransformFrom(&slstruct)
// 	plfd.Logger.Sugar().Debugf(`driver:%s,ScheduleId-[%s]:"StoplineStruct:%+v"`, cnf.DeviceTag, plfd.ScheduleId.GetScheduleId(), slst)
// 	//slhd.writeRuntimeHistorytoLDB(fmt.Sprintf(`{Info:"StoplineStruct:%+v"}`, slst))
// 	//slhd.WriteStoplineTabletoDB(&slst, cnf)
// }

// func (slhd *PlatformDriver) writeRuntimeHistorytoLDB(ss string) {
// 	autoTest := slhd.Cnf.EnableAutoTest
// 	if !autoTest || slhd.ModelCommonStruct.LDB == nil {
// 		return
// 	}

// 	var info db.IotNavComponentRuntimeHistory
// 	info.NavlockId = slhd.ServerName
// 	info.ScheduleId = slhd.ScheduleId.GetScheduleId()
// 	info.ScheduleStatus = slhd.ScheduleState.GetNavlockStateStr()
// 	info.ServerName = slhd.ServerName
// 	info.SystemName = slhd.SystemName
// 	info.DriverName = slhd.DriverName

// 	info.FormatVersion = db.FormatVersionTest + 0
// 	info.InfoLevel = uint(db.LevelInfo)
// 	info.Info = []byte(fmt.Sprintf(`{"Info":"%s"}`, ss))
// 	if err := slhd.LDB.Create(&info).Error; err != nil {
// 		slhd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"insert to db err: %+v"`, slhd.ScheduleId.GetScheduleId(), err)
// 	}
// }

// func (slhd *PlatformDriver) writeStoplineInfotoDB(db *gorm.DB, info *gzbdb.NavStoplineWarn) {
// 	if err := db.Create(info).Error; err != nil {
// 		slhd.Logger.Sugar().Errorf(`ScheduleId-[%s]:"insert to NavStoplineRuntime err: %+v"`, slhd.ScheduleId.GetScheduleId(), err)
// 	}
// }

// // 写入数据库
// func (slhd *PlatformDriver) WriteStoplineTabletoDB(st *gzbapi_stpl.StoplineStruct, cnf *gzb_cf.HttpClientConfig) {
// 	// todo: 在实际运行的过程打开 过滤掉非报警和警告的记录
// 	if !st.DectectedWarn && !st.DectectedAlarm {
// 		return
// 	}

// 	autoTest := slhd.Cnf.EnableAutoTest
// 	if !autoTest || slhd.ModelCommonStruct.LDB == nil {
// 		return
// 	}

// 	var info gzbdb.NavStoplineWarn
// 	level := ""
// 	if st.DectectedWarn {
// 		level = "报警"
// 	}

// 	if st.DectectedAlarm {
// 		level = "警告"
// 	}

// 	info.NavlockId = slhd.ModelCommonStruct.ServerName
// 	info.ScheduleId = slhd.ScheduleId.GetScheduleId()
// 	info.ScheduleStatus = slhd.ScheduleState.GetNavlockStateStr()
// 	info.DeviceTag = cnf.DeviceTag
// 	info.CrossLocation = slhd.Location.String()
// 	info.CrossLevel = level
// 	*info.StoplineWidth = st.JTX_Zone1_Width
// 	*info.CrossDistance = st.Cross_Dis

// 	// 测试开启的情况下写入本地数据和生产数据库 否则就只写入生产数据
// 	if autoTest && slhd.LDB != nil {
// 		slhd.writeStoplineInfotoDB(slhd.LDB, &info)
// 	}

// 	if slhd.PDB != nil {
// 		slhd.writeStoplineInfotoDB(slhd.PDB, &info)
// 	}

// }
