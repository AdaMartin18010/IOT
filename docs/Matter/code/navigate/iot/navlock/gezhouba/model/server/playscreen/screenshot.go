package playscreen

import (
	"context"
	"io"
	cmd_token "navigate/common/model/token"
	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
	"net/http"
	"os"
	"runtime"
	"time"
)

func (ps *PlayScreen) cmdScreenShot(setActived bool, cmdChan chan<- cmd_token.TokenCompletor) bool {
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
		ps.Logger.Sugar().Errorf("Cmd.WaitTimeout() err:%+v,setActived:%+v.%+v", Cmd.Err(), setActived, ps.Info())
		return false
	}
}

// 主程序只需要调用该函数 即可截屏
func (ps *PlayScreen) ScreenShot() bool {
	if !ps.Cnf.EnableAutoTest {
		return false
	}
	cmdNormalOK := ps.cmdScreenShot(true, ps.wkScreenShotCmdChan)
	if cmdNormalOK {
		ps.screenShotInfo.CurrentOn = ps.screenShotStatus.IsActiced()
	}
	return true
}

// http://192.168.42.129/api/screenshot
// Method: GET
// 描述: 以png格式返回当前屏幕截图。
func (ps *PlayScreen) doHttpScreenShot(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/screenshot"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		ps.httpGetScreenShot.Logger.Sugar().Errorf("ScreenShot err:%+v.%s", err, ps.Info())
	}

	if ps.Cnf.EnableDebug {
		ps.httpGetScreenShot.Logger.Sugar().Debugf("url : %s.%s",
			url,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpGetScreenShot.Logger.Sugar().Errorf("ScreenShot err:%+v.%s", err, ps.Info())
	} else {
		filename := ps.screenShotFileName.Get() + ".png"
		out, err := os.Create(filename)
		if err != nil {
			ps.httpGetScreenShot.Logger.Sugar().Errorf("ScreenShot create file err:%+v,filepath:%s.%s", err, ps.screenShotFileName.Get(), ps.Info())
		}
		defer out.Close()

		_, err = io.Copy(out, resp.Body)
		if err != nil {
			ps.httpGetScreenShot.Logger.Sugar().Errorf("ScreenShot io.Copy err:%+v.%s", err, ps.Info())
		}
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (ps *PlayScreen) ScreenShotWork(ctx context.Context) {
	ps.Logger.Sugar().Infof(`%s%s,"ScreenShotWork Starting"`, ps.CmptInfo(), ps.Info())
	defer ps.Logger.Sugar().Infof(`%s %s,"ScreenShotWork exited."`, ps.CmptInfo(), ps.Info())

	tickerCheckInterval := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer tickerCheckInterval.Stop()

	ps.screenShotStatus.SetStatus(true)
	activeShot := false
	for {
		runtime.Gosched()
		select {
		case <-ps.Ctrl().Context().Done():
			{
				ps.screenShotStatus.SetStatus(false)
				return
			}
		case cmd, ok := <-ps.wkScreenShotCmdChan:
			{
				//获取命令 检查是否Screen Shot
				if !ok {
					ps.Logger.Sugar().Warnf(`%s%s,"ScreenShotWork ScreenShotWorkCmdChan closed"`,
						ps.CmptInfo(), ps.Info())
					continue
				}

				muCmd, transformOK := cmd.(*ProgramCmd)
				if !transformOK {
					ps.Logger.Sugar().Warnf(`%s %s,"ScreenShotWork ScreenShotWorkCmdChan: TokenCompletor transform to ProgramCmd false."`,
						ps.CmptInfo(), ps.Info())
					continue
				}
				activeShot = muCmd.SetONOrOFF
				ps.screenShotStatus.SetStatus(activeShot)
				cmd.Completed()

				if ps.Cnf.EnableDebug {
					ps.Logger.Sugar().Debugf("ScreenShotWork -get cmd form chan cap:%d,len:%d,value:%+v.%s",
						cap(ps.wkScreenShotCmdChan), len(ps.wkScreenShotCmdChan), muCmd, ps.Info())
				}

				if !activeShot {
					continue
				}

				if !ps.httpGetScreenShot.IsRunning() {
					if ps.Cnf.EnableDebug {
						ps.httpGetScreenShot.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpGetScreenShot.Ctrl().Context():%+v"`,
							ps.Ctrl().Context(), ps.httpGetScreenShot.Ctrl().Context())
					}

					ps.httpGetScreenShot.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

					if ps.Cnf.EnableDebug {
						ps.httpGetScreenShot.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpGetScreenShot.Ctrl().Context():%+v"`,
							ps.Ctrl().Context(), ps.httpGetScreenShot.Ctrl().Context())
					}

					if err := ps.httpGetScreenShot.Start(); err != nil {
						ps.httpGetScreenShot.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpGetScreenShot.Info())
					}
				}

			}
		case result, ok := <-ps.httpGetScreenShot.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpGetScreenShot.Stop(); err != nil {
						ps.httpGetScreenShot.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpGetScreenShot.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseStatusCode200(ps.httpGetScreenShot, result) {
					if ps.Cnf.EnableDebug {
						ps.httpGetScreenShot.Logger.Sugar().Debugf(`filepath:%s,ok.%s`, ps.screenShotFileName.Get(), ps.Info())
					}
					//在多次连续触发中 都只成功执行一次
					activeShot = false
				} else {
					ps.httpGetScreenShot.Logger.Sugar().Errorf(`filepath:%s,failed.%s`, ps.screenShotFileName.Get(), ps.Info())
				}
			}
		case <-tickerCheckInterval.C:
			{
				// drain the chan
				select {
				case <-tickerCheckInterval.C:
				default:
				}

				if !activeShot {
					continue
				}

				if !ps.httpGetScreenShot.IsRunning() {
					if ps.Cnf.EnableDebug {
						ps.httpGetScreenShot.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpGetScreenShot.Ctrl().Context():%+v"`,
							ps.Ctrl().Context(), ps.httpGetScreenShot.Ctrl().Context())
					}

					ps.httpGetScreenShot.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

					if ps.Cnf.EnableDebug {
						ps.httpGetScreenShot.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpGetScreenShot.Ctrl().Context():%+v"`,
							ps.Ctrl().Context(), ps.httpGetScreenShot.Ctrl().Context())
					}

					if err := ps.httpGetScreenShot.Start(); err != nil {
						ps.httpGetScreenShot.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpGetScreenShot.Info())
					}
				}

			}
		}
	}

}
