package playscreen

import (
	"context"
	"runtime"
	"time"
)

// 初始化节目  如果初始化不成功的话 一直会重试
// 直到初始化成功
func (ps *PlayScreen) ProgramInitWork(ctx context.Context) {

	if ps.Cnf.EnableDebug {
		ps.wkProgramInit.Logger.Sugar().Debugf(`Program init begining.%s`, ps.TrackingSt.Info())
		defer ps.wkProgramInit.Logger.Sugar().Debugf(`Program init end.%s`, ps.TrackingSt.Info())
	}

	// todo: 需要读取配置文件 和检查配置文件的
	// 节目的初始化和激活
	// 1. 判断配置文件检查
	// 2. 判断清空节目 和 发布新的节目
	// 3. 完成节目的初始化工作 并进入主流程
	programInit := false

	clearLanProgramReturnOk := false
	publishProgramDefaultReturnOK := false
	publishProgramSpeedReturnOk := false

	tickerCheckInterval := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer tickerCheckInterval.Stop()
	for {
		runtime.Gosched()
		select {
		case <-ps.wkProgramInit.Ctrl().Context().Done():
			{
				ps.Logger.Sugar().Debugf(`user cancel works.%s`, ps.Info())
				ps.msgChanProgramInitNotifyOk <- programInit
				close(ps.msgChanProgramInitNotifyOk)
				return
			}
		case result, ok := <-ps.httpClearLanProgram.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpClearLanProgram.Stop(); err != nil {
						ps.httpClearLanProgram.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpClearLanProgram.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpClearLanProgram, result) {
					if ps.Cnf.EnableDebug {
						ps.httpClearLanProgram.Logger.Sugar().Debugf(`clearLanProgram: ok.%s`, ps.Info())
					}
					clearLanProgramReturnOk = true
				} else {
					ps.httpClearLanProgram.Logger.Sugar().Errorf(`clearLanProgram: failed.%s`, ps.Info())
				}
			}
		case result, ok := <-ps.httpPublishProgramDefault.Chan():
			{
				if !ok {
					if err := ps.httpPublishProgramDefault.Stop(); err != nil {
						ps.httpPublishProgramDefault.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpPublishProgramDefault.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpPublishProgramDefault, result) {
					if ps.Cnf.EnableDebug {
						ps.httpPublishProgramDefault.Logger.Sugar().Debugf(`httpPublishProgramDefault: ok.%s`, ps.Info())
					}
					publishProgramDefaultReturnOK = true
				} else {
					ps.httpPublishProgramDefault.Logger.Sugar().Errorf(`httpPublishProgramDefault: failed.%s`, ps.Info())
				}

			}
		case result, ok := <-ps.httpPublishProgramSpeed.Chan():
			{
				if !ok {
					if err := ps.httpPublishProgramSpeed.Stop(); err != nil {
						ps.httpPublishProgramSpeed.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpPublishProgramSpeed.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpPublishProgramSpeed, result) {
					if ps.Cnf.EnableDebug {
						ps.httpPublishProgramSpeed.Logger.Sugar().Debugf(`httpPublishProgramSpeed: ok.%s`, ps.Info())
					}
					publishProgramSpeedReturnOk = true
				} else {
					ps.httpPublishProgramSpeed.Logger.Sugar().Errorf(`httpPublishProgramSpeed: failed.%s`, ps.Info())
				}

			}
		case <-tickerCheckInterval.C:
			{

				// drain the chan
				select {
				case <-tickerCheckInterval.C:
				default:
				}

				if clearLanProgramReturnOk && publishProgramDefaultReturnOK && publishProgramSpeedReturnOk {
					programInit = true
					ps.Logger.Sugar().Debugf(`program  init works compeleted : %s.%s.`, ps.Cnf.Name, ps.Info())
					ps.msgChanProgramInitNotifyOk <- programInit
					close(ps.msgChanProgramInitNotifyOk)
					// 全部初始化完成
					return
				}

				if !programInit {
					if !clearLanProgramReturnOk {
						if !ps.httpClearLanProgram.IsRunning() {
							if ps.Cnf.EnableDebug {
								ps.httpClearLanProgram.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpClearLanProgram.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpClearLanProgram.Ctrl().Context())
							}

							ps.httpClearLanProgram.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

							if ps.Cnf.EnableDebug {
								ps.httpClearLanProgram.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpClearLanProgram.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpClearLanProgram.Ctrl().Context())
							}

							if err := ps.httpClearLanProgram.Start(); err != nil {
								ps.httpClearLanProgram.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpClearLanProgram.Info())
							}
						}
					}

					if clearLanProgramReturnOk && !publishProgramDefaultReturnOK {
						if !ps.httpPublishProgramDefault.IsRunning() {
							if ps.Cnf.EnableDebug {
								ps.httpPublishProgramDefault.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpPublishProgramDefault.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpPublishProgramDefault.Ctrl().Context())
							}

							ps.httpPublishProgramDefault.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

							if ps.Cnf.EnableDebug {
								ps.httpPublishProgramDefault.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpPublishProgramDefault.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpPublishProgramDefault.Ctrl().Context())
							}

							if err := ps.httpPublishProgramDefault.Start(); err != nil {
								ps.httpPublishProgramDefault.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpPublishProgramDefault.Info())
							}
						}
					}

					if clearLanProgramReturnOk && publishProgramDefaultReturnOK && !publishProgramSpeedReturnOk {
						if !ps.httpPublishProgramSpeed.IsRunning() {
							if ps.Cnf.EnableDebug {
								ps.httpPublishProgramSpeed.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpPublishProgramSpeed.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpPublishProgramSpeed.Ctrl().Context())
							}

							ps.httpPublishProgramSpeed.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

							if ps.Cnf.EnableDebug {
								ps.httpPublishProgramSpeed.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpPublishProgramSpeed.Ctrl().Context():%+v"`,
									ps.Ctrl().Context(), ps.httpPublishProgramSpeed.Ctrl().Context())
							}

							if err := ps.httpPublishProgramSpeed.Start(); err != nil {
								ps.httpPublishProgramSpeed.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpPublishProgramSpeed.Info())
							}
						}
					}
				}
			}
		}
	}
}
