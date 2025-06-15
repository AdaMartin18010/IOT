package playscreen

import (
	"context"
	"fmt"
	"runtime"
	"time"

	mdl "navigate/common/model"
	palyscreen_api "navigate/iot/navlock/gezhouba/api/playscreen"
)

const (
	defaultProgramLinesTag = "middle"
)

func (ps *PlayScreen) ProgramDefaultWork(ctx context.Context) {
	ps.Logger.Sugar().Debugf(`%s%s,"ProgramDefaultWork Starting"`, ps.CmptInfo(), ps.Info())

	ticker := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer ticker.Stop()

	//5分钟 抓屏一次
	tickerScreenShot := time.NewTicker(time.Duration(5*60) * time.Second)
	defer tickerScreenShot.Stop()

	lastActive := false
	Actived := false
	screenShot := false
	for {
		runtime.Gosched()
		select {
		case <-ps.wkProgramDefault.Ctrl().Context().Done():
			{
				ps.wkProgramDefault.Logger.Sugar().Debugf(`%s %s,"ProgramDefaultWork exiting"`, ps.CmptInfo(), ps.Info())
				ps.programDefaultStatus.SetStatus(false)
				return
			}
		case cmd, ok := <-ps.wkProgramDefaultCmdChan:
			{
				//获取命令 设置当前 节目工作者是否为激活状态
				if !ok {
					ps.wkProgramDefault.Logger.Sugar().Warnf(`%s%s,"ProgramDefaultWork ProgramDefaultCmdChan closed"`, ps.CmptInfo(), ps.Info())
				}

				muCmd, transformOK := cmd.(*ProgramCmd)
				if !transformOK {
					ps.wkProgramDefault.Logger.Sugar().Warnf(`%s %s,"ProgramDefaultWork ProgramDefaultWorkCmdChan: TokenCompletor transform to ProgramCmd false."`, ps.CmptInfo(), ps.Info())
					continue
				}

				if ps.Cnf.EnableDebug {
					ps.wkProgramDefault.Logger.Sugar().Debugf("ProgramDefaultWork get cmd form chan cap:%d,len:%d,value:%+v.%s", cap(ps.wkProgramDefaultCmdChan), len(ps.wkProgramDefaultCmdChan), muCmd, ps.Info())
				}

				Actived = muCmd.SetONOrOFF
				ps.programDefaultStatus.SetStatus(Actived)
				cmd.Completed()
				if ps.Cnf.EnableAutoTest {
					if Actived != lastActive {
						lastActive = Actived
						if Actived {
							ps.createScreenShotFilePathName(fmt.Sprintf("Actived_%d", ps.rand.Intn(1000)))
						} else {
							ps.createScreenShotFilePathName(fmt.Sprintf("DeActived_%d", ps.rand.Intn(1000)))
						}
						screenShot = true
					}
				}
			}
		case result, ok := <-ps.httpUpdateProgramDefaultMultiLines.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpUpdateProgramDefaultMultiLines.Stop(); err != nil {
						ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpUpdateProgramDefaultMultiLines.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpUpdateProgramDefaultMultiLines, result) {
					if ps.Cnf.Enable {
						ps.Logger.Sugar().Debugf(`ProgramDefaultWork update program default multi line ok: %s.%s`,
							ps.programDefaultUpdateLinesJsonContent, ps.Info())
					}

					if ps.Cnf.EnableAutoTest {
						ps.createScreenShotFilePathName(fmt.Sprintf("UpdateLines_%d", ps.rand.Intn(1000)))
						screenShot = true
					}
				}

			}
		case <-tickerScreenShot.C:
			{
				// drain the chan
				select {
				case <-tickerScreenShot.C:
				default:
				}

				if !ps.Cnf.EnableAutoTest {
					continue
				}

				if !Actived {
					continue
				}

				if screenShot {
					if ps.ScreenShot() {
						screenShot = false
					}
				} else {
					ps.createScreenShotFilePathName(fmt.Sprintf("SomeMinShot_%d", ps.rand.Intn(1000)))
					ps.ScreenShot()
				}

			}
		case <-ticker.C:
			{
				// drain the chan
				select {
				case <-ticker.C:
				default:
				}

				if !Actived {
					continue
				}

				//为激活状态下 每次从数据库获取play screen的用户设置 然后展示出来
				if ps.PwebDB != nil {
					//加载play screen的用户设置
					if notSame, err := ps.TrackingSt.Ref.PlayScreenSetupWrap.LoadFromDB(ps.PwebDB, ps.Logger); err != nil {
						ps.wkProgramDefault.Logger.Sugar().Errorf(`%s:PlayScreen setup  load form db  err: %+v.%s`,
							ps.CmptInfo(), err, ps.Info())
					} else {
						if ps.Cnf.EnableAutoTest {
							if notSame {
								screenShot = true
								ps.createScreenShotFilePathName(fmt.Sprintf("UserSetNoSameText_%d", ps.rand.Intn(1000)))
							}
						}
					}

					if ps.Cnf.EnableDebug {
						ps.wkProgramDefault.Logger.Sugar().Debugf(`%s:PlayScreenSetupWrap LoadFromDB: %s.%s`,
							ps.CmptInfo(), ps.TrackingSt.Ref.PlayScreenSetupWrap.Infos(), ps.Info())
					}

					var Taglines palyscreen_api.MultilLinesTag
					//每次都从原始内容中 加载 再修改成数据库的内容 再转换成原始更新内容
					if err := mdl.Json.UnmarshalFromString(ps.programDefaultLinesJsonContent, &Taglines); err != nil {
						ps.wkProgramDefault.Logger.Sugar().Errorf(`transform program default multi line:%s , err: %+v.%s`,
							ps.programDefaultLinesJsonContent, err, ps.Info())
						continue
					} else {
						Taglines.SetTextByTag(defaultProgramLinesTag, ps.TrackingSt.Ref.PlayScreenSetupWrap.GetSetup())
						ss, err0 := Taglines.TransformTo().String()
						if err0 != nil {
							ps.wkProgramDefault.Logger.Sugar().Errorf(`Taglines.TransformTo().String() err: %+v.%s`,
								err0, ps.Info())
						} else {
							ps.programDefaultUpdateLinesJsonContent = ss
							if ps.Cnf.EnableDebug {
								ps.wkProgramDefault.Logger.Sugar().Debugf(`transform to programDefaultUpdateLinesJsonContent: %s.%s`,
									ps.programDefaultUpdateLinesJsonContent, ps.Info())
							}
						}
					}
				}

				if ps.programDefaultUpdateLinesJsonContent != "" {
					ps.startUpdateProgramDefaultMultiLines()
				}

			}
		}
	}

}
