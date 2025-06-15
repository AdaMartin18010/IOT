package playscreen

import (
	"context"
	"fmt"
	"runtime"
	"time"

	mdl "navigate/common/model"
	palyscreen_api "navigate/iot/navlock/gezhouba/api/playscreen"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
)

const (
	ProgramSpeedLines12_LeftUpTag    = "left_up"
	ProgramSpeedLines12_LeftDownTag  = "left_down"
	ProgramSpeedLines12_RightUpTag   = "right_up"
	ProgramSpeedLines12_RightDownTag = "right_down"

	ProgramSpeedLines3_MiddleUpTag   = "middle_up"
	ProgramSpeedLines3_MiddleDownTag = "middle_down"

	// overSpeedText   = "超速"
	// normalSpeedText = " "
)

func (ps *PlayScreen) ProgramSpeedWork(ctx context.Context) {
	ps.Logger.Sugar().Debugf(`%s%s,"Work Starting"`, ps.CptMetaSt.CmptInfo(), ps.TrackingSt.Info())

	// 1. 获取速度信息
	// 2. 在间隔时间内聚合速度的消息
	// 3. 在激活的状态下 比较获取的速度消息 如果不同的情况下 发布给播放屏
	// 4. 非激活的状态  继续获取并聚合消息
	ps.msgChanShipSpeeds = ps.EventChans.Subscribe(gzb_sche.MsgTopic_PlayScreen_ShipSpeeds)
	defer func() {
		ps.EventChans.UnSubscribe(gzb_sche.MsgTopic_PlayScreen_ShipSpeeds, ps.msgChanShipSpeeds)
	}()

	ticker := time.NewTicker(time.Duration(ps.Cnf.DoTimeInterval) * time.Millisecond)
	defer ticker.Stop()

	//3分钟 抓屏一次
	tickerScreenShot := time.NewTicker(time.Duration(3*60) * time.Second)
	defer tickerScreenShot.Stop()

	lastActive := false
	Actived := false

	screenShot := false
	// todo:需要设置 基础的默认值 1.发布节目的默认值 2. 更新文本的默认值 都需要从配置文件中获取
	// 当前发布到播放屏 成功的速度值
	// currentShipSpeeds := gzb_sche.NewShipSpeeds()
	// 一直聚合接收速度值
	receiveShipSpeed := gzb_sche.NewShipSpeeds()
	for {
		runtime.Gosched()
		existsOverSpeed := false

		select {
		case <-ps.wkProgramSpeed.Ctrl().Context().Done():
			{
				ps.wkProgramSpeed.Logger.Sugar().Debugf(`WorkSpeedProgram exiting.%s`, ps.Info())
				ps.programSpeedStatus.SetStatus(false)
				return
			}
		case cmd, ok := <-ps.wkProgramSpeedCmdChan:
			{
				if !ok {
					ps.wkProgramSpeed.Logger.Sugar().Warnf(`WorkSpeedProgram wkProgramSpeedCmdChan closed.%s`, ps.Info())
				}

				muCmd, transformOK := cmd.(*ProgramCmd)
				if !transformOK {
					ps.wkProgramSpeed.Logger.Sugar().Warnf(`%s %s,"WorkSpeedProgram wkProgramSpeedCmdChan: TokenCompletor transform to ProgramCmd false."`,
						ps.CmptInfo(), ps.Info())
					continue
				}

				if ps.Cnf.EnableDebug {
					ps.wkProgramSpeed.Logger.Sugar().Debugf("get cmd form chan cap:%d,len:%d,value:%+v.%s",
						cap(ps.wkProgramSpeedCmdChan), len(ps.wkProgramSpeedCmdChan), muCmd, ps.Info())
				}

				Actived = muCmd.SetONOrOFF
				ps.programSpeedStatus.SetStatus(Actived)
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
		case msgs, ok := <-ps.msgChanShipSpeeds:
			{
				if !ok {
					ps.Logger.Sugar().Warnf(`WorkSpeedProgram msgChanShipSpeeds closed.%s`, ps.Info())
				}

				shipspeeds, transformOK := msgs.(*gzb_sche.ShipSpeeds)
				if !transformOK {
					ps.wkProgramSpeed.Logger.Sugar().Warnf(`WorkSpeedProgram msgChanShipSpeeds: msgs transform to ShipSpeeds false.%s`, ps.Info())
					continue
				}

				receiveShipSpeed.AggregationWith(shipspeeds)
				if ps.Cnf.Enable {
					ss, _ := shipspeeds.String()
					ass, _ := receiveShipSpeed.String()
					ps.wkProgramSpeed.Logger.Sugar().Debugf(`get msgs form Speed chans cap:%d,len:%d,value:%+v, Aggregation value:%+v.%s`,
						cap(ps.msgChanShipSpeeds), len(ps.msgChanShipSpeeds), ss, ass, ps.Info())
				}
			}
		case result, ok := <-ps.httpUpdateProgramSpeedMultiLines.Chan():
			{
				if !ok {
					// 默认就是退出的行为
					if err := ps.httpUpdateProgramSpeedMultiLines.Stop(); err != nil {
						ps.httpUpdateProgramSpeedMultiLines.Logger.Sugar().Errorf(`Stop() error: %+v.%s`,
							err, ps.httpUpdateProgramSpeedMultiLines.Info())
					}
					continue
				}

				if ps.dealWithHttpResponseBodyAndStatusCode200(ps.httpUpdateProgramSpeedMultiLines, result) {
					if ps.Cnf.Enable {
						ps.httpUpdateProgramSpeedMultiLines.Logger.Sugar().Debugf(`update program speed multi line ok: %s.%s`,
							ps.programSpeedUpdateLinesJsonContent, ps.Info())
					}

					if ps.Cnf.EnableAutoTest {
						// 只有发布更新文本行 返回成功后 才需要去抓屏
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

				// if receiveShipSpeed.LeftSpeed == nil && receiveShipSpeed.RightSpeed == nil {
				// 	continue
				// }

				// if !receiveShipSpeed.LeftSpeed.IsValid && !receiveShipSpeed.RightSpeed.IsValid {
				// 	continue
				// }

				if !Actived {
					continue
				}

				// todo : 此处 雷达速度聚合 还是有问题 没有速度数据和超时默认的文本显示
				// transform to programSpeedUpdateLinesJsonContent: {\"update\":[{\"who\":{\"x\":\"0\",\"y\":\"0\"},\"text\":\"0.0\"},{\"who\":{\"x\":\"0\",\"y\":\"80\"},\"text\":\" \"}]}
				// 每次都从配置文件中初始化json文本
				var Taglines palyscreen_api.MultilLinesTag
				if err := mdl.Json.UnmarshalFromString(ps.programSpeedLinesJsonContent, &Taglines); err != nil {
					ps.wkProgramSpeed.Logger.Sugar().Errorf(`transform program Speed multi line:%s , err: %+v.%s`,
						ps.programSpeedLinesJsonContent, err, ps.Info())
					continue
				}

				// tag 是不一样的
				if ps.TrackingSt.NavlockIdentifier == 3 {

					if receiveShipSpeed.LeftSpeed != nil && receiveShipSpeed.LeftSpeed.IsValid {
						if receiveShipSpeed.LeftSpeed.OverSpeed {
							existsOverSpeed = true
							Taglines.SetTextByTag(ProgramSpeedLines3_MiddleDownTag, ps.Cnf.OverSpeedText)
						} else {
							Taglines.SetTextByTag(ProgramSpeedLines3_MiddleDownTag, ps.Cnf.NormalSpeedText)
						}
						Taglines.SetTextByTag(ProgramSpeedLines3_MiddleUpTag, fmt.Sprintf("%.1f", receiveShipSpeed.LeftSpeed.Speed))
					}

					if receiveShipSpeed.RightSpeed != nil && receiveShipSpeed.RightSpeed.IsValid {
						if receiveShipSpeed.RightSpeed.OverSpeed {
							existsOverSpeed = true
							Taglines.SetTextByTag(ProgramSpeedLines3_MiddleDownTag, ps.Cnf.OverSpeedText)
						} else {
							Taglines.SetTextByTag(ProgramSpeedLines3_MiddleDownTag, ps.Cnf.NormalSpeedText)
						}
						Taglines.SetTextByTag(ProgramSpeedLines3_MiddleUpTag, fmt.Sprintf("%.1f", receiveShipSpeed.RightSpeed.Speed))
					}

				} else {
					//  ps.TrackingSt.NavlockIdentifier == 1 or 2
					if receiveShipSpeed.LeftSpeed != nil && receiveShipSpeed.LeftSpeed.IsValid {
						if receiveShipSpeed.LeftSpeed.OverSpeed {
							existsOverSpeed = true
							Taglines.SetTextByTag(ProgramSpeedLines12_LeftDownTag, ps.Cnf.OverSpeedText)
						} else {
							Taglines.SetTextByTag(ProgramSpeedLines12_LeftDownTag, ps.Cnf.NormalSpeedText)
						}
						Taglines.SetTextByTag(ProgramSpeedLines12_LeftUpTag, fmt.Sprintf("%.1f", receiveShipSpeed.LeftSpeed.Speed))
					}

					if receiveShipSpeed.RightSpeed != nil && receiveShipSpeed.RightSpeed.IsValid {
						if receiveShipSpeed.RightSpeed.OverSpeed {
							existsOverSpeed = true
							Taglines.SetTextByTag(ProgramSpeedLines12_RightDownTag, ps.Cnf.OverSpeedText)
						} else {
							Taglines.SetTextByTag(ProgramSpeedLines12_RightDownTag, ps.Cnf.NormalSpeedText)
						}
						Taglines.SetTextByTag(ProgramSpeedLines12_RightUpTag, fmt.Sprintf("%.1f", receiveShipSpeed.RightSpeed.Speed))
					}
				}

				if ps.Cnf.EnableAutoTest {
					if existsOverSpeed {
						screenShot = true
						ps.createScreenShotFilePathName(fmt.Sprintf("OverSpeed_%d", ps.rand.Intn(1000)))
						existsOverSpeed = false
					}
				}

				ss, err0 := Taglines.TransformTo().String()
				if err0 != nil {
					ps.wkProgramSpeed.Logger.Sugar().Debugf(`Taglines.TransformTo().String() err :%+v.%s`,
						err0, ps.Info())
				} else {
					ps.programSpeedUpdateLinesJsonContent = ss
					if ps.Cnf.EnableDebug {
						ps.wkProgramSpeed.Logger.Sugar().Debugf(`transform to programSpeedUpdateLinesJsonContent: %s.%s`,
							ps.programSpeedUpdateLinesJsonContent, ps.Info())
					}
				}

				if ps.programSpeedUpdateLinesJsonContent != "" {
					ps.startUpdateProgramSpeedMultiLines()
				}

			}
		}
	}

}
