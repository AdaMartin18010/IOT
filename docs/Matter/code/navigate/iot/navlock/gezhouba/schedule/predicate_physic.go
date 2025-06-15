package schedule

import (
	"fmt"
	"time"
)

// ********************************************************************
// 基本的概念范畴定义  只判断船闸状态 不判断通航状态
// 1.通航 --- 分船闸倒向 上下行切换 和正常的运营即实际有船通航 --- 需要连续并持续的信息判断
// 2.船闸状态  --- 只和当前的获取的船闸状态值有关系 与实际场景运营层面的通不通船无关
// 船闸状态的判断 --- 分从闸门和信号灯的不同角度来实现
// 3.船闸系统调度事件的产生  --- 只有符合通航和船闸状态判断条件的情况下 才触发系统事件
// 4.船闸系统调度事件 --- 是系统产生控制和行为的状态判断
// 5.闸次 --- 是系统产生用来标识通航中通船的所有有效数据的标识
// 包含实际有通船的完整的一个上下行进闸到出闸完成的过程
// ********************************************************************

// 针对转换的数据 提示给调用函数
type NavGetNoted uint

func (nn *NavGetNoted) String() string {
	switch *nn {
	case NavGetOK:
		return "Predicated Ok"
	case NavGetNeed:
		return "Need get agagin"
	case NavGetNeedReset:
		return "Need Reset and get agagin"
	case NavGetNeedResetTime:
		return "Need reset PredicateBeginTime and get agagin"
	case NavGetStoreAndCompare:
		return "Need store water and get agagin to compare"
	default:
		return "Unknown"
	}
}

const (
	//NavGetOK: 直接判断出了状态
	NavGetOK NavGetNoted = 0
	//NavGetNeed: 需要再次获取后判断
	NavGetNeed NavGetNoted = 1
	//NavGetNeedReset: 需要重置所有记录后再次获取后判断
	NavGetNeedReset NavGetNoted = 2
	//NavGetNeedResetTime: 需要重置推断的时间后 再次获取后判断
	NavGetNeedResetTime NavGetNoted = 3
	//NavGetStoreAndCompare: 需要存储下来 再获取一次对比判断----获取时间短-大于2秒小于5秒的情况下对比水位
	NavGetStoreAndCompare NavGetNoted = 4
)

// todo: 测试的数据对 但是不符合实际的运营情况
// 以闸室的物理状态为准结合绿灯的状态来判断实际的调度状态 一个完整的闸次 只以 入闸到出闸为基本单位  不符合的状态下 需要等到下一次入闸时开启一个新的闸次
// isNeedStore: 见前面字段值的注释
// err 返回出错的打印信息
// 注意:	调用后 能获取到通航状态和闸室状态的情况下
// 需要调用方设置NavStatusLastTime,NavlockStatusLastTime的值
// 需要持续判断的情况下 需要设置PredicateBeginTime 或者 PredicateEndTime 的值
func (nds *NavlockDynamicStatusData) PredicateBaseOnPhysicalGatesStates() (
	isNeedStore NavGetNoted, err error) {
	err = nil

	//做推断前 重置推断的通航 闸室状态 为未知状态
	//nds.NavState = 0
	nds.NavlockState = 0

	// 以闸室的物理状态为准结合绿灯来判断通航标志
	// 判断闸门的状态只需要取左右闸门一个就好
	// 上行 进闸 出闸 --- 上下左信号灯 必有且只有一个绿灯开启
	// 下行 进闸 出闸 --- 上下右信号灯 必有且只有一个绿灯开启
	// 上行进闸:下左信号灯绿灯开启 上行出闸:上左信号灯绿灯开启
	// 下行进闸:上右信号灯绿灯开启 下行出闸:下右信号灯绿灯开启
	switch nds.TUpGates.Left.State {
	case GateOpening, GateOpened:
		{
			// 上游闸门开启中或者开启 下游闸门关闭 则为--{上行出闸,下行进闸}
			// 加强条件判断水位 取消
			if nds.TDownGates.Right.State == GateClosed {
				//如果上左绿灯开启 闸室状态 则为 上行出闸; 通航状态 则为 上行
				if nds.TUpSignalLights.Left == GreenLightOn {
					nds.NavlockState = NavlockUpGoingOut
					nds.NavState = NavUp
					if (nds.TUpInnerWater - nds.TDownWater) < WaterTrendUp {
						err = fmt.Errorf("(UpInnerWater - DownWater) < %.3f", WaterTrendUp)
					}
					return NavGetOK, err
				}

				//如果上右绿灯开启 闸室状态 则为 下行进闸中; 通航状态 则为 下行
				if nds.TUpSignalLights.Right == GreenLightOn {
					nds.NavlockState = NavlockDownGoingIn
					nds.NavState = NavDown
					if (nds.TUpInnerWater - nds.TDownWater) < WaterTrendUp {
						err = fmt.Errorf("(UpInnerWater - DownWater) < %.3f", WaterTrendUp)
					}
					return NavGetOK, err
				}

				//未确定的情况下 需要再次获取后判断 不用设置错误
				return NavGetNeed, err
			}
		}
	case GateClosing:
		{
			// 上游闸门关闭中 下游闸门关闭 则为--{上行出闸完毕,下行进闸完毕}
			if nds.TDownGates.Right.State == GateClosed {
				//如果上左绿灯开启 闸室状态 则为 上行出闸完毕; 通航状态 则为 上行
				if nds.TUpSignalLights.Left == GreenLightOn {
					nds.NavlockState = NavlockUpGoOutDone
					nds.NavState = NavUp
					if (nds.TUpInnerWater - nds.TDownWater) < WaterTrendUp {
						err = fmt.Errorf("(UpInnerWater - DownWater) < %.3f", WaterTrendUp)
					}
					return NavGetOK, err
				}

				//如果上右绿灯开启 闸室状态 则为 下行进闸完毕; 通航状态 则为 下行
				if nds.TUpSignalLights.Right == GreenLightOn {
					nds.NavlockState = NavlockDownGoInDone
					nds.NavState = NavDown
					if (nds.TUpInnerWater - nds.TDownWater) < WaterTrendUp {
						err = fmt.Errorf("(UpInnerWater - DownWater) < %.3f", WaterTrendUp)
					}
					return NavGetOK, err
				}
				//未确定的情况下 需要再次获取后判断 不用设置错误
				return NavGetNeed, err
			}
		}
	case GateClosed:
		{
			switch nds.TDownGates.Right.State {
			case GateOpening:
			case GateOpened:
				{
					//上游闸门关闭 下游闸门 开启中或者开启 则为{上行进闸,下行出闸}
					//如果下左绿灯开启 闸室状态 则为 上行进闸; 通航状态 则为 上行
					if nds.TDownSignalLights.Left == GreenLightOn {
						nds.NavlockState = NavlockUpGoingIn
						nds.NavState = NavUp
						if (nds.TDownInnerWater - nds.TUpWater) > WaterTrendDown {
							err = fmt.Errorf("(DownInnerWater - UpWater) > %.3f", WaterTrendDown)
						}
						return NavGetOK, err
					}

					//如果下右绿灯开启 闸室状态 则为 下行出闸; 通航状态 则为 下行
					if nds.TDownSignalLights.Right == GreenLightOn {
						nds.NavlockState = NavlockDownGoingOut
						nds.NavState = NavDown
						if (nds.TDownInnerWater - nds.TUpWater) > WaterTrendDown {
							err = fmt.Errorf("(DownInnerWater - UpWater) > %.3f", WaterTrendDown)
						}
						return NavGetOK, err
					}

					//未确定的情况下 需要再次获取后判断 不用设置错误
					return NavGetNeed, err
				}
			case GateClosing:
				{
					// 上游闸门关闭 下游闸门关闭中 则为--{上行进闸完毕,下行出闸完毕}
					// 如果下左绿灯开启 闸室状态 则为 上行进闸完毕; 通航状态 则为 上行
					if nds.TDownSignalLights.Left == GreenLightOn {
						nds.NavlockState = NavlockUpGoInDone
						nds.NavState = NavUp
						if (nds.TDownInnerWater - nds.TUpWater) > WaterTrendDown {
							err = fmt.Errorf("(DownInnerWater - UpWater) > %.3f", WaterTrendDown)
						}
						return NavGetOK, err
					}

					//如果下右绿灯开启 闸室状态 则为 下行出闸完毕; 通航状态 则为 下行
					if nds.TDownSignalLights.Right == GreenLightOn {
						nds.NavlockState = NavlockDownGoOutDone
						nds.NavState = NavDown
						if (nds.TDownInnerWater - nds.TUpWater) > WaterTrendDown {
							err = fmt.Errorf("(DownInnerWater - UpWater) > %.3f", WaterTrendDown)
						}
						return NavGetOK, err
					}

					//未确定的情况下 需要再次获取后判断 不用设置错误
					return NavGetNeed, err
				}
			case GateClosed:
				{
					// 这种情况下
					// 一 .判断时间 --从而确定上次的通航状态有效
					// 二. 判断信号灯和上次推断的通航状态--- 如果有信号灯状态则可以 快速判断闸室可能转换的状态
					// 三. 前两者都无法判断的情况下 - 再判断闸室水位的涨退趋势 - 因为这个阶段的时间持续大于10分钟所以尽可能的使用水位趋势来判断

					// 一.首先判断 推断的时间间隔
					//		1.如果时间为零值则无法判断 则需要设置判断的时间
					if (nds.BeginTime == time.Time{}) || nds.BeginTime.IsZero() {
						// 原则上出现编程遗漏才会出现
						// 返回重新获取并设置时间
						return NavGetNeedResetTime, fmt.Errorf("operation error: Need reset PredicateBeginTime")
					}

					//		2.如果时间间隔当前少于 推断的时间间隔PredicateIntervalSecond秒 则需要再次获取后判断
					if time.Since(nds.BeginTime) < PredicateMinIntervalSecond*time.Second {
						// 如果上次推断的时间少于需要推断的时间
						// 原则上获取时间短于设定的时间才会出现
						// 返回重新获取 提示需要的间隔时间
						return NavGetNeed, fmt.Errorf("operation error:(Now - PredicateBeginTime) < %d seconds", PredicateMinIntervalSecond)
					}

					//		3.如果时间间隔当前大于 推断的时间间隔PredicateIntervalMinute分钟 则需要重置并且重新获取
					if time.Since(nds.BeginTime) > PredicateMaxIntervalMinute*time.Minute {
						// 如果上次推断的时间间隔当前时间 大于需要推断的时间
						// 返回重新获取并设置时间
						return NavGetNeedResetTime, fmt.Errorf("operation error: (Now - PredicateBeginTime) > %d Minutes", PredicateMaxIntervalMinute)
					}

					// 二 . 上游闸门关闭 下游闸门关闭 则为--{上下行进出闸完毕,空闸}
					// ----如果这个时候刚好还能获取到信号灯的状态和上次推断的状态 则可以直接判断
					// -- 这部分逻辑需要信号灯状态正确 并且刚好在状态转换前被获取到 时间上概率很低 同时可能引入错误的通航状态
					// -- 因为后续马上会有信号灯都熄灭的情况 在这种情况下 通航状态只能通过水位趋势来判断了
					// 所以直接跳过后判断水位趋势
					// if nds.UpSignalLights.Left != 0 ||
					// 	nds.UpSignalLights.Right != 0 ||
					// 	nds.DownSignalLights.Left != 0 ||
					// 	nds.DownSignalLights.Right != 0 {
					// 	return NavGetNeed, nil
					// }

					//上下游闸门全关 信号灯信号全无 并且时间上也符合的情况下:
					//  对比闸室内侧水位趋势
					//如果闸室内侧最近一次水位有值
					if nds.DownInnerWaterLast > 0.0 && nds.UpInnerWaterLast > 0.0 {
						//判断水位的趋势 需要上下游内侧两个趋势都符合的情况下
						upWaterTrend := nds.TUpInnerWater - nds.UpInnerWaterLast
						downWaterTrend := nds.TDownInnerWater - nds.DownInnerWaterLast
						if upWaterTrend > WaterTrendUp && downWaterTrend > WaterTrendUp {
							//涨水的情况下 上行进闸完成 通航上行
							nds.NavlockState = NavlockUpGoInDone
							nds.NavState = NavUp
							return NavGetOK, err
						} else if upWaterTrend < WaterTrendDown && downWaterTrend < WaterTrendDown {
							//水位下降的情况下 下行进闸完成 通航下行
							nds.NavlockState = NavlockDownGoInDone
							nds.NavState = NavDown
							return NavGetOK, err
						} else {
							//趋势判断不出来 说明需要再次获取后判断
							return NavGetNeed, err
						}
					} else {
						//如果闸室内侧最近一次水位无值
						//没有水位趋势的情况下 需要再次获取 并告知需要获取并存储水位值
						return NavGetStoreAndCompare, fmt.Errorf("operation error: need DownInnerWaterLasttime && UpInnerWaterLasttime")
					}

				}

			}
		}
	}

	//如果以上均无法判断 则需要再次获取
	return NavGetNeed, err
}

// 判断水位的趋势
func (nds *NavlockDynamicStatusData) predicateOnWaterTrend() (
	isNeedStore NavGetNoted, err error) {

	// 这种情况下
	// 一 .判断时间 --从而确定判断的状态有效符合实际情况
	// 二 .再判断闸室水位的涨退趋势 - 因为这个阶段的时间持续大于10分钟所以尽可能的使用水位趋势来判断
	{
		// 一.首先判断 推断的时间间隔
		//		1.如果时间为零值则无法判断 则需要设置判断的时间
		if (nds.BeginTime == time.Time{}) || nds.BeginTime.IsZero() {
			// 原则上出现编程遗漏才会出现
			// 返回重新获取并设置时间
			return NavGetNeedResetTime, fmt.Errorf("operation error: Need reset PredicateBeginTime")
		}

		//		2.如果时间间隔当前少于 推断的时间间隔PredicateIntervalSecond秒 则需要再次获取后判断
		if time.Since(nds.BeginTime) < PredicateMinIntervalSecond*time.Second {
			// 如果上次推断的时间少于需要推断的时间
			// 原则上获取时间短于设定的时间才会出现
			// 返回重新获取 提示需要的间隔时间
			return NavGetNeed, fmt.Errorf("operation error:(Now - PredicateBeginTime) < %d seconds", PredicateMinIntervalSecond)
		}

		//		3.如果时间间隔当前大于 推断的时间间隔PredicateIntervalMinute分钟 则需要重置并且重新获取
		if time.Since(nds.BeginTime) > PredicateMaxIntervalMinute*time.Minute {
			// 如果上次推断的时间间隔当前时间 大于需要推断的时间
			// 返回重新获取并设置时间
			return NavGetNeedResetTime, fmt.Errorf("operation error: (Now - PredicateBeginTime) > %d Minutes", PredicateMaxIntervalMinute)
		}
	}

	{
		//二 .再判断闸室水位的涨退趋势 对比闸室内侧水位趋势
		//如果闸室内侧最近一次水位有值
		if nds.DownInnerWaterLast > 0.0 && nds.UpInnerWaterLast > 0.0 {
			//判断水位的趋势 需要上下游内侧两个趋势都符合
			upWaterTrend := nds.TUpInnerWater - nds.UpInnerWaterLast
			downWaterTrend := nds.TDownInnerWater - nds.DownInnerWaterLast
			if upWaterTrend > WaterTrendUp && downWaterTrend > WaterTrendUp {
				//船闸状态 上行水位上涨
				nds.NavlockState = NavlockUpWaterTrendUp
				return NavGetOK, err
			} else if upWaterTrend < WaterTrendDown && downWaterTrend < WaterTrendDown {
				//船闸状态 下行水位下降
				nds.NavlockState = NavlockDownWaterTrendDown
				return NavGetOK, err
			} else {
				//趋势判断不出来 说明需要再次获取后判断
				return NavGetNeed, err
			}
		} else {
			//如果闸室内侧最近一次水位无值
			//没有水位趋势的情况下 需要再次获取 并告知需要获取并存储水位值
			return NavGetStoreAndCompare, fmt.Errorf("operation error: need DownInnerWaterLasttime && UpInnerWaterLasttime")
		}
	}

}

// todo: 按照10-16梳理的运营工艺流程来实现 --- todo的标签表示 需要关注和注意
// 以闸室绿灯的状态 为准 来判断实际的调度状态 一个完整的闸次 只以 入闸到出闸为基本单位  不符合的状态下 需要等到下一次入闸时开启一个新的闸次
// isNeedStore: 见前面字段值的注释
// err 返回出错的打印信息
// 注意: 调用后 能获取到通航状态和闸室状态的情况下
// 需要调用方设置NavStatusLastTime,NavlockStatusLastTime的值
// 需要持续判断的情况下 需要设置BeginTime 或者 EndTime 的值
func (nds *NavlockDynamicStatusData) PredicateBaseOnPhysicalLightStates() (
	isNeedStore NavGetNoted, err error) {
	err = nil

	// //做推断前 重置推断的通航 闸室状态 为未知状态
	// nds.NavState = NavUnknown
	// nds.NavlockState = NavlockUnknown

	//如果上下游全部的信号灯绿灯关闭 此时说明 人已经点击船闸联动程序 船闸已经进入自动状态转换的流程
	if nds.TUpSignalLights.IsAllEqualState(GreenLightOff) && nds.TDownSignalLights.IsAllEqualState(GreenLightOff) {
		// tag: 船闸状态处于 上下行进入完成的状态 此时需要结合判断水位来处理
		// 进闸完成时-闸门关运行的情况下-绿灯即关闭;出闸开启时-闸门开终的情况下-绿灯才开启
		// so 绿灯全部关闭的情况下 需判断闸门的关运行

		// 1 tag: 此时如果刚好能获取到闸门 关运行的状态下 就能快速做出判断
		{
			// 如果上游闸门关运行 下行进闸完成
			if nds.TUpGates.IsAllEqualState(GateClosing) {
				nds.NavlockState = NavlockDownGoInDone
				return NavGetOK, err
			}

			// 如果下游闸门关运行 上行进闸完成
			if nds.TDownGates.IsAllEqualState(GateClosing) {
				nds.NavlockState = NavlockUpGoInDone
				return NavGetOK, err
			}
		}

		// 2 tag: 如果全部闸门关终 无法快速判断是船闸的上下行状态 只有通过水位动态趋势来判断船闸的状态
		// tips - 直接比对当前获取的水位高低是不行的,因为只是一个状态会误判,只有水位上涨下降的趋势才能做出判定
		if nds.TUpGates.IsAllEqualState(GateClosed) &&
			nds.TDownGates.IsAllEqualState(GateClosed) {
			return nds.predicateOnWaterTrend()
		}

	} else {
		// 如果上下游的信号灯 非全关闭

		// tag: 不存在所有信号绿灯打开的情况 需要针对单独的信号灯来判断 以下是推理的依据:
		// 1. 上行 进闸 出闸 --- 上下游左信号灯 必有且只有一个绿灯开启
		// 2. 下行 进闸 出闸 --- 上下游右信号灯 必有且只有一个绿灯开启
		// 3. 进闸完成时-闸门 关运行 的情况下-绿灯即关闭;出闸开启时-闸门 开终 的情况下-绿灯才开启
		// 4. 上行进闸:下左信号灯绿灯开启 上行出闸:上左信号灯绿灯开启
		// 5. 下行进闸:上右信号灯绿灯开启 下行出闸:下右信号灯绿灯开启
		// 6. 出闸的绿灯均由联动程序自动打开 进闸的绿灯需要人工设置

		//1. 优先判断自动打开的绿灯 -- 因为一旦打开联动程序 自动产生绿灯的状态就不可逆 就是确定的
		{
			// tag: 上行出闸
			// 上游左绿灯开启  上行出闸 系统自动产生的此事件 此时上游闸门状态是全部 开终
			if nds.TUpSignalLights.Left == GreenLightOn {
				nds.NavlockState = NavlockUpGoingOut
				return NavGetOK, err
			}

			// tag: 下行出闸
			// 下游右绿灯开启  下行出闸 系统自动产生的此事件 此时下游闸门状态是全部 开终
			if nds.TDownSignalLights.Right == GreenLightOn {
				nds.NavlockState = NavlockDownGoingOut
				return NavGetOK, err
			}
		}

		//2. 再次判断人工打开的绿灯 需要持续的判断 用最新的状态覆盖掉旧状态 因为人工存在误操作
		// todo : 持续几次 和 每次间隔时间 来断定是该状态 进闸状态 需要雷达判断是否能测到来船的数据
		{
			// tag: 下行进闸中
			// 船闸状态只关注闸门 由人操作属于运营的 所以绿灯的判断放到组合判断函数层
			if nds.TUpGates.IsAllEqualState(GateOpened) {
				nds.NavlockState = NavlockDownGoingIn
				return NavGetOK, err
			}

			// tag: 上行进闸中
			// 船闸状态只关注闸门 由人操作属于运营的 所以绿灯的判断放到组合判断函数层
			if nds.TDownGates.IsAllEqualState(GateOpened) {
				nds.NavlockState = NavlockUpGoingIn
				return NavGetOK, err
			}
		}

	}

	//如果以上均无法判断 则需要再次获取
	return NavGetNeed, err
}
