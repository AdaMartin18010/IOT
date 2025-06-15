package schedule

import (
	"fmt"
	"sync"
	"time"

	cm "navigate/common/model"
	db_navlock "navigate/store/iot/navlock/model"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

// *********************************************************************************
// 船闸运营调度的基本的概念范畴定义  综合 船闸状态 推断通航状态 提供参考给程序设置系统事件
// 1.通航 --- 分船闸倒向 上下行切换 和正常的运营即实际有船通航 --- 需要连续并持续的信息判断
// 2.船闸状态  --- 只和当前的获取的船闸状态值有关系 与实际场景运营层面的通不通船无关
// 	船闸状态的判断 --- 从信号灯的角度来判断的 需要针对不同情况判断
// 3.船闸系统调度事件的产生  --- 只有符合通航和船闸状态判断条件的情况下 才触发系统事件
// 4.船闸系统调度事件 --- 是系统产生控制和行为的状态判断
// 5.闸次 --- 是系统产生用来标识通航通船的所有有效数据的标识
// 	包含实际有通船的完整的一个上下行进闸到出闸完成的过程
// **********************************************************************************

// 判断的思路分3层:
// 1. 船闸状态的推断  只关注可以断定获取的船闸状态 也即 船闸联动程序自动的状态判断
// 2. 通航状态的推断  关注人和运营层面的操作 结合船闸状态和信号灯 来断定通航的状态
// 3. 通航状态可能没有结束的信息  需要设置系统事件 结合系统事件来推断通航状态的结束

const ()

type PredicateScheduleCompositeData struct {
	NavState              NavStatus                   `json:"NavState,string"`         //当前综合推断的通航状态
	NavStateLast          NavStatus                   `json:"NavStateLast,string"`     //最近一次综合推断的通航状态
	NavlockState          NavlockStatus               `json:"NavlockState,string"`     //当前综合推断船闸状态
	NavlockStateLast      NavlockStatus               `json:"NavlockStateLast,string"` //最近一次综合推断船闸状态
	BeginTime             time.Time                   `json:"BeginTime"`               //通航状态的综合推断 开始的时间
	EndTime               time.Time                   `json:"EndTime"`                 //通航状态的综合推断 结束的时间
	SignalLightsBeginTime time.Time                   `json:"SignalLightsBeginTime"`   //进闸状态下 人工操作信号灯 开启的开始时间
	SignalLightsEndTime   time.Time                   `json:"SignalLightsEndTime"`     //进闸状态下 人工操作信号灯 开启的结束时间
	NavlockDynDatas       []*NavlockDynamicStatusData `json:"NavlockDynDatas"`         //船闸状态推断程序的缓存 只缓存不同的阶段的船闸状态 相同的状态会更新最后的推断时间
}

type PredicateCompositeData struct {
	Description       string                         `json:"Desc"`                  //当前综合推断的描述状态
	NavStatusInterval time.Duration                  `json:"NavStatusGoInInterval"` //通航状态的判断 需要的时间间隔 持续时间小于该时间的状态判定 通航换向
	PSCData           PredicateScheduleCompositeData `json:"NavScheduleData"`
}

func (pcd *PredicateCompositeData) JsonData() ([]byte, error) {
	return cm.Json.Marshal(pcd)
}

// 定义基础的状态和转换逻辑  模型当下只用于葛洲坝的定义
// Schedule Status :{NavStatus:"通航状态"，NavlStatus:"船闸状态"}
type PredicateScheduleWrapper struct {
	rwm          *sync.RWMutex `json:"-"`
	currentNavls NavlockStatus `json:"-"`
	currentNavs  NavStatus     `json:"-"`
	pcd          *PredicateCompositeData
}

// 能断定通航状态-上下行 需要持续的时间 意味着只有持续这么久的时间 才能认定这个状态是可以确定的
func NewPredicateScheduleWrapper(navStatusInInterval time.Duration, trs *TrackingSt) *PredicateScheduleWrapper {
	return &PredicateScheduleWrapper{
		rwm:          &sync.RWMutex{},
		currentNavls: NavlockUnknown,
		currentNavs:  NavUnknown,
		pcd: &PredicateCompositeData{
			PSCData:           PredicateScheduleCompositeData{},
			NavStatusInterval: navStatusInInterval,
		},
	}
}

// 获取当前综合判断json格式的数据
func (psw *PredicateScheduleWrapper) GetJsonData() ([]byte, error) {
	defer psw.rwm.RUnlock()
	psw.rwm.RLock()
	return psw.pcd.JsonData()
}

func (psw *PredicateScheduleWrapper) String() string {
	defer psw.rwm.RUnlock()
	psw.rwm.RLock()
	bs, _ := psw.pcd.JsonData()
	return string(bs)
}

// 将当前综合判断json格式的数据 存入数据库
func (psw *PredicateScheduleWrapper) SaveToDB(trs *TrackingSt, db *gorm.DB, l *zap.Logger) {
	defer psw.rwm.RUnlock()
	psw.rwm.RLock()

	var sche db_navlock.IotNavStatus
	sche.NavlockId = fmt.Sprintf("%d", trs.NavlockIdentifier)
	sche.ScheduleId = trs.ScheduleId.GetScheduleId()
	sche.NavStatus = psw.pcd.PSCData.NavState.String()
	sche.NavlockStatus = psw.pcd.PSCData.NavlockState.String()
	sche.CreatedAt = psw.pcd.PSCData.BeginTime
	sche.UpdatedAt = psw.pcd.PSCData.EndTime

	if bs, err := psw.pcd.JsonData(); err != nil {
		l.Sugar().Debugf("error: %+v", err)
	} else {
		sche.Info = bs
		sqldb, err := db.DB()
		if err != nil {
			l.Sugar().Errorf("err: %+v", err)
			return
		}

		if err = sqldb.Ping(); err != nil {
			l.Sugar().Errorf("err: %+v", err)
			return
		}

		dbd := db.Debug()
		tx := dbd.Begin()
		if err = dbd.Create(&sche).Error; err != nil {
			l.Sugar().Errorf(`insert to db error: %+v`, err)
			tx.Rollback()
			return
		}
		tx.Commit()
	}

}

// 将通航状态的切换存入数据库
func (psw *PredicateScheduleWrapper) saveNavStatusSwitchToDB(pcd *PredicateCompositeData, navs NavStatus, navls NavlockStatus,
	trs *TrackingSt, db *gorm.DB, db0 *gorm.DB, l *zap.Logger) {

	if db == nil && db0 == nil {
		return
	}

	var sche db_navlock.IotNavStatus
	sche.NavlockId = fmt.Sprintf("%d", trs.NavlockIdentifier)
	sche.ScheduleId = trs.ScheduleId.GetScheduleId()
	sche.NavStatus = navs.String()
	sche.NavlockStatus = navls.String()
	sche.CreatedAt = pcd.PSCData.BeginTime
	sche.UpdatedAt = pcd.PSCData.EndTime

	if bs, err := pcd.JsonData(); err != nil {
		l.Sugar().Debugf("error: %+v", err)
	} else {
		sche.Info = bs

		if db != nil {
			sqldb, err := db.DB()
			if err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			if err = sqldb.Ping(); err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			dbd := db.Debug()
			tx := dbd.Begin()
			sche.ID = 0
			if err = tx.Create(&sche).Error; err != nil {
				l.Sugar().Errorf(`insert to db error: %+v`, err)
				tx.Rollback()
				return
			}
			tx.Commit()
		}

		if db0 != nil {
			sqldb, err := db0.DB()
			if err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			if err = sqldb.Ping(); err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			dbd := db0.Debug()
			tx := dbd.Begin()
			sche.ID = 0
			if err = tx.Create(&sche).Error; err != nil {
				l.Sugar().Errorf(`insert to db error: %+v`, err)
				tx.Rollback()
				return
			}
			tx.Commit()
		}

	}
}

// 将船闸状态的切换存入数据
func (psw *PredicateScheduleWrapper) saveNavlockStatusSwitchToDB(navls *NavlockDynamicStatusData, navs NavStatus,
	trs *TrackingSt, db *gorm.DB, db0 *gorm.DB, l *zap.Logger) {
	if db == nil && db0 == nil {
		return
	}

	var sche db_navlock.IotNavlStatus
	sche.NavlockId = fmt.Sprintf("%d", trs.NavlockIdentifier)
	sche.ScheduleId = trs.ScheduleId.GetScheduleId()
	sche.NavStatus = navs.String()
	sche.NavlockStatus = navls.NavlockState.String()
	sche.CreatedAt = navls.BeginTime
	sche.UpdatedAt = navls.EndTime

	if bs, err := navls.JsonData(); err != nil {
		l.Sugar().Debugf("error: %+v", err)
	} else {
		sche.Info = bs

		if db != nil {
			sqldb, err := db.DB()
			if err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			if err = sqldb.Ping(); err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			dbd := db.Debug()
			tx := dbd.Begin()
			sche.ID = 0
			if err = tx.Create(&sche).Error; err != nil {
				l.Sugar().Errorf(`insert to db error: %+v`, err)
				tx.Rollback()
				return
			}
			tx.Commit()
		}

		if db0 != nil {
			sqldb, err := db0.DB()
			if err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			if err = sqldb.Ping(); err != nil {
				l.Sugar().Errorf("err: %+v", err)
				return
			}

			dbd0 := db0.Debug()
			tx := dbd0.Begin()
			sche.ID = 0
			if err = tx.Create(&sche).Error; err != nil {
				l.Sugar().Errorf(`insert to db error: %+v`, err)
				tx.Rollback()
				return
			}
			tx.Commit()
		}
	}

}

// 清空整个数据结构 重新产生记录
func (psw *PredicateScheduleWrapper) ReSet() {
	defer psw.rwm.Unlock()
	psw.rwm.Lock()
	psw.pcd = &PredicateCompositeData{
		PSCData:           PredicateScheduleCompositeData{},
		NavStatusInterval: psw.pcd.NavStatusInterval,
	}
}

// 更新联动自动程序的船闸状态 程序针对船闸状态做过滤 只缓存不同的阶段的船闸状态 相同的状态只会更新最后的推断时间
// todo: 针对人运营的状态做判断 存在逻辑上的漏洞 必然会再拉回来
func (psw *PredicateScheduleWrapper) PushNavlockAutoStatus(navlockDynState *NavlockDynamicStatusData,
	trs *TrackingSt, db *gorm.DB, db0 *gorm.DB, l *zap.Logger) (isNavUpDownSwitch, isNavlSwitch bool) {
	defer psw.rwm.Unlock()
	psw.rwm.Lock()

	// 船闸状态是否切换
	ifNavlocksSwitched := false
	// 通航状态是否切换
	ifNavsSwitched := false
	// 通航状态是否切换到 通航上下行
	ifNavsUpDownSwithed := false

	//1. 更新整个判断的结束时间
	psw.pcd.PSCData.EndTime = navlockDynState.EndTime

	//2. 初始化 在没有记录或者是清空的情况下 -只有在进闸完成的情况下清空
	if psw.pcd.PSCData.NavlockDynDatas == nil || len(psw.pcd.PSCData.NavlockDynDatas) == 0 {
		// 1. 添加到缓存
		psw.pcd.PSCData.NavlockDynDatas = append(psw.pcd.PSCData.NavlockDynDatas, navlockDynState)
		// 2. 初始化 开始时间
		psw.pcd.PSCData.BeginTime = navlockDynState.BeginTime
		// 3. 初始化 船闸状态
		psw.pcd.PSCData.NavlockState = navlockDynState.NavlockState
		psw.pcd.PSCData.NavlockStateLast = navlockDynState.NavlockState
		// 4. 判断 通航状态
		psw.predicateNavStatusOnSignalLights(navlockDynState)
		// 5. 更新添加记录的通航状态
		navlockDynState.NavState = psw.pcd.PSCData.NavState
		navlockDynState.NavStateLast = psw.pcd.PSCData.NavStateLast
		navlockDynState.Description = "程序初始化"

		psw.currentNavls = psw.pcd.PSCData.NavlockState
		psw.currentNavs = psw.pcd.PSCData.NavState
		psw.saveNavlockStatusSwitchToDB(navlockDynState, navlockDynState.NavState, trs, db, db0, l)
		return
	}

	//3. 非初始化情况下的持续添加
	lastData := psw.pcd.PSCData.NavlockDynDatas[len(psw.pcd.PSCData.NavlockDynDatas)-1]
	//3.1 判断最近一次的船闸状态是否一致 并且上下游的信号灯是否一致
	if (lastData.NavlockState == navlockDynState.NavlockState) &&
		((lastData.TUpSignalLights == navlockDynState.TUpSignalLights) &&
			(lastData.TDownSignalLights == navlockDynState.TDownSignalLights)) {
		// 如果一致 则只更新船闸状态缓存的时间
		lastData.EndTime = navlockDynState.EndTime
	} else {
		//3.2 不一致的情况下需要添加到缓存
		//3.2.1 添加到缓存 设置时间和状态
		psw.pcd.PSCData.NavlockDynDatas = append(psw.pcd.PSCData.NavlockDynDatas, navlockDynState)
		//3.2.2 如果船闸状态不同 更新船闸状态
		if lastData.NavlockState != navlockDynState.NavlockState {
			psw.pcd.PSCData.NavlockStateLast = psw.pcd.PSCData.NavlockState
			psw.pcd.PSCData.NavlockState = navlockDynState.NavlockState
			navlockDynState.NavlockStateLast = psw.pcd.PSCData.NavlockStateLast
			navlockDynState.Description = fmt.Sprintf("船闸状态转换:(%s)-(%s),前状态持续时间:%.2f分钟",
				psw.pcd.PSCData.NavlockStateLast,
				psw.pcd.PSCData.NavlockState,
				lastData.EndTime.Sub(lastData.BeginTime).Minutes())
			//船闸状态是否更新
			ifNavlocksSwitched = true
			//设置船闸状态快照
			psw.currentNavls = psw.pcd.PSCData.NavlockState
		}
	}

	//3.3 判断通航状态是否切换 和 是否是通航上下行
	ifNavsSwitched, ifNavsUpDownSwithed = psw.predicateNavStatusOnSignalLights(navlockDynState)
	//3.4 更新添加记录的通航状态
	navlockDynState.NavState = psw.pcd.PSCData.NavState
	navlockDynState.NavStateLast = psw.pcd.PSCData.NavStateLast

	if ifNavsSwitched {
		navInterval := psw.pcd.PSCData.EndTime.Sub(psw.pcd.PSCData.BeginTime).Minutes()
		psw.pcd.Description = fmt.Sprintf("通航状态:(%s),持续时间:%.2f分钟", psw.pcd.PSCData.NavStateLast, navInterval)
		psw.saveNavStatusSwitchToDB(psw.pcd, psw.pcd.PSCData.NavStateLast, psw.pcd.PSCData.NavlockState, trs, db, db0, l)
		//设置通航状态快照
		psw.currentNavs = psw.pcd.PSCData.NavState
		//~只在通航上下游切换的情况下 才更新并记录 都更新记录
		//if ifNavsUpDownSwithed {
		//只要通航状态改变 都产生闸次 便于追踪查找
		trs.ScheduleId.GenerationGet()

		//设置通航状态快照
		psw.currentNavs = psw.pcd.PSCData.NavState

		tmp := &PredicateCompositeData{
			PSCData: PredicateScheduleCompositeData{
				NavState:              psw.pcd.PSCData.NavState,
				NavStateLast:          psw.pcd.PSCData.NavStateLast,
				NavlockState:          psw.pcd.PSCData.NavlockState,
				NavlockStateLast:      psw.pcd.PSCData.NavlockStateLast,
				SignalLightsBeginTime: psw.pcd.PSCData.SignalLightsBeginTime,
				SignalLightsEndTime:   psw.pcd.PSCData.SignalLightsEndTime,
				BeginTime:             navlockDynState.BeginTime,
				EndTime:               navlockDynState.EndTime,
				NavlockDynDatas:       append([]*NavlockDynamicStatusData(nil), navlockDynState),
			},
			NavStatusInterval: psw.pcd.NavStatusInterval,
			Description: fmt.Sprintf("通航状态转换:[%s]-[%s],前状态持续时间:%.2f分钟,进闸信号灯(超设定阈值或持续)时间:%.2f秒",
				navlockDynState.NavStateLast, navlockDynState.NavState, navInterval,
				psw.pcd.PSCData.SignalLightsEndTime.Sub(psw.pcd.PSCData.SignalLightsBeginTime).Seconds()),
		}
		psw.saveNavStatusSwitchToDB(tmp, psw.pcd.PSCData.NavState, psw.pcd.PSCData.NavlockState, trs, db, db0, l)
		psw.pcd = tmp
		//}
	}

	//发生船闸状态更新的情况下 写入更新前和更新后的船闸数据
	if ifNavlocksSwitched {
		lastData.Description = fmt.Sprintf("船闸状态:(%s),持续时间;%.2f分钟", lastData.NavlockState, lastData.EndTime.Sub(lastData.BeginTime).Minutes())
		psw.saveNavlockStatusSwitchToDB(lastData, psw.pcd.PSCData.NavState, trs, db, db0, l)
		psw.saveNavlockStatusSwitchToDB(navlockDynState, psw.pcd.PSCData.NavState, trs, db, db0, l)
	}

	// 船闸状态为进闸完成时则重置信号灯判断的时间
	switch navlockDynState.NavlockState {
	case NavlockDownGoInDone, NavlockUpGoInDone:
		{
			psw.pcd.PSCData.SignalLightsBeginTime = time.Time{}
			psw.pcd.PSCData.SignalLightsEndTime = time.Time{}
		}
	}

	isNavUpDownSwitch = ifNavsUpDownSwithed
	isNavlSwitch = ifNavlocksSwitched
	return
}

// 获取当前的船闸状态和通航状态
func (psw *PredicateScheduleWrapper) GetScheduleStatus() (navlocks NavlockStatus, navs NavStatus) {
	defer psw.rwm.RUnlock()
	psw.rwm.RLock()

	//todo: 船闸状态-- 出闸完成 还需要判断事件产生的时间
	navlocks = psw.currentNavls
	navs = psw.currentNavs
	return
}

// 根据信号灯判断通航状态
func (psw *PredicateScheduleWrapper) predicateNavStatusOnSignalLights(navlockDynState *NavlockDynamicStatusData) (ifNavsSwitch bool, ifNavsUpDownSwithch bool) {
	// 通航状态是否切换
	ifNavsSwitch = false
	// 通航状态是否上下行切换
	ifNavsUpDownSwithch = false

	//1. 只有进闸状态 才需要判断是否存在人工操作的绿灯
	switch navlockDynState.NavlockState {
	case NavlockDownGoingIn, NavlockUpGoingIn:
		{
			// 上游右信号灯或者是下游左信号灯 是人工开启的
			if navlockDynState.TUpSignalLights.Right == GreenLightOn ||
				navlockDynState.TDownSignalLights.Left == GreenLightOn {
				//此时一定会发生通航状态的切换   所以此时设置 上次的通航状态的值 便于后续判断
				psw.pcd.PSCData.NavStateLast = psw.pcd.PSCData.NavState
				//1.1 信号灯开启的 开始时间为空值 说明是第一次获取到信号灯的值 赋值并返回
				if (psw.pcd.PSCData.SignalLightsBeginTime == time.Time{}) ||
					psw.pcd.PSCData.SignalLightsBeginTime.IsZero() {
					//1.1.1 设置时间
					psw.pcd.PSCData.SignalLightsBeginTime = navlockDynState.EndTime
					psw.pcd.PSCData.SignalLightsEndTime = navlockDynState.EndTime
					return
				}
				//1.2 信号灯开启的 开始时间非空的情况下 设置信号灯开启的结束时间 进入下一轮判断
				psw.pcd.PSCData.SignalLightsEndTime = navlockDynState.EndTime
			}
		}
	}

	// // TOD: 会出现各个闸次混淆不清
	// //~~~~~~小于设置的时间段 也做判断 由通航换向转换 --> 通航上下行  向下兼容
	// //2. 信号灯开启的开始时间和结束时间 有值 并且 综合判断的时间 与 信号灯开启的开始时间 间隔时间小于等于设置的判断的时间间隔的情况下 不做判断
	// if time.Duration(psw.pcd.PSCData.EndTime.Sub(psw.pcd.PSCData.SignalLightsBeginTime).Seconds()) <= psw.pcd.NavStatusInterval {
	// 	return
	// }

	// 当前通航状态 针对通航切换 避免在入闸时间段一直切换
	switch navlockDynState.NavState {
	case NavDown:
		{
			return
		}
	case NavUp:
		{
			return
		}
	}

	// 分别针对上下行进闸的情况处理
	//3. 信号灯开启的开始时间和结束时间 有值 并且 间隔时间大于 设置的判断时间间隔 则做出判断
	switch navlockDynState.NavlockState {
	case NavlockDownGoingIn:
		{
			//3.1 下行进闸的情况下 上右信号灯点亮
			if navlockDynState.TUpSignalLights.Right == GreenLightOn {
				// 3.1.1 通航下行
				if psw.predicateManOperatedInterval() {
					if psw.pcd.PSCData.NavStateLast != NavDown {
						ifNavsSwitch = true
						ifNavsUpDownSwithch = true
					}
					psw.pcd.PSCData.NavState = NavDown
				} else {
					// 3.1.2 通航换向-下行换上行
					if psw.pcd.PSCData.NavStateLast != NavSwitchDownToUp {
						ifNavsSwitch = true
					}
					psw.pcd.PSCData.NavState = NavSwitchDownToUp
				}
			}
		}
	case NavlockUpGoingIn:
		{
			//3.2 上行进闸的情况下 下左信号灯点亮
			if navlockDynState.TDownSignalLights.Left == GreenLightOn {
				//3.2.1 通航上行
				if psw.predicateManOperatedInterval() {
					if psw.pcd.PSCData.NavStateLast != NavUp {
						ifNavsSwitch = true
						ifNavsUpDownSwithch = true
					}
					psw.pcd.PSCData.NavState = NavUp
				} else {
					//3.2.2 通航换向-上行换下行
					if psw.pcd.PSCData.NavStateLast != NavSwitchUpToDown {
						ifNavsSwitch = true
					}
					psw.pcd.PSCData.NavState = NavSwitchUpToDown
				}
			}
		}
	}
	return
}

// 判断人工操作的时间间隔 大于设置的时间间隔-true
func (psw *PredicateScheduleWrapper) predicateManOperatedInterval() bool {
	return (time.Duration(psw.pcd.PSCData.SignalLightsEndTime.Sub(psw.pcd.PSCData.SignalLightsBeginTime).Seconds()) > psw.pcd.NavStatusInterval)
}

// 获取当前的船闸状态
func (psw *PredicateScheduleWrapper) GetNavlockStatus() (navlocks NavlockStatus) {
	psw.rwm.RLock()
	defer psw.rwm.RUnlock()
	navlocks = psw.currentNavls
	return
}

// 设置当前的船闸状态
func (psw *PredicateScheduleWrapper) SetNavlockStatus(navlocks NavlockStatus) {
	psw.rwm.Lock()
	defer psw.rwm.Unlock()
	psw.currentNavls = navlocks
}

// 获取当前的船闸状态 判断是否是 船舶测速系统有效的 船闸状态
func (psw *PredicateScheduleWrapper) GetIsValidShipSpeedNavlState() (navls NavlockStatus, valid bool) {
	psw.rwm.RLock()
	defer psw.rwm.RUnlock()
	navls = psw.currentNavls
	switch navls {
	case NavlockUpGoingIn, NavlockUpGoingOut:
		valid = true
	case NavlockDownGoingIn, NavlockDownGoingOut:
		valid = true
	default:
		valid = false
	}
	return
}

// 获取当前的船闸状态 判断是否是 禁停线系统有效的 船闸状态
func (psw *PredicateScheduleWrapper) GetIsValidStopLineNavlState() (navls NavlockStatus, valid bool) {
	psw.rwm.RLock()
	defer psw.rwm.RUnlock()
	navls = psw.currentNavls
	switch navls {
	case NavlockUpGoingIn, NavlockUpGoInDone, NavlockUpWaterTrendUp, NavlockUpGoingOut:
		valid = true
	case NavlockDownGoingIn, NavlockDownGoInDone, NavlockDownWaterTrendDown, NavlockDownGoingOut:
		valid = true
	default:
		valid = false
	}
	return
}

// 获取当前的通航状态
func (psw *PredicateScheduleWrapper) GetNavStatus() (navs NavStatus) {
	psw.rwm.RLock()
	defer psw.rwm.RUnlock()
	navs = psw.currentNavs
	return
}

// 设置当前的通航状态
func (psw *PredicateScheduleWrapper) SetNavStatus(navs NavStatus) {
	psw.rwm.Lock()
	defer psw.rwm.Unlock()
	psw.currentNavs = navs
}

// 获取当前的通航状态 是否是船舶测速 或者是禁停线 等系统 通航运营有效的通航状态
func (psw *PredicateScheduleWrapper) GetIsValidNavState() (navs NavStatus, valid bool) {
	psw.rwm.RLock()
	defer psw.rwm.RUnlock()
	navs = psw.currentNavs
	switch navs {
	case NavUp, NavDown:
		valid = true
	default:
		valid = false
	}
	return
}
