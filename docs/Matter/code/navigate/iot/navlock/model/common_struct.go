package model

import (
	"context"
	"errors"
	cm "navigate/common/model"
	cmpt "navigate/common/model/component"
	evchs "navigate/common/model/eventchans"
	g "navigate/global"
	"sync"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

var (
	//ModelLoger = cm.L.With(zap.Namespace("tag"), zap.String("model", "navlock"))
	ModelLoger = cm.L.With(zap.Namespace("tag"))

	_ cm.WorkerRecover = (*CptSt)(nil)
	//_ cmpt.Component          = (*ComponentStruct)(nil)
	_ cmpt.CptsOperator = (*CptSt)(nil)
	//_ cmpt.ComponentComposite = (*ComponentStruct)(nil)
	//_ cmpt.ComponentRoot      = (*ComponentStruct)(nil)
)

const (
	eventchansLengthMin = 10000
)

var ()

func init() {
}

// model 组件的公用结构
type CptSt struct {
	*cmpt.CptMetaSt
	cmpt.Cpts
	evchs.EventChans
	cmpt.Cmder

	rwm *sync.RWMutex
}

// 暂时只初始化基本的使用 后续再更新
func NewCptSt(kn cmpt.KindName, in cmpt.IdName, evclen uint, cmstct *cm.CtrlSt) *CptSt {
	var evcs evchs.EventChans
	if evclen == 0 {
		evcs = nil
	} else if evclen < eventchansLengthMin {
		evclen = eventchansLengthMin
		evcs = evchs.NewEvtChans(evclen)
	}

	return &CptSt{
		CptMetaSt:  cmpt.NewCptMetaSt(kn, in, cmstct),
		Cpts:       cmpt.NewCpts(),
		EventChans: evcs,
		Cmder:      cmpt.NewCmder(),
		rwm:        &sync.RWMutex{},
	}
}

func (mcs *CptSt) Validate() error {
	defer mcs.rwm.RUnlock()
	mcs.rwm.RLock()

	if mcs.CptMetaSt == nil {
		return errors.New("ModelComponentStruct.CptBaseData is nil")
	}

	if mcs.CptMetaSt.Ctrl() == nil {
		return errors.New("ModelComponentStruct.CptBaseData.CtlData is nil")
	}

	return nil
}

func (mcs *CptSt) Fork(kn cmpt.KindName, in cmpt.IdName) *CptSt {
	mcs.rwm.RLock()
	defer mcs.rwm.RUnlock()
	// EventChans 是公用的
	return &CptSt{
		CptMetaSt:  cmpt.NewCpt(kn, in, mcs.CptMetaSt.Ctrl().ForkCtxWg()),
		Cpts:       cmpt.NewCpts(),
		EventChans: mcs.EventChans,
		Cmder:      cmpt.NewCmder(),
		rwm:        &sync.RWMutex{},
	}
}

// func (mcs *CptSt) ForkCtxTimeout(kn cmpt.KindName, in cmpt.IdName, tm time.Duration) *CptSt {
// 	mcs.rwm.RLock()
// 	defer mcs.rwm.RUnlock()

// 	return &CptSt{
// 		CptMetaSt:  cmpt.NewCpt(kn, in, mcs.CptMetaSt.Ctrl().ForkCtxTimeout(tm)),
// 		Cpts:       cmpt.NewCpts(),
// 		EventChans: mcs.EventChans,
// 		Cmder:      cmpt.NewCmder(),
// 		rwm:        &sync.RWMutex{},
// 	}
// }

// func (mcs *CptSt) WithCtrl(ctrl *cm.CtrlSt) *CptSt {
// 	mcs.rwm.Lock()
// 	defer mcs.rwm.Unlock()
// 	mcs.CptMetaSt.Ctrl().WithCtrl(ctrl)
// 	return mcs
// }

func (mcs *CptSt) WithCtx(ctx context.Context) *CptSt {
	mcs.rwm.Lock()
	defer mcs.rwm.Unlock()

	mcs.CptMetaSt.Ctrl().WithCtx(ctx)
	return mcs
}

// func (mcs *CptSt) WithTimeout(ctx context.Context, tm time.Duration) *CptSt {
// 	mcs.rwm.Lock()
// 	defer mcs.rwm.Unlock()

// 	mcs.CptMetaSt.Ctrl().WithTimeout(ctx, tm)
// 	return mcs
// }

// model 系统运行时的公用结构
type ModelSysSt struct {
	Logger     *zap.Logger // 全局日志分支
	LDB        *gorm.DB    // 本地数据库 local database
	PDB        *gorm.DB    // 生产数据库 Production database
	PwebDB     *gorm.DB    // ProduceWebDB web数据库访问
	SystemName string      // 系统名称
	ServerName string      // 服务名称
	DriverName string      // 组件实例或者驱动名称
}

// 使用默认的日志格式初始化
func NewModelSysSt(logNameSpace, sysnm, cptnm, svrnm, dvrnm string) *ModelSysSt {
	temp := &ModelSysSt{
		LDB:        g.LocalDB,
		PDB:        g.ProduceDB,
		PwebDB:     g.ProduceWebDB,
		ServerName: svrnm,
		SystemName: sysnm,
		DriverName: dvrnm,
	}
	logger := cm.L
	if !(logNameSpace == "") {
		logger = logger.With(zap.Namespace(logNameSpace))
	}

	if !(sysnm == "") {
		logger = logger.With(zap.String("sys", sysnm))
	}

	if !(cptnm == "") {
		logger = logger.With(zap.String("cpt", cptnm))
	}

	if !(svrnm == "") {
		logger = logger.With(zap.String("svr", svrnm))
	}

	if !(dvrnm == "") {
		logger = logger.With(zap.String("dvr", dvrnm))
	}

	temp.Logger = logger
	return temp
}

// func (m *ModelSysSt) Validate() error {
// 	if m.Logger == nil {
// 		return errors.New("ModelSysStruct.Logger is nil")
// 	}

// 	if m.LDB == nil {
// 		return errors.New("ModelSysStruct.LDB is nil")
// 	}

// 	return nil
// }

// func (m *ModelSysSt) Clone() *ModelSysSt {
// 	return &ModelSysSt{
// 		Logger:     m.Logger,
// 		LDB:        m.LDB,
// 		PDB:        m.PDB,
// 		PwebDB:     m.PwebDB,
// 		SystemName: m.SystemName,
// 		ServerName: m.ServerName,
// 		DriverName: m.DriverName,
// 	}
// }

// // 重新生成log
// func (m *ModelSysSt) ForkLogAll(sern, sysn, drn string) *ModelSysSt {
// 	return &ModelSysSt{
// 		Logger: ModelLoger.With(zap.String("svr", sern),
// 			zap.String("sys", sysn),
// 			zap.String("dvr", drn)),
// 		LDB:        m.LDB,
// 		PDB:        m.PDB,
// 		PwebDB:     m.PwebDB,
// 		SystemName: sysn,
// 		ServerName: sern,
// 		DriverName: drn,
// 	}
// }

// func (m *ModelSysSt) ForkLogSys(l *zap.Logger, sysn, drn string) *ModelSysSt {
// 	return &ModelSysSt{
// 		Logger: l.With(zap.String("sys", sysn),
// 			zap.String("dvr", drn)),
// 		LDB:        m.LDB,
// 		PDB:        m.PDB,
// 		PwebDB:     m.PwebDB,
// 		SystemName: sysn,
// 		ServerName: m.ServerName,
// 		DriverName: drn,
// 	}
// }

// func (m *ModelSysSt) ForkLogDr(l *zap.Logger, sysn, drn string) *ModelSysSt {
// 	return &ModelSysSt{
// 		Logger:     l.With(zap.String("dvr", drn)),
// 		LDB:        m.LDB,
// 		PDB:        m.PDB,
// 		PwebDB:     m.PwebDB,
// 		SystemName: sysn,
// 		ServerName: m.ServerName,
// 		DriverName: drn,
// 	}
// }
