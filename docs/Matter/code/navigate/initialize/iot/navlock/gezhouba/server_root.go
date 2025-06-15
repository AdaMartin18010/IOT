package gezhouba

import (
	"fmt"
	"os"
	"time"

	cm "navigate/common"
	mdl "navigate/common/model"
	cmpt "navigate/common/model/component"
	g "navigate/global"
	navlmd "navigate/iot/navlock/model"
)

var (
	_ cmpt.CptRoot = (*ServiceMasterAdapter)(nil)

	GServerMaster *ServiceMasterAdapter = nil
)

func InitServerMaster(ctr *mdl.CtrlSt) {
	GServerMaster = NewServiceMasterAdapter(ctr)
}

type ServiceMasterAdapter struct {
	*navlmd.ServerRoot
}

func NewServiceMasterAdapter(ctr *mdl.CtrlSt) *ServiceMasterAdapter {
	tmp := &ServiceMasterAdapter{
		ServerRoot: navlmd.NewServerRoot(ctr),
	}
	tmp.WorkerRecover = tmp
	return tmp
}

func (sra *ServiceMasterAdapter) Work() error {
	return nil
}

func (sra *ServiceMasterAdapter) Start() error {
	if sra.State.Load().(bool) {
		return fmt.Errorf("%s is already Started", sra.CmptInfo())
	}

	L.Sugar().Infof("--------- Init configuration ---------")
	InitViper()

	//读取配置文件
	err := SetUpConfig(GVerbose)
	if err != nil {
		L.Sugar().Infof("Read Configure file err : %+v,Exited !", err)
		os.Exit(0)
	}

	L.Sugar().Infof("%s Creating base components -------------------", sra.CmptInfo())
	//初始化基础组件
	InitDatabase()
	L.Sugar().Infof("%s Created base components -------------------", sra.CmptInfo())

	L.Sugar().Infof("%s Creating navlock components -------------------", sra.CmptInfo())
	//创建服务
	CreateAndAddComponents()
	L.Sugar().Infof("%s Created navlock components -------------------", sra.CmptInfo())

	L.Sugar().Infof("%s is Starting-------------------", sra.CmptInfo())
	sra.Ctrl().WaitGroup().StartingWait(sra.ServerRoot.WorkerRecover)
	err = sra.Cpts.Start()
	if err != nil {
		L.Sugar().Errorf("%s is Starting,err: %#v", sra.CmptInfo(), err)
		return err
	}
	L.Sugar().Infof("%s is Started.----------------------", sra.CmptInfo())
	L.Sugar().Infof(`Os_Information : --- time:"%s",%+v`, time.Now(), cm.GOsInfo)
	g.ExecutingCurrentFilePath = cm.ExecutingCurrentFilePath()
	L.Sugar().Infof("ExecutingCurrentFilePath: %+v", cm.ExecutingCurrentFilePath())
	g.CompiledExectionFilePath = cm.CompiledExectionFilePath()
	L.Sugar().Infof("CompiledExectionFilePath: %+v", g.CompiledExectionFilePath)

	g.ExecutedCurrentFilePath, err = cm.ExecutedCurrentFilePath()
	if err != nil {
		L.Sugar().Fatalf("ExecutedCurrentFilePath: err: %+v\n", err)
	}
	L.Sugar().Infof("ExecutedCurrentFilePath: %+v\n", g.ExecutedCurrentFilePath)
	L.Sugar().Infof(`--- You can cancel Or Waiting for seconds. Server checking then auto start....`)
	timer := time.NewTimer(1 * time.Second)
	select {
	case <-sra.Ctrl().Context().Done():
		mdl.L.Sugar().Info(`User cancel Exit.`)
		L.Sync()
		CloseDBs()
		os.Exit(0)
	case <-timer.C:
		break
	}
	// 保证所有的组件 都是fanout接受 主程序的控制
	sra.Ctrl().WaitGroup().StartAsync()
	sra.State.Store(true)
	return nil
}

func (sra *ServiceMasterAdapter) Finalize() error {
	sra.ServerRoot.Finalize()
	// 最后退出日志服务
	defer L.Sync()
	<-sra.Ctrl().Context().Done()
	//结束服务依赖的基础组件运行
	ServerFinalization()
	// 等待所有组件退出
	//sra.Ctrl().WaitGroup().WaitAsync()
	return nil
}
