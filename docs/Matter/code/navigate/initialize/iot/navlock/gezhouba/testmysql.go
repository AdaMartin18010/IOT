package gezhouba

import (
	"fmt"
	"time"

	mdl "navigate/common/model"
	g "navigate/global"
	database "navigate/store/iot/navlock/model"
)

var (
	L = mdl.L
)

func Mysql8Testcase() {
	defer L.Sugar().Debug("Mysql8Testcase: End.Checkout sqlite database")
	defer func() {
		if err := recover(); err != nil {
			L.Sugar().Errorf("Mysql8Testcase: catch panic: %+v", err)
		}
	}()

	L.Sugar().Infof("Mysql8Testcase: Begin")

	db := g.ProduceDB
	ldb := g.LocalDB

	// sysNLR := database.SysNavigationlockRuntimeInfo{}
	// sys := database.SysShipsSpeedInfo{}
	sqldb, err := db.DB()
	if err != nil {
		L.Sugar().Errorf("err: %+v", err)
	}
	if err := sqldb.Ping(); err != nil {
		L.Sugar().Errorf("err: %+v", err)
	}
	ldb.Create(&database.IotNavlCpntState{ServerName: "Mysql8Testcase", Info: fmt.Sprintf(`{"db.Stats": "%+v"}`, sqldb.Stats())})
	// Create
	begin := time.Now()
	count := 5000
	tx := db.Begin()
	for i := 0; i < count; i++ {
		sysRTH := database.IotNavlCpntState{}
		sysRTH.ScheduleId = fmt.Sprintf("%d", i)
		sysRTH.Info = fmt.Sprintf(`{"i":%d}`, i)
		err := tx.Create(&sysRTH).Error
		if err != nil {
			L.Sugar().Errorf("err: %+v", err)
		}
	}
	tx.Commit()

	timeInterval := time.Since(begin).Milliseconds()
	qps := float64(0.0)
	if timeInterval > 0 {
		qps = float64(count) / float64(timeInterval/1000)
	}
	ldb.Create(&database.IotNavlCpntState{ServerName: "Mysql8Testcase-insert", Info: fmt.Sprintf(`{"db.Stats": "%+v","qps-tx":"%f"}`, sqldb.Stats(), qps)})

	// Select
	begin = time.Now()
	for i := 0; i < count; i++ {
		sysRTH := database.IotNavlCpntState{}
		err := db.First(&sysRTH, "schedule_id = ?", i).Error
		if err != nil {
			L.Sugar().Errorf("err: %+v", err)
		}
	}

	timeInterval = time.Since(begin).Milliseconds()
	qps = 0
	if timeInterval > 0 {
		qps = float64(count) / float64(timeInterval/1000)
	}
	ldb.Create(&database.IotNavlCpntState{ServerName: "Mysql8Testcase-select", Info: fmt.Sprintf(`{"db.Stats": "%+v","qps":"%f"}`, sqldb.Stats(), qps)})

	// Update
	begin = time.Now()
	tx = db.Begin()
	for i := 0; i < count; i++ {
		//sysRTH.ScheduleId = fmt.Sprintf("%d", i)
		err := tx.Model(&database.IotNavlCpntState{}).Where("schedule_id = ?", i).Update("info", fmt.Sprintf(`{"i":%d}`, i)).Error
		if err != nil {
			L.Sugar().Errorf("err: %+v", err)
		}
	}
	tx.Commit()

	timeInterval = time.Since(begin).Milliseconds()
	qps = 0
	if timeInterval > 0 {
		qps = float64(count) / float64(timeInterval/1000)
	}
	ldb.Create(&database.IotNavlCpntState{ServerName: "Mysql8Testcase-Update", Info: fmt.Sprintf(`{"db.Stats": "%+v","qps-tx":"%f"}`, sqldb.Stats(), qps)})

	// Delete - delete product
	begin = time.Now()
	tx = db.Begin()
	for i := 0; i < count; i++ {
		sysRTH := database.IotNavlCpntState{}
		sysRTH.ScheduleId = fmt.Sprintf("%d", i)
		err := tx.Unscoped().Where("schedule_id = ?", sysRTH.ScheduleId).Delete(&sysRTH).Error
		if err != nil {
			L.Sugar().Errorf("err: %+v", err)
		}
	}
	tx.Commit()
	timeInterval = time.Since(begin).Milliseconds()
	qps = 0
	if timeInterval > 0 {
		qps = float64(count) / float64(timeInterval/1000)
	}
	ldb.Create(&database.IotNavlCpntState{ServerName: "Mysql8Testcase-Delete", Info: fmt.Sprintf(`{"db.Stats": "%+v","qps-tx":"%f"}`, sqldb.Stats(), qps)})
	g.MainCtr.Cancel()
}
