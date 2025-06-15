package model

import (
	"context"
	"fmt"
	"time"

	//"github.com/glebarez/sqlite"
	mdl "navigate/common/model"
	g "navigate/global"

	"gorm.io/datatypes"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type Product struct {
	gorm.Model
	Code  string
	Price uint
}

func TestSqlite3(ctx context.Context) {
	dbcon, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
	if err != nil {
		panic("failed to connect database")
	}

	// Migrate the schema
	db := dbcon.WithContext(ctx)
	if err := db.AutoMigrate(&Product{}); err != nil {
		mdl.L.Sugar().Debugf("err: %+v", err)
	}

	sqldb, err := db.DB()
	if err != nil {
		mdl.L.Sugar().Debugf("err: %+v", err)
	}
	sqldb.Ping()
	mdl.L.Sugar().Debugf("info: %+v", sqldb.Stats())
	// Create
	dbs := db.Create(&Product{Code: "D42", Price: 100})
	if db.Error != nil {
		mdl.L.Sugar().Debugf("err: %+v", dbs.Error)
	}

	// Read
	var product Product
	db.First(&product, 1)                 // find product with integer primary key
	db.First(&product, "code = ?", "D42") // find product with code D42

	// Update - update product's price to 200
	db.Model(&product).Update("Price", 200)
	// Update - update multiple fields
	db.Model(&product).Updates(Product{Price: 200, Code: "F42"}) // non-zero fields
	db.Model(&product).Updates(map[string]interface{}{"Price": 100, "Code": "D42"})

	// Delete - delete product
	//db.Delete(&product, 1)

}

func TestSqlite3Json(ctx context.Context, db *gorm.DB) {

	type UserWithJSON struct {
		gorm.Model
		Name       string
		Attributes datatypes.JSON
	}

	// dbcon, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
	// if err != nil {
	// 	panic("failed to connect database")
	// }

	// Migrate the schema
	DB := db.WithContext(ctx)
	if DB.Migrator().HasTable(&UserWithJSON{}) {
		DB.Migrator().DropTable(&UserWithJSON{})
	}

	if err := DB.AutoMigrate(&UserWithJSON{}); err != nil {
		mdl.L.Sugar().Debugf("err: %+v", err)
	}

	// Go's json marshaler removes whitespace & orders keys alphabetically
	// use to compare against marshaled []byte of datatypes.JSON
	user1Attrs := `{"age":18,"name":"json-1","orgs":{"orga":"orga"},"tags":["tag1","tag2"],"admin":true}`

	users := []UserWithJSON{{
		Name:       "json-1",
		Attributes: datatypes.JSON([]byte(user1Attrs)),
	}, {
		Name:       "json-2",
		Attributes: datatypes.JSON([]byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)),
	}, {
		Name:       "json-3",
		Attributes: datatypes.JSON([]byte(`["tag1","tag2","tag3"]`)),
	}}

	if err := DB.Create(&users).Error; err != nil {
		mdl.L.Sugar().Debugf("Failed to create users %v", err)
	}

	// Check JSON has keys
	DB.Find(&users, datatypes.JSONQuery("attributes").HasKey("role"))
	DB.Find(&users, datatypes.JSONQuery("attributes").HasKey("orgs", "orga"))
	// MySQL
	// SELECT * FROM `users` WHERE JSON_EXTRACT(`attributes`, '$.role') IS NOT NULL
	// SELECT * FROM `users` WHERE JSON_EXTRACT(`attributes`, '$.orgs.orga') IS NOT NULL

	// PostgreSQL
	// SELECT * FROM "user" WHERE "attributes"::jsonb ? 'role'
	// SELECT * FROM "user" WHERE "attributes"::jsonb -> 'orgs' ? 'orga'

	// Check JSON extract value from keys equal to value
	//datatypes.JSONQuery("attributes").Equals(value, keys...)

	DB.First(&users, datatypes.JSONQuery("attributes").Equals("jinzhu", "name"))
	DB.First(&users, datatypes.JSONQuery("attributes").Equals("orgb", "orgs", "orgb"))
	// MySQL
	// SELECT * FROM `user` WHERE JSON_EXTRACT(`attributes`, '$.name') = "jinzhu"
	// SELECT * FROM `user` WHERE JSON_EXTRACT(`attributes`, '$.orgs.orgb') = "orgb"

	// PostgreSQL
	// SELECT * FROM "user" WHERE json_extract_path_text("attributes"::json,'name') = 'jinzhu'
	// SELECT * FROM "user" WHERE json_extract_path_text("attributes"::json,'orgs','orgb') = 'orgb'
}

func TestGormInsert() {
	db := g.ProduceDB
	// rt := []nmdb.SysNavigationlockRuntimeInfo{
	// 	{NavigationlockId: "#1", ScheduleId: "0120220703065719", ScheduleStatus: "上行进闸",
	// 		Info: []byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)},
	// 	{NavigationlockId: "#2", ScheduleId: "0120220703065720", ScheduleStatus: "上行进闸",
	// 		Info: []byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)},
	// 	{NavigationlockId: "#3", ScheduleId: "0120220703065730", ScheduleStatus: "上行出闸",
	// 		Info: []byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)},
	// 	{NavigationlockId: "#1", ScheduleId: "0120220703065730", ScheduleStatus: "上行出闸",
	// 		Info: []byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)},
	// 	{NavigationlockId: "#3", ScheduleId: "0120220703065730", ScheduleStatus: "上行出闸",
	// 		Info: []byte(`{"name": "json-2", "age": 28, "tags": ["tag1", "tag3"], "role": "admin", "orgs": {"orgb": "orgb"}}`)},
	// }
	// if err := db.Create(&rt).Error; err != nil {
	// 	g.Glog.Sugar().Infof("Failed to create SysNavigationlockRuntimeInfo: %+v", err)
	// }

	// srth := nmdb.SysRuntimeHistory{NavigationlockId: "#3", ScheduleId: "123213",
	// 	ScheduleStatus: "上行出闸", ServerName: "123", SystemName: "123", DriverName: "123", FormatVersion: 1, InfoLevel: 1,
	// 	Info: []byte(`{"db.Stats":"qps"}`)}
	// srth := nmdb.SysRuntimeHistory{Info: []byte(`{"db.Stats":"qps"}`)}
	// dbs := db.Create(&srth)
	// if db.Error != nil {
	// 	g.Glog.Sugar().Debugf("err: %+v", dbs.Error)
	// }

	// // Create
	begin := time.Now()
	count := 10
	for i := 0; i < count; i++ {
		sysRTH := IotNavlCpntState{}
		sysRTH.ScheduleId = fmt.Sprintf("%d", i)
		sysRTH.Info = fmt.Sprintf(`{"i":%d}`, i)
		dbs := db.Create(&sysRTH)
		if db.Error != nil {
			mdl.L.Sugar().Debugf("err: %+v", dbs.Error)
		}
	}
	timeInterval := time.Since(begin).Milliseconds()
	qps := float64(0.0)
	if timeInterval > 0 {
		qps = float64(count) / float64(timeInterval/1000)
	}
	mdl.L.Sugar().Debugf("qps: %G", qps)
}

func SqliteTest(ctx context.Context) {
	TestSqlite3(ctx)
}
