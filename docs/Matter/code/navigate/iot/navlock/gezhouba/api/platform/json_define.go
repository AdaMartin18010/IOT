package platform

import (
	"fmt"
	"time"

	cm "navigate/common/model"
)

const (
	timeformat     = "2006-01-02 15:04:05"
	expirationTime = time.Minute * 15
)

//平台接口

type SpeedlimitRecord struct {
	Id             int       `json:"Sid,string"`
	Time           time.Time `json:"Time"`
	NavlockId      string    `json:"NavlockId"`
	ScheduleId     string    `json:"ScheduleId"`
	ScheduleStatus string    `json:"ScheduleStatus"`
	DeviceTag      string    `json:"DeviceTag"`
	Warn           string    `json:"Warn"`
	Speed          float32   `json:"Speed,string"`
	//RadarTag       string    `json:"RadarTag"`
}

type StoplineRecord struct {
	Id         int       `json:"Sid,string"`
	Time       time.Time `json:"Time"`
	NavlockId  string    `json:"NavlockId"`
	ScheduleId string    `json:"ScheduleId"`
	//	ScheduleStatus string    `json:"ScheduleStatus"`
	DeviceTag     string `json:"DeviceTag"`
	CrossLocation string `json:"CrossLocation"`
	CrossLevel    string `json:"CrossLevel"`
	StoplineWidth int    `json:"StoplineWidth,string"`
	CrossDistance int    `json:"CrossDistance,string"`
}

// create table
type Table struct {
	TableName   string            `json:"TableName"`
	TableColumn map[string]string `json:"TableColumn"`
}

// map 顺序不是按照初始化的顺序
func (t *Table) Json() (rs []byte, err error) {
	// t = &Table{
	// 	TableName: "speedlimit",
	// 	TableColumn: map[string]string{
	// 		"Id":             "序号-int",
	// 		"Time":           "时间-UTC",
	// 		"NavlockId":      "船闸名称-str",
	// 		"ScheduleId":     "闸次-str",
	// 		"ScheduleStatus": "船闸调度状态-str",
	// 		"DeviceTag":      "设备标识IP-str",
	// 		"Warn":           "警告Flag-str",
	// 		"Speed":          "船速(m/s)-str",
	// 		"RadarTag":       "雷达Tag-str",
	// 	}}
	rs, err = cm.Json.Marshal(t)
	return
}

func (t *Table) TransformFromStopline(slr *StoplineRecord) (rs []byte, err error) {
	t = &Table{
		TableName: "speedlimit",
		TableColumn: map[string]string{
			"Sid":           fmt.Sprintf("%d", slr.Id),
			"Time":          slr.Time.String(),
			"NavlockId":     slr.NavlockId,
			"ScheduleId":    slr.ScheduleId,
			"DeviceTag":     slr.DeviceTag,
			"CrossLocation": slr.CrossLocation,
			"CrossLevel":    slr.CrossLevel,
			"StoplineWidth": fmt.Sprintf("%d", slr.StoplineWidth),
			"CrossDistance": fmt.Sprintf("%d", slr.CrossDistance),
		}}
	rs, err = cm.Json.Marshal(t)
	slr = nil
	return
}

func (t *Table) TransformFromSpeedlimit(slr *SpeedlimitRecord) (rs []byte, err error) {
	t = &Table{
		TableName: "speedlimit",
		TableColumn: map[string]string{
			"Sid":            fmt.Sprintf("%d", slr.Id),
			"Time":           slr.Time.String(),
			"NavlockId":      slr.NavlockId,
			"ScheduleId":     slr.ScheduleId,
			"ScheduleStatus": slr.ScheduleStatus,
			"DeviceTag":      slr.DeviceTag,
			"Warn":           slr.Warn,
			"Speed":          fmt.Sprintf("%.3f", slr.Speed),
		}}

	rs, err = cm.Json.Marshal(t)
	return
}

func (t *Table) SpeedlimitStr() string {
	return `{
	"TableName":"speedlimit",
	"TableColumn":{
		"Sid":"序号-int",
		"Time":"时间-UTC",
		"NavlockId":"船闸名称-str",
		"ScheduleId":"闸次-str",
		"ScheduleStatus":"船闸调度状态-str",
		"DeviceTag":"设备标识IP-str",
		"Warn":"警告Flag-str",
		"Speed":"船速(m/s)-str"
		}
	}`
}

func (t *Table) StopLineStr() string {
	return `{
	"TableName":"stopline",
	"TableColumn":{
		"Sid":"序号-int",
		"Time":"时间-UTC",
		"NavlockId":"船闸名称-str",
		"ScheduleId":"闸次-str",
		"CrossLocation":"越线位置-str",
		"CrossLevel":"越线等级-str",
		"DeviceTag":"设备标识IP-str",
		"StoplineWidth":"禁停线绿色区域宽度(cm)",
		"CrossDistance":"越线距离(cm)"
		}
	}`
}

type Response struct {
	Flag bool   `json:"flag"`
	Err  string `json:"err"`
	Data any    `json:"Data"`
}

func (resp *Response) ParseTokenMsg(tkmsg *HttpTokenMsg) (err error) {
	ss, ok := resp.Data.(map[string]any)
	if !ok {
		return fmt.Errorf(`Response.Data is not a map[string]any type")`)
	}

	tkmsg.Token, ok = ss["token"].(string)
	if !ok {
		return fmt.Errorf(`Response.Data["token"].(string) is not string type`)
	}

	tmp, ok := ss["expiration"].(string)
	if !ok {
		return fmt.Errorf(`Response.Data["expiration"].(string) is not string type`)
	}

	tkmsg.Expiration, err = time.ParseInLocation(timeformat, tmp, time.Local)
	if err != nil {
		return fmt.Errorf(`parse Response.Data["expiration"] to time.time error: %v`, err)
	}

	tkmsg.Type, ok = ss["type"].(string)
	if !ok {
		return fmt.Errorf(`Response.Data["type"].(string) is not string type`)
	}
	return nil
}

type HttpTokenMsg struct {
	Token      string    `json:"token"`
	Expiration time.Time `json:"expiration"`
	Type       string    `json:"type"`
}

func (t *HttpTokenMsg) IsTimeToGetToken() bool {
	if (t.Expiration == time.Time{}) || (t.Expiration.IsZero()) {
		return true
	}

	if time.Until(t.Expiration) < expirationTime {
		return true
	}

	return false
}

type PlatformData struct {
	Key     string        `json:"key"`
	Value   string        `json:"value"`
	TokenSt *HttpTokenMsg `json:"-"`
}
