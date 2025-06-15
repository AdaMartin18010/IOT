package monitor_test

import (
	"fmt"
	navls "navigate/iot/navlock/gezhouba/api/navlstatus"
	"strings"
	"testing"
	"time"

	cm "navigate/common/model"
)

func ReplaceAll(s *string) *string {
	*s = strings.ReplaceAll(*s, "\n", "")
	*s = strings.ReplaceAll(*s, "\t", "")
	//*s = strings.ReplaceAll(*s, " ", "")
	return s
}

func TestResponseJsonStruct(t *testing.T) {
	xmlss := `<?xml version="1.0" encoding="utf-8"?>
	<string xmlns="http://tempuri.org/">
	{"time":"2023-03-06 11:12:43","SZZM_GYX":"0","SYZM_GYX":"0","XZZM_GYX":"0","XYZM_GYX":"0","SZZM_KZ":"1","SYZM_KZ":"1",
	"XZZM_KZ":"0","XYZM_KZ":"0","SZZM_GZ":"0","SYZM_GZ":"0","XZZM_GZ":"1","XYZM_GZ":"1","SZZM_KYX":"0",
	"SYZM_KYX":"0","XZZM_KYX":"0","XYZM_KYX":"0","SZ_XHD":"0","SY_XHD":"1","XZ_XHD":"1","XY_XHD":"1",
	"TH_ZT":"1","SX_ZT":"1","XX_ZT":"0","SY_SW":"63.8","XY_SW":"43.91"}</string>
`
	xmlss = *ReplaceAll(&xmlss)
	begin := strings.Index(xmlss, ">{")
	end := strings.Index(xmlss, "}<")
	jsonss := ""
	if begin >= 0 && end >= 0 && end >= begin {
		jsonss = xmlss[(begin + 1):(end + 1)]
		var MonitorInfoStruct navls.NavlStatusResp
		err := cm.Json.UnmarshalFromString(jsonss, &MonitorInfoStruct)
		t.Logf("jsonss: %v", jsonss)
		//err := cm.Json.Unmarshal([]byte(jsonss), &MonitorInfoStruct)
		if err != nil {
			t.Errorf("err: %+v", err)
		} else {
			fmt.Printf("MonitorInfoStruct: %+v\n", MonitorInfoStruct)
		}
		MonitorInfoStruct.TransformValue()
		fmt.Printf("MonitorInfoStruct: %v\n", MonitorInfoStruct)

		tm := (time.Time)(MonitorInfoStruct.Time)
		t.Logf("time: %v\n", tm)
	} else {
		t.Errorf("xmlss parse errno: begin:%d ,end:%d \n", begin, end)
	}
}

func TestTimeParse(t *testing.T) {
	tmp := "2023-03-06 11:12:43"
	tm, e := time.ParseInLocation(time.DateTime, tmp, time.Local)
	if e != nil {
		t.Errorf("err: %+v", e)
	} else {
		t.Logf("time: %+v", tm)
	}
}
