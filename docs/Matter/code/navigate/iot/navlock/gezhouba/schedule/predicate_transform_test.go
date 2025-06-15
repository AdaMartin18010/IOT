package schedule_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	cm "navigate/common/model"
	gzbapi_navls "navigate/iot/navlock/gezhouba/api/navlstatus"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
)

func PredicateNavStatus(t *testing.T, xmlss string, navDynData *gzb_sche.NavlockDynamicStatusData) {
	xmlss = strings.ReplaceAll(xmlss, "\n", "")
	xmlss = strings.ReplaceAll(xmlss, "\t", "")
	// 会造成 时间字符串 解析出错
	//xmlss = strings.ReplaceAll(xmlss, " ", "")
	begin := strings.Index(xmlss, ">{")
	end := strings.Index(xmlss, "}<")
	jsonss := ""
	var MonitorInfoStruct gzbapi_navls.NavlStatusResp
	if begin >= 0 && end >= 0 && end >= begin {
		jsonss = xmlss[(begin + 1):(end + 1)]
		err := cm.Json.UnmarshalFromString(jsonss, &MonitorInfoStruct)
		if err != nil {
			t.Errorf("err: %+v", err)
		} else {
			fmt.Printf("解析后的数据对应字段: \n MonitorInfoStruct: %+v\n", MonitorInfoStruct)
		}

		MonitorInfoStruct.TransformValue()
		fmt.Printf("解析后转换信号灯的数据对应字段: \n MonitorInfoStruct: %+v\n", MonitorInfoStruct)
	} else {
		t.Errorf("xmlss parse errno: begin:%d ,end:%d \n", begin, end)
	}

	if err := navDynData.TransformFrom(&MonitorInfoStruct); err != nil {
		fmt.Printf("TransformFrom error: %+v\n", err)
		return
	}

	//_, err := navDynData.PredicateBaseOnPhysicalGatesStates()
	_, err := navDynData.PredicateBaseOnPhysicalLightStates()
	if err != nil {
		fmt.Printf("TransformAndValidData error: %+v\n", err)
		return
	}
}

func TestResponseJsonStruct(t *testing.T) {
	fmt.Printf("------------------------------%s--------begin--------------------------\n", "测试数据1")
	xmlss01 := `<string xmlns="http://tempuri.org/">
	{"time":"2023-03-06 11:12:43","SZZM_GYX":"0","SYZM_GYX":"0","XZZM_GYX":"0","XYZM_GYX":"0","SZZM_KZ":"1","SYZM_KZ":"1","XZZM_KZ":"0",
	"XYZM_KZ":"0","SZZM_GZ":"0","SYZM_GZ":"0","XZZM_GZ":"1","XYZM_GZ":"1","SZZM_KYX":"0","SYZM_KYX":"0",
	"XZZM_KYX":"0","XYZM_KYX":"0","SZ_XHD":"0","SY_XHD":"1","XZ_XHD":"1","XY_XHD":"1","TH_ZT":"1","SX_ZT":"1",
	"XX_ZT":"0","ZSSY_SW":"64.26","ZSXY_SW":"64.04","SZZM_KD":"100.13","SYZM_KD":"100.13","XZZM_KD":"0.32",
	"XYZM_KD":"-0.06","SY_SW":"64.3","XY_SW":"44.46"}</string>`

	fmt.Printf("原始数据:\n %s\n", xmlss01)
	var navDynData gzb_sche.NavlockDynamicStatusData
	PredicateNavStatus(t, xmlss01, &navDynData)
	if navDynData.NavState != 0 && navDynData.NavlockState != 0 {
		//navDynData.BeginTime = time.Now()
		navDynData.NavStateLast = navDynData.NavState
		navDynData.NavlockStateLast = navDynData.NavlockState
	}

	fmt.Printf("解析转换后 一系列判断后的数据对应结构 可读信息: \n navDynData: %+v\n", navDynData)
	fmt.Printf("对应的json结构 navDynDafta  程序信息: %s\n", navDynData.String())
	fmt.Printf("-------------------------------%s--------end---------------------------------\n", "测试数据1")

	fmt.Printf("------------------------------%s--------begin--------------------------\n", "测试数据2")
	xmlss02 := `<string xmlns="http://tempuri.org/">{"time":"2023-03-06 11:15:43","SZZM_GYX":"0","SYZM_GYX":"0","XZZM_GYX":"0",
	"XYZM_GYX":"0","SZZM_KZ":"0","SYZM_KZ":"0","XZZM_KZ":"1","XYZM_KZ":"1","SZZM_GZ":"1","SYZM_GZ":"1",
	"XZZM_GZ":"0","XYZM_GZ":"0","SZZM_KYX":"0","SYZM_KYX":"0","XZZM_KYX":"0","XYZM_KYX":"0","SZ_XHD":"1","SY_XHD":"1",
	"XZ_XHD":"0","XY_XHD":"1","TH_ZT":"1","SX_ZT":"0","XX_ZT":"1","ZSSY_SW":"43.54","ZSXY_SW":"43.64","SZZM_KD":"-0.6",
	"SYZM_KD":"1.11","XZZM_KD":"99.93","XYZM_KD":"99.66","SY_SW":"63.69","XY_SW":"43.71"}</string>`

	fmt.Printf("原始数据:\n %s\n", xmlss02)
	navDynData.ReSet(false)
	PredicateNavStatus(t, xmlss02, &navDynData)
	if navDynData.NavState != 0 && navDynData.NavlockState != 0 {
		//navDynData.BeginTime = time.Now()
		navDynData.NavStateLast = navDynData.NavState
		navDynData.NavlockStateLast = navDynData.NavlockState
	}

	fmt.Printf("解析转换后 一系列判断后的数据对应结构 可读信息: \n navDynData: %+v\n", navDynData)
	fmt.Printf("对应的json结构 navDynData  程序信息: %s\n", navDynData.String())
	fmt.Printf("-------------------------------%s--------end---------------------------------\n", "测试数据2")

	fmt.Printf("------------------------------%s--------begin--------------------------\n", "测试数据3")
	xmlss03 := `<string xmlns="http://tempuri.org/">{"time":"2023-03-06 11:30:43","SZZM_GYX":"0","SYZM_GYX":"0","XZZM_GYX":"0","XYZM_GYX":"0","SZZM_KZ":"0",
	"SYZM_KZ":"0","XZZM_KZ":"1","XYZM_KZ":"1","SZZM_GZ":"1","SYZM_GZ":"1","XZZM_GZ":"0","XYZM_GZ":"0","SZZM_KYX":"0",
	"SYZM_KYX":"0","XZZM_KYX":"0","XYZM_KYX":"0","SZ_XHD":"1","SY_XHD":"1","XZ_XHD":"1","XY_XHD":"0","TH_ZT":"1",
	"SX_ZT":"0","XX_ZT":"1","ZSSY_SW":"43.2","ZSXY_SW":"43.2","SZZM_KD":"0","SYZM_KD":"0","XZZM_KD":"100.13",
	"XYZM_KD":"100.26","SY_SW":"63.23","XY_SW":"43.2"}</string>`
	fmt.Printf("原始数据:\n %s\n", xmlss03)
	navDynData.ReSet(false)
	PredicateNavStatus(t, xmlss03, &navDynData)
	navDynData.EndTime = time.Now()
	fmt.Printf("解析转换后 一系列判断后的数据对应结构 可读信息: \n navDynData: %+v\n", navDynData)
	sss := navDynData.String()
	fmt.Printf("对应的json结构 navDynData 程序信息: %s\n", sss)
	var temp gzb_sche.NavlockDynamicStatusData
	err := cm.Json.UnmarshalFromString(sss, &temp)
	if err != nil {
		t.Errorf("err: %+v", err)
	} else {
		fmt.Printf("反向解码json后的结构:\n%+v\n", temp)
	}

	fmt.Printf("------------------------------%s--------end--------------------------\n", "测试数据3")
}

func UnmarshalFromString(t *testing.T, sss *string, temp *gzb_sche.NavlockDynamicStatusData) error {
	err := cm.Json.UnmarshalFromString(*sss, temp)
	if err != nil {
		t.Errorf("err: %+v", err)
	} else {
		fmt.Printf("反向解码json后的结构:\n %+v \n", temp)
	}
	return err
}

func WrapPredicateCase(t *testing.T, temp *gzb_sche.NavlockDynamicStatusData) {
	//ret, err := temp.PredicateBaseOnPhysicalGatesStates()
	ret, err := temp.PredicateBaseOnPhysicalLightStates()
	fmt.Printf("return: code:%d,note:%s, err: %+v,\nNavigationlockDynamicData:%+v\n", ret, ret.String(), err, temp)
}

func TestPredicateCases(t *testing.T) {
	var temp gzb_sche.NavlockDynamicStatusData
	sss := `{"NavStatusPredicated":"0","NavlockStatusPredicated":"0",
"PredicateBeginTime":"0001-01-01T00:00:00Z","PredicateEndTime":"0001-01-01T00:00:00Z",
"NavStatusLasttime":"0","NavlockStatusLasttime":"0","UpInnerWaterLasttime":"0","DownInnerWaterLasttime":"0",
"NavStatus":"3","UpWater":"64.24","UpInnerWater":"47.6","DownWater":"42.86","DownInnerWater":"47.7",
"UpGates":{"Left":{"State":4,"OpenAngle":"-0.6"},"Right":{"State":4,"OpenAngle":"-0.79"}},
"DownGates":{"Left":{"State":4,"OpenAngle":"-0.26"},"Right":{"State":4,"OpenAngle":"-2.2"}},
"UpSignalLights":{"Left":0,"Right":0},"DownSignalLights":{"Left":0,"Right":0}}`
	temp.ReSet(false)
	UnmarshalFromString(t, &sss, &temp)
	temp.BeginTime = time.Now()
	temp.UpInnerWaterLast = temp.TUpInnerWater
	temp.DownInnerWaterLast = temp.TDownInnerWater
	UpInnerWater := temp.TUpInnerWater
	DownInnerWater := temp.TDownInnerWater

	fmt.Printf("--------------\n%+v\n", temp)
	WrapPredicateCase(t, &temp)
	fmt.Printf("--------------\n%+v\n", temp)

	sss = `{"NavStatusPredicated":"0","NavlockStatusPredicated":"0",
"PredicateBeginTime":"0001-01-01T00:00:00Z","PredicateEndTime":"0001-01-01T00:00:00Z",
"NavStatusLasttime":"0","NavlockStatusLasttime":"0","UpInnerWaterLasttime":"0","DownInnerWaterLasttime":"0",
"NavStatus":"3","UpWater":"64.16","UpInnerWater":"50.68","DownWater":"42.82","DownInnerWater":"50.69",
"UpGates":{"Left":{"State":4,"OpenAngle":"-0.6"},"Right":{"State":4,"OpenAngle":"-0.79"}},
"DownGates":{"Left":{"State":4,"OpenAngle":"-0.26"},"Right":{"State":4,"OpenAngle":"-2.2"}},
"UpSignalLights":{"Left":0,"Right":0},"DownSignalLights":{"Left":0,"Right":0}}`
	temp.ReSet(false)
	UnmarshalFromString(t, &sss, &temp)
	temp.BeginTime = time.Now()
	temp.UpInnerWaterLast = UpInnerWater
	temp.DownInnerWaterLast = DownInnerWater

	time.Sleep(time.Second * 2)

	fmt.Printf("--------------\n%+v\n", temp)
	WrapPredicateCase(t, &temp)
	fmt.Printf("--------------\n%+v\n", temp)
}
