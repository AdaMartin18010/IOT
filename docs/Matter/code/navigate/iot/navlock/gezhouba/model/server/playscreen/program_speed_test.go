package playscreen_test

import (
	"fmt"
	"testing"

	mdl "navigate/common/model"
	palyscreen_api "navigate/iot/navlock/gezhouba/api/playscreen"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
)

func TestTagMiddle(t *testing.T) {
	var Taglines palyscreen_api.MultilLinesTag
	testlines := string(`{
		"update": [
		  {
			"nav_server_tag": "middle",
			"who": {
			  "x": "0",
			  "y": "0"
			},
			"text": "注意安全"
		  }
		]
	  }`)
	if err := mdl.Json.UnmarshalFromString(testlines, &Taglines); err != nil {
		t.Logf(`transform program Speed multi line: err: %+v`, err)
	}
	Taglines.SetTextByTag("middle", "注意安全哦")
	ss, err := Taglines.String()
	t.Logf("Taglines:%+v,error:%+v", ss, err)

	sslines, err0 := Taglines.TransformTo().String()
	t.Logf("Taglines:%+v,error:%+v", sslines, err0)
}

func TestTagSetUp(t *testing.T) {
	var Taglines palyscreen_api.MultilLinesTag
	testlines := string(`{
		"update": [
		  {
			"nav_server_tag": "left_up",
			"who": {
			  "x": "0",
			  "y": "0"
			},
			"text": "0.0"
		  },
		  {
			"nav_server_tag": "right_up",
			"who": {
			  "x": "160",
			  "y": "0"
			},
			"text": "0.0"
		  },
		  {
			"nav_server_tag": "left_down",
			"who": {
			  "x": "0",
			  "y": "80"
			},
			"text": " "
		  },
		  {
			"nav_server_tag": "right_down",
			"who": {
			  "x": "160",
			  "y": "80"
			},
			"text": " "
		  }
		]
	  }`)

	if err := mdl.Json.UnmarshalFromString(testlines, &Taglines); err != nil {
		t.Logf(`transform program Speed multi line: err: %+v`, err)
	}
	Taglines.SetTextByTag("left_up", fmt.Sprintf("%.1f", 0.23))
	Taglines.SetTextByTag("left_down", " ")
	Taglines.SetTextByTag("right_up", fmt.Sprintf("%.1f", 2.23))
	Taglines.SetTextByTag("right_down", "超速")

	ss, err := Taglines.String()
	t.Logf("Taglines:%+v,error:%+v", ss, err)

	sslines, err0 := Taglines.TransformTo().String()
	t.Logf("Taglines:%+v,error:%+v", sslines, err0)

}

func TestTagSpeedSetUp(t *testing.T) {

	sps0 := gzb_sche.NewShipSpeeds()
	sps0.LeftSpeed.IsValid = true
	sps0.LeftSpeed.OverSpeed = false
	sps0.LeftSpeed.Speed = 1.23

	sps0.RightSpeed.IsValid = true
	sps0.RightSpeed.OverSpeed = false
	sps0.RightSpeed.Speed = 1.45

	str0, err0 := sps0.String()
	t.Logf("shipspeeds sps0 : %+v,error : %+v", str0, err0)

	sps := gzb_sche.NewShipSpeeds()
	sps.LeftSpeed.IsValid = true
	sps.LeftSpeed.OverSpeed = true
	sps.LeftSpeed.Speed = 2.12

	sps.RightSpeed.IsValid = true
	sps.RightSpeed.OverSpeed = true
	sps.RightSpeed.Speed = 1.57

	str, err := sps.String()
	t.Logf("shipspeeds sps: %+v,error : %+v", str, err)

	sps0.RightSpeed = nil
	sps.AggregationWith(sps0)
	str1, err1 := sps.String()
	t.Logf("shipspeeds AggregationWith: %+v,error : %+v", str1, err1)

	var Taglines palyscreen_api.MultilLinesTag
	testlines := string(`{
		"update": [
		  {
			"nav_server_tag": "left_up",
			"who": {
			  "x": "0",
			  "y": "0"
			},
			"text": "0.0"
		  },
		  {
			"nav_server_tag": "right_up",
			"who": {
			  "x": "160",
			  "y": "0"
			},
			"text": "0.0"
		  },
		  {
			"nav_server_tag": "left_down",
			"who": {
			  "x": "0",
			  "y": "80"
			},
			"text": " "
		  },
		  {
			"nav_server_tag": "right_down",
			"who": {
			  "x": "160",
			  "y": "80"
			},
			"text": " "
		  }
		]
	  }`)

	if err := mdl.Json.UnmarshalFromString(testlines, &Taglines); err != nil {
		t.Logf(`transform program Speed multi line: err: %+v`, err)
	}

	if sps.LeftSpeed != nil && sps.LeftSpeed.IsValid {
		if sps.LeftSpeed.OverSpeed {
			Taglines.SetTextByTag("left_down", "超速")
		} else {
			Taglines.SetTextByTag("left_down", " ")
		}
		Taglines.SetTextByTag("left_up", fmt.Sprintf("%.1f", sps.LeftSpeed.Speed))
	}

	if sps.RightSpeed != nil && sps.RightSpeed.IsValid {
		if sps.RightSpeed.OverSpeed {
			Taglines.SetTextByTag("right_down", "超速")
		} else {
			Taglines.SetTextByTag("right_down", " ")
		}
		Taglines.SetTextByTag("right_up", fmt.Sprintf("%.1f", sps.RightSpeed.Speed))
	}

	ss, err := Taglines.String()
	t.Logf("Taglines:%+v,error:%+v", ss, err)

	sslines, err0 := Taglines.TransformTo().String()
	t.Logf("Taglines:%+v,error:%+v", sslines, err0)

}
