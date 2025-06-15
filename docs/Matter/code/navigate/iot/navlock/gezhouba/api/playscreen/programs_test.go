package playscreen_test

import (
	"testing"

	cm "navigate/common/model"
	palyscreen "navigate/iot/navlock/gezhouba/api/playscreen"

	jsoniter "github.com/json-iterator/go"
)

func TestResponseJsonStruct(t *testing.T) {
	pgs := palyscreen.ProgramStruct{
		Programs: palyscreen.Programs{
			palyscreen.Program{
				Pages: []palyscreen.Page{
					{
						Width:  "320",
						Height: "160",
						Regions: []palyscreen.Region{
							{
								Layer: 1,
								Rect: palyscreen.Rect{
									X:      "0",
									Y:      "0",
									Width:  "320",
									Height: "160",
								},
								Items: []palyscreen.Item{
									{
										Type:          "4",
										Text:          "0.9M/S",
										TextColor:     "#FFF0F5",
										IsScroll:      0,
										Speed:         15,
										CenteralAlign: 1,
										LogFont: palyscreen.LogFont{
											LfHeight:    "86",
											LfWeight:    "86",
											LfItalic:    "0",
											LfUnderline: "0",
											LfFaceName:  "宋体",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	st, _ := cm.Json.MarshalToString(pgs)
	t.Logf("ProgramStruct: %s\n", st)
	var pss palyscreen.ProgramStruct
	err := cm.Json.UnmarshalFromString(st, &pss)
	if err != nil {
		t.Errorf("err: %+v", err)
	} else {
		t.Logf("ProgramStruct: %+v\n", pss)
	}
}

func TestTextLines(t *testing.T) {
	MultiText := palyscreen.UpdateMultiText{
		Update: []palyscreen.Update{
			{
				Who: palyscreen.Who{X: "1",
					Y: "1"},
				Text: "1",
			},
			{
				Who: palyscreen.Who{X: "2",
					Y: "2"},
				Text: "1",
			},
		},
	}

	st, _ := cm.Json.MarshalToString(MultiText)
	t.Logf("UpdateMultiText: %s\n", st)
	var pss palyscreen.UpdateMultiText
	err := cm.Json.UnmarshalFromString(st, &pss)
	if err != nil {
		t.Errorf("err: %+v", err)
	} else {
		t.Logf("UpdateMultiText: %+v\n", pss)
	}

	pss.Update[1].Text = "111"
	pss.Update[0].Text = "000"
	t.Logf("UpdateMultiText: %+v\n", pss)
	st, _ = cm.Json.MarshalToString(pss)
	t.Logf("UpdateMultiText: %s\n", st)
}

func TestMultiLine1203(t *testing.T) {
	var mls12, mls03 palyscreen.MultilLinesTag
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	var templete12 = []byte(`
	{
		"update":[
			{"nav_server_tag":"middle","who":{"x":"0","y":"0"},"text":"注意安全"}
		]
	}
	`)

	var templete03 = []byte(`
	{
		"update":[
			{
				"nav_server_tag":"left_up",
				"who":{
					"x":"0",
					"y":"0"
				},
				"text":"0.00"
			},
			{	
				"nav_server_tag":"right_up",
				"who":{
					"x":"160",
					"y":"0"
				},
				"text":"0.00"
			},
			{	
				"nav_server_tag":"left_down",
				"who":{
					"x":"0",
					"y":"80"
				},
				"text":""
			},
			{	
				"nav_server_tag":"right_down",
				"who":{
					"x":"160",
					"y":"80"
				},
				"text":"        "
			}
		]
	}
`)

	err := json.Unmarshal(templete12, &mls12)
	ss, err := mls12.String()
	t.Logf("error:%+v,mls12:%#v", err, ss)
	t.Logf("mls12.Update[0].NavServerTag:%+v", mls12.Update[0].NavServerTag)
	mls12.SetTextByTag("middle", "middle")
	ss, err = mls12.String()
	t.Logf("mls12:%+v", ss)
	ml12 := mls12.TransformTo()
	ss, err = ml12.String()
	t.Logf("ml12: %+v", ss)

	err = json.Unmarshal(templete03, &mls03)
	ss, err = mls03.String()
	t.Logf("error:%+v,mls03:%#v", err, ss)
	t.Logf("mls03.Update[0].NavServerTag:%+v", mls03.Update[0].NavServerTag)
	mls03.SetTextByTag("left_up", "left_up")
	mls03.SetTextByTag("right_up", "right_up")
	mls03.SetTextByTag("left_down", "left_down")
	mls03.SetTextByTag("right_down", "right_down")
	ss, err = mls03.String()
	t.Logf("mls03:%+v", ss)
	ml03 := mls03.TransformTo()
	ss, err = ml03.String()
	t.Logf("ml03: %+v", ss)
}
