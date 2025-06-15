package playscreen

import (
	playscreen_api "navigate/iot/navlock/gezhouba/api/playscreen"
)

const (
	ProgramUnknown ProgramType = 0 //无状态与零值对应-0
	PrograDefault  ProgramType = 1 //正常状态
	ProgramSpeed   ProgramType = 2 //如果为雷达测速阶段-上下行进出闸 正常展示速度状态
	//ProgramOverSpeed ProgramType = 3 //如果为雷达测速阶段 超速状态展示
)

var (
	//Program状态对照map
	programTypeName = map[ProgramType]string{
		0: "ProgramUnknown",
		1: "ProgramDefault",
		2: "ProgramSpeed",
		//3: "ProgramOverSpeed",
	}

	Navl1_PrograNormal           playscreen_api.ProgramStruct
	Navl1_PrograNormal_lines     playscreen_api.UpdateMultiText
	Navl1_ProgramSpeed           playscreen_api.ProgramStruct
	Navl1_ProgramSpeed_lines     playscreen_api.UpdateMultiText
	Navl1_ProgramOverSpeed       playscreen_api.ProgramStruct
	Navl1_ProgramOverSpeed_lines playscreen_api.UpdateMultiText

	Navl2_PrograNormal           playscreen_api.ProgramStruct
	Navl2_PrograNormal_lines     playscreen_api.UpdateMultiText
	Navl2_ProgramSpeed           playscreen_api.ProgramStruct
	Navl2_ProgramSpeed_lines     playscreen_api.UpdateMultiText
	Navl2_ProgramOverSpeed       playscreen_api.ProgramStruct
	Navl2_ProgramOverSpeed_lines playscreen_api.UpdateMultiText

	Navl3_PrograNormal           playscreen_api.ProgramStruct
	Navl3_PrograNormal_lines     playscreen_api.UpdateMultiText
	Navl3_ProgramSpeed           playscreen_api.ProgramStruct
	Navl3_ProgramSpeed_lines     playscreen_api.UpdateMultiText
	Navl3_ProgramOverSpeed       playscreen_api.ProgramStruct
	Navl3_ProgramOverSpeed_lines playscreen_api.UpdateMultiText
)

type ProgramType int

func (pt ProgramType) String() string {
	switch pt {
	case PrograDefault:
		return programTypeName[PrograDefault]
	case ProgramSpeed:
		return programTypeName[ProgramSpeed]
	// case ProgramOverSpeed:
	// 	return programTypeName[ProgramOverSpeed]
	default:
		return programTypeName[0]
	}
}

func navl1ProgramInit() {
	Navl1_PrograNormal = playscreen_api.ProgramStruct{
		Programs: playscreen_api.Programs{
			Program: playscreen_api.Program{
				Pages: []playscreen_api.Page{
					{
						Width:  "320",
						Height: "160",
						Regions: []playscreen_api.Region{
							{
								Layer: 1,
								Rect: playscreen_api.Rect{
									X:      "0",
									Y:      "0",
									Width:  "320",
									Height: "160",
								},
								Items: []playscreen_api.Item{
									{
										Type:          "4",
										Text:          "0.9M/S",
										TextColor:     "#FFF0F5",
										IsScroll:      0,
										Speed:         15,
										CenteralAlign: 1,
										LogFont: playscreen_api.LogFont{
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

	Navl1_PrograNormal_lines = playscreen_api.UpdateMultiText{
		Update: []playscreen_api.Update{
			{
				Who: playscreen_api.Who{X: "1",
					Y: "1"},
				Text: "1",
			},
			{
				Who: playscreen_api.Who{X: "2",
					Y: "2"},
				Text: "1",
			},
		},
	}

	Navl1_ProgramSpeed = playscreen_api.ProgramStruct{
		Programs: playscreen_api.Programs{
			Program: playscreen_api.Program{
				Pages: []playscreen_api.Page{
					{
						Width:  "320",
						Height: "160",
						Regions: []playscreen_api.Region{
							{
								Layer: 1,
								Rect: playscreen_api.Rect{
									X:      "0",
									Y:      "0",
									Width:  "320",
									Height: "160",
								},
								Items: []playscreen_api.Item{
									{
										Type:          "4",
										Text:          "0.9M/S",
										TextColor:     "#FFF0F5",
										IsScroll:      0,
										Speed:         15,
										CenteralAlign: 1,
										LogFont: playscreen_api.LogFont{
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

	Navl1_ProgramSpeed_lines = playscreen_api.UpdateMultiText{
		Update: []playscreen_api.Update{
			{
				Who: playscreen_api.Who{X: "1",
					Y: "1"},
				Text: "1",
			},
			{
				Who: playscreen_api.Who{X: "2",
					Y: "2"},
				Text: "1",
			},
		},
	}

	Navl1_ProgramOverSpeed = playscreen_api.ProgramStruct{
		Programs: playscreen_api.Programs{
			Program: playscreen_api.Program{
				Pages: []playscreen_api.Page{
					{
						Width:  "320",
						Height: "160",
						Regions: []playscreen_api.Region{
							{
								Layer: 1,
								Rect: playscreen_api.Rect{
									X:      "0",
									Y:      "0",
									Width:  "320",
									Height: "160",
								},
								Items: []playscreen_api.Item{
									{
										Type:          "4",
										Text:          "0.9M/S",
										TextColor:     "#FFF0F5",
										IsScroll:      0,
										Speed:         15,
										CenteralAlign: 1,
										LogFont: playscreen_api.LogFont{
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

	Navl1_ProgramOverSpeed_lines = playscreen_api.UpdateMultiText{
		Update: []playscreen_api.Update{
			{
				Who: playscreen_api.Who{X: "1",
					Y: "1"},
				Text: "1",
			},
			{
				Who: playscreen_api.Who{X: "2",
					Y: "2"},
				Text: "1",
			},
		},
	}

}

func init() {
	navl1ProgramInit()
}
