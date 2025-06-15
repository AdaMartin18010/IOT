package playscreen

type ProgramStruct struct {
	Programs Programs `json:"Programs"`
}

type Programs struct {
	Program Program `json:"Program"`
}

type Program struct {
	Pages []Page `json:"Pages"`
}

type Page struct {
	Width   string   `json:"Width"`
	Height  string   `json:"Height"`
	Regions []Region `json:"Regions"`
}

type Region struct {
	Layer int    `json:"Layer"`
	Rect  Rect   `json:"Rect"`
	Items []Item `json:"Items"`
}

type Rect struct {
	X      string `json:"X"`
	Y      string `json:"Y"`
	Width  string `json:"Width"`
	Height string `json:"Height"`
}

type Item struct {
	Type          string  `json:"Type"`
	Text          string  `json:"Text"`
	TextColor     string  `json:"TextColor"`
	IsScroll      int     `json:"IsScroll"`
	Speed         int     `json:"Speed"`
	CenteralAlign int     `json:"CenteralAlign"`
	LogFont       LogFont `json:"LogFont"`
}

type LogFont struct {
	LfHeight    string `json:"lfHeight"`
	LfWeight    string `json:"lfWeight"`
	LfItalic    string `json:"lfItalic"`
	LfUnderline string `json:"lfUnderline"`
	LfFaceName  string `json:"lfFaceName"`
}

// 更新多个文本
