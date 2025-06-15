package playscreen

import (
	cm "navigate/common/model"
	"strings"
)

type Update struct {
	Who  Who    `json:"who"`
	Text string `json:"text"`
}

type Who struct {
	X string `json:"x"`
	Y string `json:"y"`
}

type UpdateMultiText struct {
	Update []Update `json:"update"`
}

func (umt UpdateMultiText) ToJson() ([]byte, error) {
	return cm.Json.Marshal(umt)
}

func (umt UpdateMultiText) String() (string, error) {
	bs, err := cm.Json.Marshal(umt)
	if err != nil {
		return "", err
	}
	return string(bs), nil
}

type TagLines struct {
	NavServerTag string `json:"nav_server_tag"`
	Who          struct {
		X string `json:"x"`
		Y string `json:"y"`
	} `json:"who"`
	Text string `json:"text"`
}

type MultilLinesTag struct {
	Update []*TagLines `json:"update"`
}

func (mlt MultilLinesTag) String() (string, error) {
	bs, err := cm.Json.Marshal(mlt)
	if err != nil {
		return "", err
	}
	return string(bs), nil
}

func (mlt MultilLinesTag) ToJson() ([]byte, error) {
	return cm.Json.Marshal(mlt)
}

func (mlt MultilLinesTag) SetTextByTag(tag, value string) bool {
	for _, v := range mlt.Update {
		if strings.TrimSpace(v.NavServerTag) == tag {
			v.Text = value
			return true
		}
	}
	return false
}

func (mlt MultilLinesTag) TransformTo() *UpdateMultiText {
	var rs UpdateMultiText
	for _, v := range mlt.Update {
		u := Update{
			Who: Who{X: v.Who.X,
				Y: v.Who.Y},
			Text: v.Text,
		}
		rs.Update = append(rs.Update, u)
	}
	return &rs
}
