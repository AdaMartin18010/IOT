package web

type NavPlayScreenSetup struct {
	Id        int    `gorm:"column:id" json:"id"`
	SortId    int    `gorm:"column:sortId;index:idx_led" json:"sortId"`       //排序标识
	LedText   string `gorm:"column:ledText;index:idx_led" json:"ledText"`     //led文字
	NavlockId int    `gorm:"column:navLockId;index:idx_led" json:"navLockId"` //船闸名称ID
	Remark    string `gorm:"column:remark" json:"remark"`                     //备注
}

func NewNavPlayScreenSetup(Navlockid int) *NavPlayScreenSetup {
	return &NavPlayScreenSetup{NavlockId: Navlockid}
}

func (nls *NavPlayScreenSetup) TableName() string {
	return "nav_led_setup"
}
