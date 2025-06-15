package gezhouba

type Navigationlocks struct {
	NavlockConf01 *NavlockConf `mapstructure:"navigationlock01" json:"navigationlock01" yaml:"navigationlock01"`
	NavlockConf02 *NavlockConf `mapstructure:"navigationlock02" json:"navigationlock02" yaml:"navigationlock02"`
	NavlockConf03 *NavlockConf `mapstructure:"navigationlock03" json:"navigationlock03" yaml:"navigationlock03"`
}

func (nlc *Navigationlocks) Validate() error {
	if nlc.NavlockConf01 != nil {
		return nlc.NavlockConf01.Validate()
	}

	if nlc.NavlockConf02 != nil {
		return nlc.NavlockConf02.Validate()
	}

	if nlc.NavlockConf03 != nil {
		return nlc.NavlockConf03.Validate()
	}

	return nil
}
