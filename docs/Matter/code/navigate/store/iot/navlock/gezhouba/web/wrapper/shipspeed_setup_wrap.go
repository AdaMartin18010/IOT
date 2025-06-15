package wrapper

import (
	"fmt"

	"sync"

	cm "navigate/common/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

type NavShipspeedSetup = gzb_db_web.NavShipSpeedSetup

const (
	DistanceMax = 350
)

type DistanceSpeedMax struct {
	Distance float64 `gorm:"column:distance" json:"distance"` //距离雷达长度
	SpeedMax float64 `gorm:"column:speedMax" json:"speedMax"` //最大速度
}

type DisSpeedMaxArray []DistanceSpeedMax

// 只检查距离的有效值
func (dsa DisSpeedMaxArray) CheckValid(distanceMax float64) bool {
	if len(dsa) == 0 {
		return false
	}

	//假设按照距离由小往大排序
	if dsa[len(dsa)-1].Distance-distanceMax > 0.01 {
		return false
	}

	return true
}

func (dsa DisSpeedMaxArray) CheckOverSpeed(distance, speed float64) bool {
	l := len(dsa) - 1
	compareSpeed := 0.0
	//从小往大的距离判断
	for i := 0; i < l; i++ {
		if dsa[i].Distance-distance >= 0.0 {
			compareSpeed = dsa[i].SpeedMax
			break
		}
	}

	//即未取到区间值
	if compareSpeed == 0.0 {
		//对比最后一个区间
		if dsa[l].Distance-distance >= 0.0 {
			compareSpeed = dsa[l].SpeedMax
		} else {
			return false
		}
	}

	//如果取到值 并且比该值大
	if speed-compareSpeed > 0.01 {
		return true
	}

	return false
}

func (dsa DisSpeedMaxArray) String() string {
	rs, _ := cm.Json.Marshal(dsa)
	return string(rs)
}

type NavShipSpeedSetupWrap struct {
	rwm             *sync.RWMutex
	shipSpeed       *NavShipspeedSetup
	buf             DisSpeedMaxArray
	defaultSpeedMax float32
	useDefaultSpeed bool
}

func NewNavShipSpeedSetupWrap(Navlockid int) *NavShipSpeedSetupWrap {
	tmp := &NavShipSpeedSetupWrap{
		rwm:             &sync.RWMutex{},
		shipSpeed:       &NavShipspeedSetup{NavlockId: int64(Navlockid)},
		defaultSpeedMax: 1.5,
		useDefaultSpeed: true,
	}
	return tmp
}

func (ssw *NavShipSpeedSetupWrap) Infos() string {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	rs, _ := cm.Json.Marshal(ssw.shipSpeed)
	return string(rs)
}

func (ssw *NavShipSpeedSetupWrap) LoadFromDB(db *gorm.DB, l *zap.Logger) (err error) {
	ssw.rwm.Lock()
	defer ssw.rwm.Unlock()

	if db == nil || l == nil {
		ssw.useDefaultSpeed = true
		return fmt.Errorf("db or l cannot be nil")
	}
	tx := db.Begin()
	res := new([]DistanceSpeedMax)
	result := tx.Model(&NavShipspeedSetup{}).Where("navLockId = ?", ssw.shipSpeed.NavlockId).Limit(100).Order("distance ASC").Find(res)
	if result.Error != nil {
		l.Sugar().Errorf(`get ShipSpeed setup from  db table %s error: %+v`, ssw.shipSpeed.TableName(), result.Error)
		ssw.useDefaultSpeed = true
		tx.Rollback()
		return result.Error
	}

	if result.RowsAffected > 0 {
		l.Sugar().Debugf("%+v", res)
	}
	ssw.buf = DisSpeedMaxArray(*res)
	if ssw.buf.CheckValid(DistanceMax) {
		ssw.useDefaultSpeed = false
	} else {
		ssw.useDefaultSpeed = true
	}
	return tx.Commit().Error
}

func (ssw *NavShipSpeedSetupWrap) CheckOverSpeed(distance, speed float64) bool {
	ssw.rwm.RLock()
	defer ssw.rwm.RUnlock()

	if ssw.useDefaultSpeed {
		return (speed-float64(ssw.defaultSpeedMax) > 0.01)
	} else {
		return ssw.buf.CheckOverSpeed(distance, speed)
	}

}
