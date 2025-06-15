package wrapper

import (
	"fmt"
	"sync"

	cm "navigate/common/model"
	gzb_db_web "navigate/store/iot/navlock/gezhouba/web"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	defaultLedText = "注意安全"
)

type NavLedSetup = gzb_db_web.NavPlayScreenSetup

type NavPlayScreenSetupWrap struct {
	rwm          *sync.RWMutex
	ledsetup     *NavLedSetup
	ledsetupLast *NavLedSetup
	isNotFirst   bool
}

// 默认是 "注意安全"
func NewNavPlayScreenSetupWrap(Navlockid int) *NavPlayScreenSetupWrap {
	tmp := &NavPlayScreenSetupWrap{
		rwm:          &sync.RWMutex{},
		ledsetup:     &NavLedSetup{NavlockId: Navlockid},
		ledsetupLast: &NavLedSetup{NavlockId: Navlockid, LedText: defaultLedText},
		isNotFirst:   false,
	}
	return tmp
}

func (lsw *NavPlayScreenSetupWrap) Infos() string {
	lsw.rwm.RLock()
	defer lsw.rwm.RUnlock()

	rs, _ := cm.Json.Marshal(lsw.ledsetupLast)
	return string(rs)
}

func (lsw *NavPlayScreenSetupWrap) LoadFromDB(db *gorm.DB, l *zap.Logger) (notSame bool, err error) {
	lsw.rwm.Lock()
	defer lsw.rwm.Unlock()

	if db == nil || l == nil {
		return false, fmt.Errorf("db or l cannot be nil")
	}

	result := db.Where("navLockId = ?", lsw.ledsetup.NavlockId).Last(lsw.ledsetup)
	if result.Error != nil {
		l.Sugar().Errorf(`get last from  db table %s error: %+v`, lsw.ledsetup.TableName(), result.Error)
		return false, result.Error
	}

	if result.RowsAffected > 0 {
		//防止gorm 取到最后一个记录 不执行where 条件判断
		if lsw.ledsetup.NavlockId != lsw.ledsetupLast.NavlockId {
			lsw.ledsetup.NavlockId = lsw.ledsetupLast.NavlockId
		} else {
			//如果设置有值的话 就交换信息
			if len(lsw.ledsetup.LedText) > 0 {
				// 如果文本不同
				if lsw.ledsetupLast.LedText != lsw.ledsetup.LedText {
					notSame = true
				}
				tmp := lsw.ledsetupLast
				lsw.ledsetupLast = lsw.ledsetup
				lsw.ledsetup = tmp
			}
		}
	}

	return notSame, nil
}

func (lsw *NavPlayScreenSetupWrap) GetSetup() string {
	lsw.rwm.RLock()
	defer lsw.rwm.RUnlock()

	return lsw.ledsetupLast.LedText
}
