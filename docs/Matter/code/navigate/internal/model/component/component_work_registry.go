package component

import (
	"context"
	"sync"
	"time"

	"github.com/gofrs/uuid"
)

// ComponentWorkRegister contains all the work units registered on a ComponentWork
type ComponentWorkRegister struct {
	sync.RWMutex

	R map[string]*ComponentWork
}

// Get returns the ComponentWork specified by the provided ID.
// To delete something from the registry, it's
// necessary to call its context.CancelFunc,
// which will perform a goroutine-safe delete on the underlying map.
func (cwr *ComponentWorkRegister) Get(id uuid.UUID) *ComponentWork {
	cwr.Lock()
	defer cwr.Unlock()
	return cwr.R[id.String()]
}

// Delete returns the ComponentWork specified by the provided ID
func (cwr *ComponentWorkRegister) Delete(id uuid.UUID) {
	cwr.Lock()
	defer cwr.Unlock()
	delete(cwr.R, id.String())
}

// registerAfter creates a new unit of ComponentWork and sets up its context/cancellation
func (cwr *ComponentWorkRegister) registerAfter(ctx context.Context, d time.Duration, f func()) *ComponentWork {
	cwr.Lock()
	defer cwr.Unlock()

	id, _ := uuid.NewV4()
	rw := &ComponentWork{
		Id:       id,
		Kind:     AfterWorkKind,
		Function: f,
		duration: d,
	}

	rw.Ctx, rw.Cancel = context.WithCancel(ctx)
	cwr.R[id.String()] = rw
	return rw
}

// registerEvery creates a new unit of ComponentWork and sets up its context/cancellation
func (cwr *ComponentWorkRegister) registerEvery(ctx context.Context, d time.Duration, f func()) *ComponentWork {
	cwr.Lock()
	defer cwr.Unlock()

	id, _ := uuid.NewV4()
	rw := &ComponentWork{
		Id:       id,
		Kind:     EveryWorkKind,
		Function: f,
		duration: d,
		ticker:   time.NewTicker(d),
	}

	rw.Ctx, rw.Cancel = context.WithCancel(ctx)

	cwr.R[id.String()] = rw
	return rw
}
