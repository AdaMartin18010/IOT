package component

import (
	"context"
	"fmt"
	"time"

	"github.com/gofrs/uuid"
)

const (
	EveryWorkKind = "every"
	AfterWorkKind = "after"
)

// ComponentWork and the ComponentWork registry represent units of executing computation
// managed at the model level. Unlike the utility functions model.After and model.Every,
// ComponentWork units require a context.Context, and can be cancelled externally by calling code.
//
// Usage:
//
//	someWork := component.Every(context.Background(), time.Second * 2, func(){
//		fmt.Println("Here I am doing work")
//	})
//
//	someWork.CallCancelFunc() // Cancel next tick and remove from work registry
//
// goroutines for Every and After are run on their own WaitGroups for synchronization:
//
//	someWork2 := component.Every(context.Background(), time.Second * 2, func(){
//		fmt.Println("Here I am doing more work")
//	})
//
//	somework2.CallCancelFunc()
//
//	// wait for both Every calls to finish
//	component.WorkEveryWaitGroup().Wait()
type ComponentWork struct {
	Id        uuid.UUID
	Kind      string
	tickCount int
	Ctx       context.Context
	Cancel    context.CancelFunc
	Function  func()
	ticker    *time.Ticker
	duration  time.Duration
}

// ID returns the UUID of the ComponentWork
func (cp *ComponentWork) ID() uuid.UUID {
	return cp.Id
}

// CancelFunc returns the context.CancelFunc used to cancel the work
func (cp *ComponentWork) CancelFunc() context.CancelFunc {
	return cp.Cancel
}

// CallCancelFunc calls the context.CancelFunc used to cancel the work
func (cp *ComponentWork) CallCancelFunc() {
	cp.Cancel()
}

// Ticker returns the time.Ticker used in an Every so that calling code can sync on the same channel
func (cp *ComponentWork) Ticker() *time.Ticker {
	if cp.Kind == AfterWorkKind {
		return nil
	}
	return cp.ticker
}

// TickCount returns the number of times the function successfully ran
func (cp *ComponentWork) TickCount() int {
	return cp.tickCount
}

// TickCount returns the number of times the function successfully ran
func (cp *ComponentWork) SetTickCount(tc int) {
	cp.tickCount = tc
}

// Duration returns the timeout until an After fires or the period of an Every
func (cp *ComponentWork) Duration() time.Duration {
	return cp.duration
}

// Duration returns the timeout until an After fires or the period of an Every
func (cp *ComponentWork) SetDuration(dt time.Duration) {
	cp.duration = dt
}

func (cp *ComponentWork) String() string {
	format := `ID: %s,Kind: %s,TickCount: %d`
	return fmt.Sprintf(format, cp.Id, cp.Kind, cp.tickCount)
}

// WorkRegistry returns the Robot's WorkRegistry
func (cp *Component) WorkRegistry() *ComponentWorkRegister {
	return cp.WorkRegister
}

// Every calls the given function for every tick of the provided duration.
func (cp *Component) Every(ctx context.Context, d time.Duration, f func()) *ComponentWork {
	rw := cp.WorkRegister.registerEvery(ctx, d, f)
	cp.WorkEveryWaitGroup.Add(1)
	go func() {
	EVERYWORK:
		for {
			select {
			case <-rw.Ctx.Done():
				cp.WorkRegister.Delete(rw.Id)
				rw.ticker.Stop()
				break EVERYWORK
			case <-rw.ticker.C:
				f()
				rw.tickCount++
			}
		}
		cp.WorkEveryWaitGroup.Done()
	}()
	return rw
}

// After calls the given function after the provided duration has elapsed
func (cp *Component) After(ctx context.Context, d time.Duration, f func()) *ComponentWork {
	rw := cp.WorkRegister.registerAfter(ctx, d, f)
	ch := time.After(d)
	cp.WorkAfterWaitGroup.Add(1)
	go func() {
	AFTERWORK:
		for {
			select {
			case <-rw.Ctx.Done():
				cp.WorkRegister.Delete(rw.Id)
				break AFTERWORK
			case <-ch:
				f()
			}
		}
		cp.WorkAfterWaitGroup.Done()
	}()
	return rw
}
