package component_test

import (
	"context"
	"testing"
	"time"

	cp "navigate/internal/model/component"

	"github.com/gofrs/uuid"
	"github.com/stretchr/testify/assert"
)

func TestComponentWork(t *testing.T) {
	id, _ := uuid.NewV4()

	rw := &cp.ComponentWork{
		Id:       id,
		Kind:     cp.EveryWorkKind,
		Function: func() {},
	}

	duration := time.Second * 1
	ctx, cancelFunc := context.WithCancel(context.Background())

	rw.Ctx = ctx
	rw.Cancel = cancelFunc
	rw.SetDuration(duration)

	t.Run("ID()", func(t *testing.T) {
		assert.Equal(t, rw.ID(), id)
	})

	t.Run("Ticker()", func(t *testing.T) {
		t.Skip()
	})

	t.Run("Duration()", func(t *testing.T) {
		assert.Equal(t, rw.Duration(), duration)
	})
}

func TestComponentWorkRegistry(t *testing.T) {
	cmpt := cp.NewComponent("testbot")

	rw := cmpt.Every(context.Background(), time.Millisecond*250, func() {
		_ = 1 + 1
	})

	t.Run("Get retrieves", func(t *testing.T) {
		assert.Equal(t, cmpt.WorkRegister.Get(rw.Id), rw)
	})

	t.Run("delete deletes", func(t *testing.T) {
		cmpt.WorkRegister.Delete(rw.Id)
		postDeleteKeys := collectStringKeysFromWorkRegistry(cmpt.WorkRegister)
		assert.NotContains(t, postDeleteKeys, rw.Id.String())
	})
}

func TestComponentAutomationFunctions(t *testing.T) {
	t.Run("Every with cancel", func(t *testing.T) {
		comp := cp.NewComponent("testbot")

		rw := comp.Every(context.Background(), time.Millisecond*10, func() {
			_ = 1 + 1 // perform mindless computation!
		})

		time.Sleep(time.Millisecond * 25)
		rw.CallCancelFunc()

		comp.WorkEveryWaitGroup.Wait()

		assert.Equal(t, 2, rw.TickCount())
		postDeleteKeys := collectStringKeysFromWorkRegistry(comp.WorkRegister)
		assert.NotContains(t, postDeleteKeys, rw.Id.String())
	})

	t.Run("After with cancel", func(t *testing.T) {
		comp := cp.NewComponent("testbot")

		rw := comp.After(context.Background(), time.Millisecond*10, func() {
			_ = 1 + 1 // perform mindless computation!
		})

		rw.CallCancelFunc()

		comp.WorkAfterWaitGroup.Wait()

		postDeleteKeys := collectStringKeysFromWorkRegistry(comp.WorkRegister)
		assert.NotContains(t, postDeleteKeys, rw.Id.String())
	})
}

func collectStringKeysFromWorkRegistry(cwr *cp.ComponentWorkRegister) []string {
	keys := make([]string, len(cwr.R))
	for k := range cwr.R {
		keys = append(keys, k)
	}
	return keys
}
