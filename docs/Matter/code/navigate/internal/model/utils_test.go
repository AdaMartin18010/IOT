package model_test

import (
	"strings"
	"testing"
	"time"

	model "navigate/internal/model"
)

func TestEvery(t *testing.T) {
	i := 0
	begin := time.Now().UnixNano()
	sem := make(chan int64, 1)
	model.Every(2*time.Millisecond, func() {
		i++
		if i == 2 {
			sem <- time.Now().UnixNano()
		}
	})
	end := <-sem
	if end-begin < 4000000 {
		t.Error("Test should have taken at least 4 milliseconds")
	}
}

func TestEveryWhenStopped(t *testing.T) {
	sem := make(chan bool)

	done := model.Every(50*time.Millisecond, func() {
		sem <- true
	})

	select {
	case <-sem:
		done.Stop()
	case <-time.After(60 * time.Millisecond):
		t.Errorf("Every was not called")
	}

	select {
	case <-time.After(60 * time.Millisecond):
	case <-sem:
		t.Error("Every should have stopped")
	}
}

func TestAfter(t *testing.T) {
	i := 0
	sem := make(chan bool)

	model.After(1*time.Millisecond, func() {
		i++
		sem <- true
	})

	select {
	case <-sem:
	case <-time.After(10 * time.Millisecond):
		t.Errorf("After was not called")
	}

	if i != 1 {
		t.Fail()

	}
}

func TestFromScale(t *testing.T) {
	if model.FromScale(5, 0, 10) != 0.5 {
		t.Fail()
	}
}

func TestToScale(t *testing.T) {
	if model.ToScale(500, 0, 10) != 10.0 {
		t.Fail()
	}

	if model.ToScale(-1, 0, 10) != 0.0 {
		t.Fail()
	}

	if model.ToScale(0.5, 0, 10) != 5.0 {
		t.Fail()
	}

}

func TestRescale(t *testing.T) {
	if model.Rescale(500, 0, 1000, 0, 10) != 5.0 {
		t.Fail()
	}
	if model.Rescale(-1.0, -1, 0, 490, 350) != 490.0 {
		t.Fail()
	}

}

func TestRand(t *testing.T) {
	a := model.Rand(10000)
	b := model.Rand(10000)
	if a == b {
		t.Errorf("%v should not equal %v", a, b)
	}
}

func TestDefaultName(t *testing.T) {
	name := model.DefaultName("tester")
	if strings.Contains(name, "tester") != true {
		t.Fail()
	}
}
