package component_test

import (
	"fmt"
	"testing"
	"time"

	md "navigate/internal/model"
	cp "navigate/internal/model/component"
	th "navigate/internal/model/testhelp"
)

func TestConnectionEach(t *testing.T) {
	r := cp.NewComponent("ComponentTest")

	i := 0
	r.Connections().Each(func(conn md.Connection) {
		i++
	})

	if r.Connections().Len() != 0 {
		t.Fail()
	}

	if i != 0 {
		t.Fail()
	}
}

func TestComponentToJSON(t *testing.T) {
	r := cp.NewComponent("ComponentTest")
	r.AddCommand("test_function", func(params map[string]any) any {
		return nil
	})

	json := cp.NewJsonComponent(r)
	if len(json.Devices) != r.Devices().Len() {
		t.Fail()
	}

	if len(json.Commands) != len(r.Commands()) {
		t.Fail()
	}

}

func TestDevicesToJSON(t *testing.T) {
	r := cp.NewComponent("ComponentTest")
	json := cp.NewJsonComponent(r)

	if len(json.Devices) != r.Devices().Len() {
		t.Fail()
	}

	fmt.Printf(json.Devices[0].Name)

	fmt.Printf(json.Devices[0].Driver)

	fmt.Printf(json.Devices[0].Connection)

	if len(json.Devices[0].Commands) != 1 {
		t.Fail()
	}
}

func TestComponentStart(t *testing.T) {
	r := cp.NewComponent("Robot99")

	if r.Start() != nil {
		t.Fail()
	}

	if r.Stop() != nil {
		t.Fail()
	}

	if r.Running() != false {
		t.Fail()
	}
}

func TestComponentStartAutoRun(t *testing.T) {
	adaptor1 := th.NewTestAdaptor("Connection1", "/dev/null")
	driver1 := th.NewTestDriver(adaptor1, "Device1", "0")

	//work := func() {}
	r := cp.NewComponent("autorun",
		[]md.Connection{adaptor1},
		[]md.Device{driver1},
		//work,
	)

	go func() {
		th.Assert(t, r.Start(), nil)
	}()

	time.Sleep(10 * time.Millisecond)
	th.Assert(t, r.Running(), true)

	// stop it
	th.Assert(t, r.Stop(), nil)
	th.Assert(t, r.Running(), false)
}
