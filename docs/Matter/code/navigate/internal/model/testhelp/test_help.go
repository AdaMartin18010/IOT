package model

import (
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
	"testing"

	md "navigate/internal/model"
)

var errFunc = func(t *testing.T, message string) {
	t.Errorf(message)
}

func logFailure(t *testing.T, message string) {
	_, file, line, _ := runtime.Caller(2)
	s := strings.Split(file, "/")
	errFunc(t, fmt.Sprintf("%v:%v: %v", s[len(s)-1], line, message))
}

// Assert checks if a and b are equal, emits a t.Errorf if they are not equal.
func Assert(t *testing.T, a interface{}, b interface{}) {
	if !reflect.DeepEqual(a, b) {
		logFailure(t, fmt.Sprintf("%v - \"%v\", should equal,  %v - \"%v\"",
			a, reflect.TypeOf(a), b, reflect.TypeOf(b)))
	}
}

// Refute checks if a and b are equal, emits a t.Errorf if they are equal.
func Refute(t *testing.T, a interface{}, b interface{}) {
	if reflect.DeepEqual(a, b) {
		logFailure(t, fmt.Sprintf("%v - \"%v\", should not equal,  %v - \"%v\"",
			a, reflect.TypeOf(a), b, reflect.TypeOf(b)))
	}
}

func ExecCommand(command string, args ...string) *exec.Cmd {
	cs := []string{"-test.run=TestHelperProcess", "--", command}
	cs = append(cs, args...)
	cmd := exec.Command(os.Args[0], cs...)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	return cmd
}

type NullReadWriteCloser struct{}

func (NullReadWriteCloser) Write(p []byte) (int, error) {
	return len(p), nil
}

func (NullReadWriteCloser) Read(b []byte) (int, error) {
	return len(b), nil
}

func (NullReadWriteCloser) Close() error {
	return nil
}

type testDriver struct {
	name       string
	pin        string
	connection md.Connection
	md.Commander
}

var testDriverStart = func() (err error) { return }
var testDriverHalt = func() (err error) { return }

func (t *testDriver) Start() (err error)        { return testDriverStart() }
func (t *testDriver) Halt() (err error)         { return testDriverHalt() }
func (t *testDriver) Name() string              { return t.name }
func (t *testDriver) SetName(n string)          { t.name = n }
func (t *testDriver) Pin() string               { return t.pin }
func (t *testDriver) Connection() md.Connection { return t.connection }

func NewTestDriver(adaptor *testAdaptor, name string, pin string) *testDriver {
	t := &testDriver{
		name:       name,
		connection: adaptor,
		pin:        pin,
		Commander:  md.NewCommander(),
	}

	t.AddCommand("DriverCommand", func(params map[string]interface{}) interface{} { return nil })

	return t
}

type testAdaptor struct {
	name string
	port string
}

var testAdaptorConnect = func() (err error) { return }
var testAdaptorFinalize = func() (err error) { return }

func (t *testAdaptor) Finalize() (err error) { return testAdaptorFinalize() }
func (t *testAdaptor) Connect() (err error)  { return testAdaptorConnect() }
func (t *testAdaptor) Name() string          { return t.name }
func (t *testAdaptor) SetName(n string)      { t.name = n }
func (t *testAdaptor) Port() string          { return t.port }

func NewTestAdaptor(name string, port string) *testAdaptor {
	return &testAdaptor{
		name: name,
		port: port,
	}
}

// func newTestComponent(name string) *Component {
// 	adaptor1 := newTestAdaptor("Connection1", "/dev/null")
// 	adaptor2 := newTestAdaptor("Connection2", "/dev/null")
// 	adaptor3 := newTestAdaptor("", "/dev/null")
// 	driver1 := newTestDriver(adaptor1, "Device1", "0")
// 	driver2 := newTestDriver(adaptor2, "Device2", "2")
// 	driver3 := newTestDriver(adaptor3, "", "1")
// 	work := func() {}
// 	r := NewComponent(name,
// 		[]Connection{adaptor1, adaptor2, adaptor3},
// 		[]Device{driver1, driver2, driver3},
// 		work,
// 	)
// 	r.AddCommand("ComponentCommand", func(params map[string]interface{}) interface{} { return nil })
// 	r.trap = func(c chan os.Signal) {
// 		c <- os.Interrupt
// 	}

// 	return r
// }
