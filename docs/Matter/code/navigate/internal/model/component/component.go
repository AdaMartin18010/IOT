package component

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"

	md "navigate/internal/model"
	cn "navigate/internal/model/connection"
	dv "navigate/internal/model/device"
	et "navigate/internal/model/eventer"

	multierror "github.com/hashicorp/go-multierror"
)

// JsonComponent a JSON representation of a Component.
type JsonComponent struct {
	Name        string               `json:"name"`
	Commands    []string             `json:"commands"`
	Connections []*md.JsonConnection `json:"connections"`
	Devices     []*md.JsonDevice     `json:"devices"`
}

// NewJsonComponent returns a JsonComponent given a Component.
func NewJsonComponent(compt *Component) *JsonComponent {
	jsonComponent := &JsonComponent{
		Name:        compt.Name,
		Commands:    []string{},
		Connections: []*md.JsonConnection{},
		Devices:     []*md.JsonDevice{},
	}

	for command := range compt.Commands() {
		jsonComponent.Commands = append(jsonComponent.Commands, command)
	}

	compt.Devices().Each(func(device md.Device) {
		jsonDevice := md.NewJsonDevice(device)
		jsonComponent.Connections = append(jsonComponent.Connections, md.NewJSONConnection(compt.Connection(jsonDevice.Connection)))
		jsonComponent.Devices = append(jsonComponent.Devices, jsonDevice)
	})
	return jsonComponent
}

// Component is a named entity that manages a collection of connections and devices.
// It contains its own work routine and a collection of
// custom commands to control a Component remotely via the interface api.
type Component struct {
	Name    string
	trap    func(chan os.Signal)
	AutoRun bool
	running atomic.Value
	done    chan bool

	Work        func()
	connections *cn.Connections
	devices     *dv.Devices

	WorkRegister *ComponentWorkRegister

	WorkEveryWaitGroup *sync.WaitGroup
	WorkAfterWaitGroup *sync.WaitGroup
	md.Commander
	et.Eventer
}

// Components is a collection of Component
type Components []*Component

// Len returns the amount of Components in the collection.
func (cps *Components) Len() int {
	return len(*cps)
}

// Start calls the Start method of each Component in the collection
func (cps *Components) Start(args ...any) (err error) {
	autoRun := true
	if args[0] != nil {
		autoRun = args[0].(bool)
	}
	for _, cp := range *cps {
		if rerr := cp.Start(autoRun); rerr != nil {
			err = multierror.Append(err, rerr)
			return
		}
	}
	return
}

// Stop calls the Stop method of each Component in the collection
func (cps *Components) Stop() (err error) {
	for _, cp := range *cps {
		if rerr := cp.Stop(); rerr != nil {
			err = multierror.Append(err, rerr)
			return
		}
	}
	return
}

// Each enumerates through the Components and calls specified callback function.
func (cps *Components) Each(f func(*Component)) {
	for _, cp := range *cps {
		f(cp)
	}
}

// NewComponent returns a new Component. It supports the following optional params:
//
//	name:	string with the name of the Component. A name will be automatically generated if no name is supplied.
//	[]Connection: Connections which are automatically started and stopped with the Component
//	[]Device: Devices which are automatically started and stopped with the Component
//	func(): The work routine the Component will execute once all devices and connections have been initialized and started
func NewComponent(v ...interface{}) *Component {
	cp := &Component{
		Name:        fmt.Sprintf("%X", md.Rand(int(^uint(0)>>1))),
		connections: &cn.Connections{},
		devices:     &dv.Devices{},
		done:        make(chan bool, 1),
		trap: func(c chan os.Signal) {
			signal.Notify(c, os.Interrupt)
		},
		AutoRun:   true,
		Work:      nil,
		Eventer:   et.NewEventer(),
		Commander: md.NewCommander(),
	}

	for i := range v {
		switch v[i].(type) {
		case string:
			cp.Name = v[i].(string)
		case []md.Connection:
			log.Println("Initializing connections...")
			for _, connection := range v[i].([]md.Connection) {
				c := cp.AddConnection(connection)
				log.Println("Initializing connection", c.Name(), "...")
			}
		case []md.Device:
			log.Println("Initializing devices...")
			for _, device := range v[i].([]md.Device) {
				d := cp.AddDevice(device)
				log.Println("Initializing device", d.Name(), "...")
			}
		case func():
			cp.Work = v[i].(func())
		}
	}

	cp.WorkRegister = &ComponentWorkRegister{
		R: make(map[string]*ComponentWork),
	}
	cp.WorkAfterWaitGroup = &sync.WaitGroup{}
	cp.WorkEveryWaitGroup = &sync.WaitGroup{}

	cp.running.Store(false)
	log.Println("Component", cp.Name, "initialized.")

	return cp
}

// Start a Component's Connections, Devices, and work.
func (cp *Component) Start(params ...any) (err error) {
	if len(params) > 0 && params[0] != nil {
		cp.AutoRun = params[0].(bool)
	}
	log.Println("Starting Component", cp.Name, "...")
	if cerr := cp.Connections().Start(); cerr != nil {
		err = multierror.Append(err, cerr)
		log.Println(err)
		return
	}
	if derr := cp.Devices().Start(); derr != nil {
		err = multierror.Append(err, derr)
		log.Println(err)
		return
	}
	if cp.Work == nil {
		cp.Work = func() {}
	}

	log.Println("Starting work...")
	go func() {
		cp.Work()
		<-cp.done
	}()

	cp.running.Store(true)
	if cp.AutoRun {
		c := make(chan os.Signal, 1)
		cp.trap(c)

		// waiting for interrupt coming on the channel
		<-c

		// Stop calls the Stop method on itself, if we are "auto-running".
		cp.Stop()
	}

	return
}

// Stop stops a Component's connections and Devices
func (cp *Component) Stop() error {
	var result error
	log.Println("Stopping Component", cp.Name, "...")
	err := cp.Devices().Halt()

	if err != nil {
		result = multierror.Append(result, err)
	}
	err = cp.Connections().Finalize()
	if err != nil {
		result = multierror.Append(result, err)
	}

	cp.done <- true
	cp.running.Store(false)
	return result
}

// Running returns if the Component is currently started or not
func (cp *Component) Running() bool {
	return cp.running.Load().(bool)
}

// Devices returns all devices associated with this Component.
func (cp *Component) Devices() *dv.Devices {
	return cp.devices
}

// AddDevice adds a new Device to the Components collection of devices. Returns the
// added device.
func (cp *Component) AddDevice(d md.Device) md.Device {
	*cp.devices = append(*cp.Devices(), d)
	return d
}

// Device returns a device given a name. Returns nil if the Device does not exist.
func (cp *Component) Device(name string) md.Device {
	if cp == nil {
		return nil
	}

	for _, device := range *cp.devices {
		if device.Name() == name {
			return device
		}
	}
	return nil
}

// Connections returns all connections associated with this Component.
func (cp *Component) Connections() *cn.Connections {
	return cp.connections
}

// AddConnection adds a new connection to the Components collection of connections.
// Returns the added connection.
func (cp *Component) AddConnection(c md.Connection) md.Connection {
	*cp.connections = append(*cp.Connections(), c)
	return c
}

// Connection returns a connection given a name. Returns nil if the Connection
// does not exist.
func (cp *Component) Connection(name string) md.Connection {
	if cp == nil {
		return nil
	}
	for _, connection := range *cp.connections {
		if connection.Name() == name {
			return connection
		}
	}

	return nil
}
