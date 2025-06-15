package model

import (
	"reflect"
)

// ************************* Component *********************
type Component interface {
	Id() string
	SetId(id string)
	TypeName() string

	Running() bool

	Add(cpt *Component) error
	Component(id string) *Component
	Components(typename string) []*Component

	Start() error
	Stop() error
	Finalize() error
}

// ************************ Driver ************************
// Driver is the interface that describes a driver in IOT
type Driver interface {
	// Name returns the label for the Driver
	Name() string
	// SetName sets the label for the Driver
	SetName(s string)
	// Start initiates the Driver
	Start() error
	// Halt terminates the Driver
	Halt() error
	// Connection returns the Connection associated with the Driver
	Connection() Connection
}

// Pinner is the interface that describes a driver's pin
type Pinner interface {
	Pin() string
}

// A Device is an instnace of a Driver
type Device Driver

// JsonDevice is a JSON representation of a Device.
type JsonDevice struct {
	Name       string   `json:"name"`
	Driver     string   `json:"driver"`
	Connection string   `json:"connection"`
	Commands   []string `json:"commands"`
}

// NewJsonDevice returns a JSONDevice given a Device.
func NewJsonDevice(device Device) *JsonDevice {
	jsonDevice := &JsonDevice{
		Name:       device.Name(),
		Driver:     reflect.TypeOf(device).String(),
		Commands:   []string{},
		Connection: "",
	}
	if device.Connection() != nil {
		jsonDevice.Connection = device.Connection().Name()
	}
	if commander, ok := device.(Commander); ok {
		for command := range commander.Commands() {
			jsonDevice.Commands = append(jsonDevice.Commands, command)
		}
	}
	return jsonDevice
}

// ************************ Adaptor ************************
// Adaptor is the interface that describes an adaptor in iot
type Adaptor interface {
	// Name returns the label for the Adaptor
	Name() string
	// SetName sets the label for the Adaptor
	SetName(n string)
	// Connect initiates the Adaptor
	Connect() error
	// Finalize terminates the Adaptor
	Finalize() error
}

// Porter is the interface that describes an adaptor's port
type Porter interface {
	Port() string
}

// A Connection is an instance of an Adaptor
type Connection Adaptor

// JsonConnection is a JSON representation of a Connection.
type JsonConnection struct {
	Name    string `json:"name"`
	Adaptor string `json:"adaptor"`
}

// NewJSONConnection returns a JSONConnection given a Connection.
func NewJSONConnection(connection Connection) *JsonConnection {
	return &JsonConnection{
		Name:    connection.Name(),
		Adaptor: reflect.TypeOf(connection).String(),
	}
}

// *************************** Commander ***************************
// Commander is the interface which describes the behaviour for a Driver or Adaptor
// which exposes API commands.
type Commander interface {
	// Command returns a command given a name. Returns nil if the command is not found.
	Command(string) (command func(map[string]any) any)
	// Commands returns a map of commands.
	Commands() (commands map[string]func(map[string]any) any)
	// AddCommand adds a command given a name.
	AddCommand(name string, command func(map[string]any) any)
}
