package component

import (
	"os"
	"os/signal"
	"sync/atomic"

	md "navigate/internal/model"
	et "navigate/internal/model/eventer"

	multierror "github.com/hashicorp/go-multierror"
)

// JsonMaster is a JSON representation of a Component Master.
type JsonMaster struct {
	Components []*JsonComponent `json:"components"`
	Commands   []string         `json:"commands"`
}

// NewJsonMaster returns a JSONMaster given a Component Master.
func NewJsonMaster(cmpt *Master) *JsonMaster {
	jsonMaster := &JsonMaster{
		Components: []*JsonComponent{},
		Commands:   []string{},
	}

	for command := range cmpt.Commands() {
		jsonMaster.Commands = append(jsonMaster.Commands, command)
	}

	cmpt.components.Each(func(r *Component) {
		jsonMaster.Components = append(jsonMaster.Components, NewJsonComponent(r))
	})
	return jsonMaster
}

// Master is the main type of your component application and contains a collection of
// Components, API commands that apply to the Master, and Events that apply to the Master.
type Master struct {
	components *Components
	trap       func(chan os.Signal)
	AutoRun    bool
	running    atomic.Value
	md.Commander
	et.Eventer
}

// NewMaster returns a new component Master
func NewMaster() *Master {
	m := &Master{
		components: &Components{},
		trap: func(c chan os.Signal) {
			signal.Notify(c, os.Interrupt)
		},
		AutoRun:   true,
		Commander: md.NewCommander(),
		Eventer:   et.NewEventer(),
	}
	m.running.Store(false)
	return m
}

// Start calls the Start method on each component in its collection of components. On
// error, call Stop to ensure that all components are returned to a sane, stopped
// state.
func (m *Master) Start() (err error) {
	if rerr := m.components.Start(!m.AutoRun); rerr != nil {
		err = multierror.Append(err, rerr)
		return
	}

	m.running.Store(true)

	if m.AutoRun {
		c := make(chan os.Signal, 1)
		m.trap(c)

		// waiting for interrupt coming on the channel
		<-c

		// Stop calls the Stop method on each component in its collection of components.
		m.Stop()
	}

	return err
}

// Stop calls the Stop method on each component in its collection of components.
func (m *Master) Stop() (err error) {
	if rerr := m.components.Stop(); rerr != nil {
		err = multierror.Append(err, rerr)
	}

	m.running.Store(false)
	return
}

// Running returns if the Master is currently started or not
func (m *Master) Running() bool {
	return m.running.Load().(bool)
}

// Components returns all components associated with this component Master.
func (m *Master) Components() *Components {
	return m.components
}

// AddComponent adds a new component to the internal collection of components. Returns the
// added component
func (m *Master) AddComponent(r *Component) *Component {
	*m.components = append(*m.components, r)
	return r
}

// Component returns a component given name. Returns nil if the Component does not exist.
func (m *Master) Component(name string) *Component {
	for _, component := range *m.Components() {
		if component.Name == name {
			return component
		}
	}
	return nil
}
