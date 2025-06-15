package connection

import (
	"log"

	md "navigate/internal/model"

	multierror "github.com/hashicorp/go-multierror"
)

//var _ Connection = (*Connections)(nil)

// Connections represents a collection of Connection
type Connections []md.Connection

// Len returns connections length
func (c *Connections) Len() int {
	return len(*c)
}

// Each enumerates through the Connections and calls specified callback function.
func (c *Connections) Each(f func(md.Connection)) {
	for _, connection := range *c {
		f(connection)
	}
}

// Start calls Connect on each Connection in c
func (c *Connections) Start() (err error) {
	log.Println("Starting connections...")
	for _, connection := range *c {
		info := "Starting connection " + connection.Name()

		if porter, ok := connection.(md.Porter); ok {
			info = info + " on port " + porter.Port()
		}

		log.Println(info + "...")

		if cerr := connection.Connect(); cerr != nil {
			err = multierror.Append(err, cerr)
		}
	}
	return err
}

// Finalize calls Finalize on each Connection in c
func (c *Connections) Finalize() (err error) {
	for _, connection := range *c {
		if cerr := connection.Finalize(); cerr != nil {
			err = multierror.Append(err, cerr)
		}
	}
	return err
}
