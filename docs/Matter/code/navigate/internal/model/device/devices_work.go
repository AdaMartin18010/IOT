package device

import (
	"log"

	md "navigate/internal/model"

	multierror "github.com/hashicorp/go-multierror"
)

//var _ Driver = (*Devices)(nil)

// Devices represents a collection of Device
type Devices []md.Device

// Len returns devices length
func (d *Devices) Len() int {
	return len(*d)
}

// Each enumerates through the Devices and calls specified callback function.
func (d *Devices) Each(f func(md.Device)) {
	for _, device := range *d {
		f(device)
	}
}

// Start calls Start on each Device in d
func (d *Devices) Start() (err error) {
	log.Println("Starting devices...")
	for _, device := range *d {
		info := "Starting device " + device.Name()

		if pinner, ok := device.(md.Pinner); ok {
			info = info + " on pin " + pinner.Pin()
		}

		log.Println(info + "...")
		if derr := device.Start(); derr != nil {
			err = multierror.Append(err, derr)
		}
	}
	return err
}

// Halt calls Halt on each Device in d
func (d *Devices) Halt() (err error) {
	for _, device := range *d {
		if derr := device.Halt(); derr != nil {
			err = multierror.Append(err, derr)
		}
	}
	return err
}
