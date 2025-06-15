package eventer

import (
	"sync"
)

type eventChan chan *Event

type eventer struct {
	// map of valid Event names
	eventnames map[string]string

	// new events get put in to the event channel
	in eventChan

	// map of out channels used by subscribers
	outs map[eventChan]eventChan

	// mutex to protect the eventChannel map
	*sync.Mutex
}

const eventChanBufferSize = 10

// NewEventer returns a new Eventer.
func NewEventer() Eventer {
	evtr := &eventer{
		eventnames: make(map[string]string),
		in:         make(eventChan, eventChanBufferSize),
		outs:       make(map[eventChan]eventChan),
		Mutex:      &sync.Mutex{},
	}

	// goroutine to cascade "in" events to all "out" event channels
	go func() {
		// for {
		// 	select {
		// 	case evt, ok := <-evtr.in:
		// 		if !ok {
		// 			return
		// 		}
		// 		evtr.eventsMutex.Lock()
		// 		for _, out := range evtr.outs {
		// 			out <- evt
		// 		}
		// 		evtr.eventsMutex.Unlock()
		// 	}
		// }

		for evt := range evtr.in {
			evtr.Lock()
			for _, out := range evtr.outs {
				out <- evt
			}
			evtr.Unlock()
		}

	}()

	return evtr
}

// Events returns the map of valid Event names.
func (e *eventer) Events() map[string]string {
	return e.eventnames
}

// Event returns an Event string from map of valid Event names.
// Mostly used to validate that an Event name is valid.
func (e *eventer) Event(name string) string {
	return e.eventnames[name]
}

// AddEvent registers a new Event name.
func (e *eventer) AddEvent(name string) {
	e.eventnames[name] = name
}

// DeleteEvent removes a previously registered Event name.
func (e *eventer) DeleteEvent(name string) {
	delete(e.eventnames, name)
}

// Publish new events to anyone that is subscribed
func (e *eventer) Publish(name string, data any) {
	evt := NewEvent(name, data)
	e.in <- evt
}

// Subscribe to any events from this eventer
func (e *eventer) Subscribe() eventChan {
	e.Lock()
	defer e.Unlock()
	out := make(eventChan, eventChanBufferSize)
	e.outs[out] = out
	return out
}

// Unsubscribe from the event channel
func (e *eventer) Unsubscribe(events eventChan) {
	e.Lock()
	defer e.Unlock()
	delete(e.outs, events)
}

// On executes the event handler f when e is Published to.
func (e *eventer) On(n string, f func(s any)) (err error) {
	out := e.Subscribe()
	go func() {
		// for {
		// 	select {
		// 	case evt, _ := <-out:
		// 		if evt.Name == n {
		// 			f(evt.Data)
		// 		}
		// 	}
		// }

		for evt := range out {
			if evt.Name == n {
				f(evt.Data)
			}
		}

	}()

	return
}

// Once is similar to On except that it only executes f one time.
func (e *eventer) Once(n string, f func(s any)) (err error) {
	out := e.Subscribe()
	go func() {
	ProcessEvents:
		for evt := range out {
			if evt.Name == n {
				f(evt.Data)
				e.Unsubscribe(out)
				break ProcessEvents
			}
		}
	}()

	return
}
