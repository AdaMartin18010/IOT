package eventer

// Event represents when something asynchronous happens in a Driver
// or Adaptor
type Event struct {
	Name string
	Data any
}

// NewEvent returns a new Event and its associated data.
func NewEvent(name string, data any) *Event {
	return &Event{Name: name, Data: data}
}

// Eventer is the interface which describes how a Driver or Adaptor
// handles events.
type Eventer interface {
	// Events returns the map of valid Event names.
	Events() (eventnames map[string]string)

	// Event returns an Event string from map of valid Event names.
	// Mostly used to validate that an Event name is valid.
	Event(name string) string

	// AddEvent registers a new Event name.
	AddEvent(name string)

	// DeleteEvent removes a previously registered Event name.
	DeleteEvent(name string)

	// Publish new events to any subscriber
	Publish(name string, data any)

	// Subscribe to events
	Subscribe() (events eventChan)

	// Unsubscribe from an event channel
	Unsubscribe(events eventChan)

	// Event handler
	On(name string, f func(s any)) (err error)

	// Event handler, only executes one time
	Once(name string, f func(s any)) (err error)
}
