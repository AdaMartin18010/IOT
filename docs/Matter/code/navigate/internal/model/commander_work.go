package model

type commander struct {
	commands map[string]func(map[string]any) any
}

// NewCommander returns a new Commander.
func NewCommander() Commander {
	return &commander{
		commands: make(map[string]func(map[string]any) any),
	}
}

// Command returns the command interface whene passed a valid command name
func (c *commander) Command(name string) (command func(map[string]any) any) {
	command = c.commands[name]
	return
}

// Commands returns the entire map of valid commands
func (c *commander) Commands() map[string]func(map[string]any) any {
	return c.commands
}

// AddCommand adds a new command, when passed a command name and the command interface.
func (c *commander) AddCommand(name string, command func(map[string]any) any) {
	c.commands[name] = command
}
