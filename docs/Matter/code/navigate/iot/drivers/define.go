package drivers

type Connecter interface {
	Connect() error
	Close() error
}

// Transporter specifies the transport layer.
type Transporter interface {
	Connecter
}
