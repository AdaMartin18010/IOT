//实现上 依赖golang的编译器reflect当前实现 后续变更会出现不兼容
//性能上 由于依赖reflect反射机制 带来运行时开销 2个数量级
//调试上 如果用错会导致非常底层的错误 不熟悉reflect机制的情况下很难定位和排错

package eventbus

// EvtBus is interface for global (subscribe, publish, control) bus behavior
type EvtBus interface {
	BusController
	BusSubscriber
	BusPublisher
}

// BusSubscriber defines subscription-related bus behavior
type BusSubscriber interface {
	Subscribe(topic string, fn any) error
	SubscribeAsync(topic string, fn any, transactional bool) error
	SubscribeOnce(topic string, fn any) error
	SubscribeOnceAsync(topic string, fn any) error
	Unsubscribe(topic string, handler any) error
}

// BusPublisher defines publishing-related bus behavior
type BusPublisher interface {
	Publish(topic string, args ...any)
}

// BusController defines bus control behavior (checking handler's presence, synchronization)
type BusController interface {
	HasCallbackLen(topic string) int
	WaitAsync()
}
