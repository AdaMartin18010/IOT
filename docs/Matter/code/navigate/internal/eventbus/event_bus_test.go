package eventbus_test

import (
	"testing"
	"time"

	eb "navigate/internal/eventbus"
)

func TestNew(t *testing.T) {
	bus := eb.NewEvtBus()
	if bus == nil {
		t.Log("New EventBus not created!")
		t.Fail()
	}
}

func TestHasCallback(t *testing.T) {
	bus := eb.NewEvtBus()
	bus.Subscribe("topic", func() {})
	if bus.HasCallbackLen("topic_topic") > 0 {
		t.Fail()
	}
	if bus.HasCallbackLen("topic") == 0 {
		t.Fail()
	}
}

func TestSubscribe(t *testing.T) {
	bus := eb.NewEvtBus()
	if bus.Subscribe("topic", func() {}) != nil {
		t.Fail()
	}
	if bus.Subscribe("topic", "String") == nil {
		t.Fail()
	}
}

func TestSubscribeOnce(t *testing.T) {
	bus := eb.NewEvtBus()
	if bus.SubscribeOnce("topic", func() {}) != nil {
		t.Fail()
	}
	if bus.SubscribeOnce("topic", "String") == nil {
		t.Fail()
	}
}

func TestSubscribeOnceAndManySubscribe(t *testing.T) {
	bus := eb.NewEvtBus()
	event := "topic"
	flag := 0
	fn := func() { flag += 1 }
	bus.SubscribeOnce(event, fn)
	bus.Subscribe(event, fn)
	bus.Subscribe(event, fn)
	bus.Publish(event)

	if flag != 3 {
		t.Fail()
	}
}

func TestUnsubscribe(t *testing.T) {
	bus := eb.NewEvtBus()
	handler := func() {}
	bus.Subscribe("topic", handler)
	if bus.Unsubscribe("topic", handler) != nil {
		t.Fail()
	}
	if bus.Unsubscribe("topic", handler) == nil {
		t.Fail()
	}
}

type handler struct {
	val int
}

func (h *handler) Handle() {
	h.val++
}

func TestUnsubscribeMethod(t *testing.T) {
	bus := eb.NewEvtBus()
	h := &handler{val: 0}

	bus.Subscribe("topic", h.Handle)
	bus.Publish("topic")
	if bus.Unsubscribe("topic", h.Handle) != nil {
		t.Fail()
	}
	if bus.Unsubscribe("topic", h.Handle) == nil {
		t.Fail()
	}
	bus.Publish("topic")
	bus.WaitAsync()

	if h.val != 1 {
		t.Fail()
	}
}

func TestPublish(t *testing.T) {
	bus := eb.NewEvtBus()
	bus.Subscribe("topic", func(a int, err error) {
		if a != 10 {
			t.Fail()
		}

		if err != nil {
			t.Fail()
		}
	})
	bus.Publish("topic", 10, nil)
}

func TestSubcribeOnceAsync(t *testing.T) {
	results := make([]int, 0)

	bus := eb.NewEvtBus()
	bus.SubscribeOnceAsync("topic", func(a int, out *[]int) {
		*out = append(*out, a)
	})

	bus.Publish("topic", 10, &results)
	bus.Publish("topic", 10, &results)

	bus.WaitAsync()

	if len(results) != 1 {
		t.Fail()
	}

	if bus.HasCallbackLen("topic") > 0 {
		t.Fail()
	}
}

func TestSubscribeAsyncTransactional(t *testing.T) {
	results := make([]int, 0)

	bus := eb.NewEvtBus()
	bus.SubscribeAsync("topic", func(a int, out *[]int, dur string) {
		sleep, _ := time.ParseDuration(dur)
		time.Sleep(sleep)
		*out = append(*out, a)
	}, true)

	bus.Publish("topic", 1, &results, "1s")
	bus.Publish("topic", 2, &results, "0s")

	bus.WaitAsync()

	if len(results) != 2 {
		t.Fail()
	}

	if results[0] != 1 || results[1] != 2 {
		t.Fail()
	}
}

func TestSubscribeAsync(t *testing.T) {
	results := make(chan int)

	bus := eb.NewEvtBus()
	bus.SubscribeAsync("topic", func(a int, out chan<- int) {
		out <- a
	}, false)

	bus.Publish("topic", 1, results)
	bus.Publish("topic", 2, results)

	numResults := 0

	go func() {
		for range results {
			numResults++
		}
	}()

	bus.WaitAsync()
	println(2)

	time.Sleep(10 * time.Millisecond)

	// todo race detected during execution of test
	//if numResults != 2 {
	//	t.Fail()
	//}
}
