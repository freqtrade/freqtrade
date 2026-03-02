"""
Tests for EventBus — publish/subscribe, history, async consumers, SubprocessEventRelay.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
import time

import pytest

from genetic_algorithm.web.event_bus import (
    Event,
    EventBus,
    EventType,
    SubprocessEventRelay,
    get_event_bus,
    reset_event_bus,
)


# ── Basic Pub/Sub ──────────────────────────────────────────────────


class TestEventBusPubSub:

    def test_publish_and_subscribe(self, event_bus: EventBus):
        received = []
        event_bus.subscribe(lambda e: received.append(e))
        event = Event(type=EventType.RUN_STARTED, run_id="r1", data={"foo": "bar"})
        event_bus.publish(event)

        assert len(received) == 1
        assert received[0].type == EventType.RUN_STARTED
        assert received[0].run_id == "r1"
        assert received[0].data["foo"] == "bar"

    def test_type_specific_subscription(self, event_bus: EventBus):
        started = []
        stopped = []
        event_bus.subscribe(lambda e: started.append(e), EventType.RUN_STARTED)
        event_bus.subscribe(lambda e: stopped.append(e), EventType.RUN_STOPPED)

        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.RUN_STOPPED, run_id="r1"))
        event_bus.publish(Event(type=EventType.NEW_BEST, run_id="r1"))

        assert len(started) == 1
        assert len(stopped) == 1

    def test_wildcard_receives_all(self, event_bus: EventBus):
        all_events = []
        event_bus.subscribe(lambda e: all_events.append(e))  # no type filter

        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))
        event_bus.publish(Event(type=EventType.NEW_BEST, run_id="r1"))

        assert len(all_events) == 3

    def test_unsubscribe(self, event_bus: EventBus):
        received = []
        cb = lambda e: received.append(e)
        event_bus.subscribe(cb)
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        assert len(received) == 1

        event_bus.unsubscribe(cb)
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        assert len(received) == 1  # no new events

    def test_subscriber_exception_does_not_break(self, event_bus: EventBus):
        """A failing subscriber should not prevent other subscribers from receiving events."""
        good_events = []

        def bad_cb(e):
            raise ValueError("boom")

        event_bus.subscribe(bad_cb)
        event_bus.subscribe(lambda e: good_events.append(e))

        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        assert len(good_events) == 1

    def test_multiple_subscribers_same_type(self, event_bus: EventBus):
        r1, r2 = [], []
        event_bus.subscribe(lambda e: r1.append(e), EventType.NEW_BEST)
        event_bus.subscribe(lambda e: r2.append(e), EventType.NEW_BEST)

        event_bus.publish(Event(type=EventType.NEW_BEST, run_id="r1"))
        assert len(r1) == 1
        assert len(r2) == 1


# ── History ────────────────────────────────────────────────────────


class TestEventBusHistory:

    def test_history_stores_events(self, event_bus: EventBus):
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.RUN_STOPPED, run_id="r2"))

        history = event_bus.get_history()
        assert len(history) == 2

    def test_history_limit(self, event_bus: EventBus):
        for i in range(10):
            event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))

        history = event_bus.get_history(limit=3)
        assert len(history) == 3

    def test_history_filter_by_run_id(self, event_bus: EventBus):
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r2"))
        event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))

        r1_history = event_bus.get_history(run_id="r1")
        assert len(r1_history) == 2
        assert all(e.run_id == "r1" for e in r1_history)

    def test_history_filter_by_event_type(self, event_bus: EventBus):
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))
        event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r2"))

        gen_history = event_bus.get_history(event_type=EventType.GENERATION_END)
        assert len(gen_history) == 2

    def test_clear_history(self, event_bus: EventBus):
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        assert len(event_bus.get_history()) == 1
        event_bus.clear_history()
        assert len(event_bus.get_history()) == 0

    def test_history_bounded(self, event_bus: EventBus):
        """History should not grow beyond _max_history."""
        event_bus._max_history = 50
        for i in range(100):
            event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))
        assert len(event_bus.get_history(limit=200)) <= 50


# ── Async Consumer ─────────────────────────────────────────────────


class TestAsyncConsumer:

    def test_create_and_remove_consumer(self, event_bus: EventBus):
        q = event_bus.create_async_consumer()
        assert q in event_bus._async_queues

        event_bus.remove_async_consumer(q)
        assert q not in event_bus._async_queues

    def test_async_consumer_receives_events(self, event_bus: EventBus):
        q = event_bus.create_async_consumer()
        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))

        event = q.get_nowait()
        assert event.type == EventType.RUN_STARTED

    def test_async_queue_full_drops_oldest(self, event_bus: EventBus):
        """When async queue is full, oldest event is dropped to make room."""
        q = asyncio.Queue(maxsize=2)
        event_bus._async_queues.append(q)

        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))
        event_bus.publish(Event(type=EventType.GENERATION_END, run_id="r1"))
        event_bus.publish(Event(type=EventType.NEW_BEST, run_id="r1"))

        # Queue has 2 items (oldest was dropped)
        assert q.qsize() == 2


# ── Event serialization ───────────────────────────────────────────


class TestEventSerialization:

    def test_event_to_dict(self):
        event = Event(
            type=EventType.NEW_BEST,
            run_id="r1",
            data={"fitness": 0.95},
            timestamp=1234567890.0,
        )
        d = event.to_dict()
        assert d["type"] == "new_best"
        assert d["run_id"] == "r1"
        assert d["data"]["fitness"] == 0.95
        assert d["timestamp"] == 1234567890.0


# ── Singleton ──────────────────────────────────────────────────────


class TestEventBusSingleton:

    def test_get_event_bus_returns_same_instance(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus(self):
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2


# ── EventType enum ─────────────────────────────────────────────────


class TestEventType:

    def test_all_event_types_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)

    def test_event_type_count(self):
        """Verify all expected event types exist."""
        assert len(EventType) >= 18

    def test_event_type_from_value(self):
        assert EventType("run.started") == EventType.RUN_STARTED
        assert EventType("generation.end") == EventType.GENERATION_END
        assert EventType("new_best") == EventType.NEW_BEST

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValueError):
            EventType("nonexistent.event")


# ── SubprocessEventRelay ──────────────────────────────────────────


class TestSubprocessEventRelay:

    def test_relay_start_stop(self):
        relay = SubprocessEventRelay()
        relay.start()
        assert relay._running is True
        assert relay._thread is not None
        assert relay._thread.is_alive()

        relay.stop()
        assert relay._running is False

    def test_relay_forwards_events(self):
        """Relay should forward dict events from the queue to the parent EventBus."""
        relay = SubprocessEventRelay()
        relay.start()

        parent_bus = get_event_bus()
        received = []
        parent_bus.subscribe(lambda e: received.append(e))

        # Simulate a child process pushing a serialized event
        relay.queue.put({
            "type": "run.started",
            "run_id": "child_run",
            "data": {"test": True},
            "timestamp": time.time(),
        })

        # Give the drain thread time to process
        time.sleep(0.5)
        relay.stop()

        assert len(received) == 1
        assert received[0].type == EventType.RUN_STARTED
        assert received[0].run_id == "child_run"

    def test_relay_skips_unknown_event_types(self):
        """Relay should gracefully skip events with unknown types."""
        relay = SubprocessEventRelay()
        relay.start()

        parent_bus = get_event_bus()
        received = []
        parent_bus.subscribe(lambda e: received.append(e))

        relay.queue.put({
            "type": "totally.unknown.event",
            "run_id": "r1",
            "data": {},
        })

        time.sleep(0.5)
        relay.stop()

        assert len(received) == 0  # unknown event was skipped

    def test_relay_skips_malformed_items(self):
        """Relay should skip items missing required keys."""
        relay = SubprocessEventRelay()
        relay.start()

        parent_bus = get_event_bus()
        received = []
        parent_bus.subscribe(lambda e: received.append(e))

        relay.queue.put({"bad": "data"})
        relay.queue.put("not_a_dict")
        # valid one
        relay.queue.put({
            "type": "new_best",
            "run_id": "r1",
            "data": {"fitness": 0.9},
        })

        time.sleep(0.5)
        relay.stop()

        # Only the valid event should have been forwarded
        assert len(received) == 1
        assert received[0].type == EventType.NEW_BEST

    def test_relay_thread_safety(self):
        """Multiple threads publishing to the relay queue should not crash."""
        relay = SubprocessEventRelay()
        relay.start()
        parent_bus = get_event_bus()
        received = []
        parent_bus.subscribe(lambda e: received.append(e))

        total_events = 30

        def publisher(n):
            for i in range(10):
                relay.queue.put({
                    "type": "generation.end",
                    "run_id": f"r_{n}",
                    "data": {"generation": i},
                })

        threads = [threading.Thread(target=publisher, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait until all events are drained (up to 5s)
        for _ in range(50):
            if len(received) >= total_events:
                break
            time.sleep(0.1)

        relay.stop()
        assert len(received) == total_events


# ── Relay queue on EventBus ────────────────────────────────────────


class TestRelayQueue:

    def test_attach_relay_queue(self, event_bus: EventBus):
        q = mp.Queue()
        event_bus.attach_relay_queue(q)

        event_bus.publish(Event(type=EventType.RUN_STARTED, run_id="r1"))

        item = q.get(timeout=1)
        assert item["type"] == "run.started"
        assert item["run_id"] == "r1"
