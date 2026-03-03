"""
Event Bus — In-process publish/subscribe system for GA events.

Decouples the evolution engine from its consumers (WebSocket monitor,
terminal monitor, data persistence, etc.).  Thread-safe and asyncio-aware.

Event types mirror the existing Monitor interface callbacks plus
additional lifecycle events for multi-run management.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, enum.Enum):
    """All event types emitted by the GA engine and run manager."""

    # ── Run lifecycle ──────────────────────────────────────────────
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_STOPPED = "run.stopped"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_ERROR = "run.error"

    # ── Generation lifecycle ───────────────────────────────────────
    GENERATION_START = "generation.start"
    GENERATION_END = "generation.end"

    # ── Phase tracking ─────────────────────────────────────────────
    PHASE_START = "phase.start"
    PHASE_END = "phase.end"

    # ── Evaluation progress ────────────────────────────────────────
    EVAL_PROGRESS = "eval.progress"

    # ── Notable events ─────────────────────────────────────────────
    NEW_BEST = "new_best"
    CONVERGENCE_WARNING = "convergence.warning"
    CHECKPOINT_SAVED = "checkpoint.saved"

    # ── Evolution complete ─────────────────────────────────────────
    EVOLUTION_COMPLETE = "evolution.complete"

    # ── Strategy injection ─────────────────────────────────────────
    STRATEGY_INJECTED = "strategy.injected"

    # ── System ─────────────────────────────────────────────────────
    LOG = "log"
    ERROR = "error"


@dataclass
class Event:
    """A single event in the bus."""

    type: EventType
    run_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# Type alias for subscriber callbacks
SyncCallback = Callable[[Event], None]
AsyncCallback = Callable[[Event], Any]  # Coroutine


class EventBus:
    """
    Thread-safe, asyncio-compatible event bus.

    Supports both synchronous and asynchronous subscribers.
    Async subscribers receive events via an :class:`asyncio.Queue`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # event_type → list of sync callbacks
        self._sync_subscribers: Dict[Optional[EventType], List[SyncCallback]] = {}
        # Each async consumer gets its own Queue (filled by publish)
        self._async_queues: List[asyncio.Queue] = []
        # Event history (bounded) for late-joining clients
        self._history: List[Event] = []
        self._max_history = 2000
        # Relay queue for subprocess → parent bridging (set via attach_relay_queue)
        self._relay_queue = None

    # ── Subscribe ──────────────────────────────────────────────────

    def subscribe(
        self,
        callback: SyncCallback,
        event_type: Optional[EventType] = None,
    ) -> None:
        """
        Register *callback* for *event_type* (or all events if ``None``).

        Callback is invoked synchronously in the publishing thread —
        keep it fast and non-blocking.
        """
        with self._lock:
            self._sync_subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(
        self,
        callback: SyncCallback,
        event_type: Optional[EventType] = None,
    ) -> None:
        with self._lock:
            subs = self._sync_subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def create_async_consumer(self) -> asyncio.Queue:
        """
        Create a new async Queue that will receive **all** future events.

        Used by WebSocket endpoints — each connected client gets one.
        Call :meth:`remove_async_consumer` on disconnect.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._async_queues.append(q)
        return q

    def remove_async_consumer(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._async_queues:
                self._async_queues.remove(q)

    # ── Publish ────────────────────────────────────────────────────

    def attach_relay_queue(self, relay_queue) -> None:
        """
        Attach an mp.Queue so that every published event is also serialised
        and pushed onto it.  Used in child processes to relay events to parent.
        """
        with self._lock:
            self._relay_queue = relay_queue

    def publish(self, event: Event) -> None:
        """Publish *event* to all subscribers (sync + async)."""
        with self._lock:
            # Append to history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Sync: specific-type subscribers
            for cb in self._sync_subscribers.get(event.type, []):
                try:
                    cb(event)
                except Exception:
                    logger.exception("Sync subscriber error for %s", event.type)

            # Sync: wildcard subscribers
            for cb in self._sync_subscribers.get(None, []):
                try:
                    cb(event)
                except Exception:
                    logger.exception("Wildcard subscriber error for %s", event.type)

            # Async queues — non-blocking put
            for q in self._async_queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop oldest, then retry
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:
                        pass

            # Relay to parent process (if in subprocess)
            relay = getattr(self, '_relay_queue', None)
            if relay is not None:
                try:
                    relay.put_nowait(event.to_dict())
                except Exception:
                    pass  # queue full or closed — drop

    # ── History ────────────────────────────────────────────────────

    def get_history(
        self,
        run_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 200,
    ) -> List[Event]:
        """Return recent events, optionally filtered."""
        with self._lock:
            events = self._history
            if run_id:
                events = [e for e in events if e.run_id == run_id]
            if event_type:
                events = [e for e in events if e.type == event_type]
            return events[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ── Module-level singleton ─────────────────────────────────────────
_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the global EventBus singleton (created on first call)."""
    global _global_bus
    if _global_bus is None:
        with _bus_lock:
            if _global_bus is None:
                _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global bus (useful in tests)."""
    global _global_bus
    with _bus_lock:
        _global_bus = None


# ── Subprocess ↔ Parent event relay ───────────────────────────────
# When the GA runs in a subprocess (via RunManager), its EventBus is a
# separate instance.  SubprocessEventRelay serialises events through an
# mp.Queue so they reach the parent's EventBus.

import multiprocessing as _mp


class SubprocessEventRelay:
    """
    Bridges events from a subprocess EventBus to the parent EventBus.

    Usage (parent side):
        relay = SubprocessEventRelay()
        relay.start()   # spawns a daemon thread that reads the relay queue
        # pass relay.queue to the child process

    Usage (child side):
        bus = get_event_bus()
        bus.attach_relay_queue(relay_queue)   # all publishes also go to queue
    """

    def __init__(self) -> None:
        self.queue: _mp.Queue = _mp.Queue(maxsize=5000)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the daemon thread that drains the relay queue into the parent bus."""
        self._running = True
        self._thread = threading.Thread(target=self._drain, daemon=True, name="event-relay")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Push sentinel to unblock .get()
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=3)

    def _drain(self) -> None:
        """Read events from the mp.Queue and re-publish on the parent EventBus."""
        parent_bus = get_event_bus()
        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    continue
                # Reconstruct Event from dict
                event = Event(
                    type=EventType(item["type"]),
                    run_id=item["run_id"],
                    data=item.get("data", {}),
                    timestamp=item.get("timestamp", time.time()),
                )
                parent_bus.publish(event)
            except (ValueError, KeyError):
                pass  # unknown event type or malformed item
            except Exception:
                if self._running:
                    logger.debug("Event relay error", exc_info=True)
