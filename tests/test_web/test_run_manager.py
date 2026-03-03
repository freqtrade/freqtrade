"""
Tests for RunManager — lifecycle management without spawning real subprocesses.

Tests: _on_event state sync, list_runs, stop/pause/resume/inject logic,
_reap_finished, RunHandle.to_summary().
"""

from __future__ import annotations

import multiprocessing as mp
import time
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.web.event_bus import Event, EventType, reset_event_bus
from genetic_algorithm.web.models.run import RunStatus
from genetic_algorithm.web.run_manager import RunHandle, RunManager


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def manager():
    """Create a RunManager with a fresh EventBus."""
    reset_event_bus()
    mgr = RunManager()
    yield mgr
    reset_event_bus()


def _make_handle(
    run_id: str = "run_test",
    status: RunStatus = RunStatus.RUNNING,
    **overrides,
) -> RunHandle:
    """Create a RunHandle for testing without real subprocess."""
    defaults = dict(
        run_id=run_id,
        status=status,
        config={"genetic_algorithm": {"population_size": 10, "generations": 50},
                "backtesting": {"pairs": ["BTC/USDT"]}},
        config_name="test",
        stop_event=mp.Event(),
        pause_event=mp.Event(),
        injection_queue=mp.Queue(maxsize=10),
        started_at=time.time() - 100,
        total_generations=50,
    )
    defaults.update(overrides)
    return RunHandle(**defaults)


# ── RunHandle Tests ───────────────────────────────────────────


class TestRunHandle:

    def test_to_summary_basic(self):
        h = _make_handle()
        s = h.to_summary()
        assert s.run_id == "run_test"
        assert s.status == RunStatus.RUNNING
        assert s.population_size == 10
        assert s.pairs == ["BTC/USDT"]
        assert s.elapsed_seconds is not None
        assert s.elapsed_seconds > 0

    def test_to_summary_finished(self):
        h = _make_handle(
            status=RunStatus.COMPLETED,
            started_at=1000.0,
            finished_at=1500.0,
        )
        s = h.to_summary()
        assert s.elapsed_seconds == pytest.approx(500.0)
        assert s.status == RunStatus.COMPLETED

    def test_to_summary_no_start_time(self):
        h = _make_handle(started_at=None)
        s = h.to_summary()
        assert s.elapsed_seconds is None


# ── RunManager._on_event Tests ────────────────────────────────


class TestOnEvent:

    def test_generation_end_updates_state(self, manager):
        handle = _make_handle(run_id="r1")
        manager._runs["r1"] = handle

        event = Event(
            type=EventType.GENERATION_END,
            run_id="r1",
            data={
                "generation": 10,
                "best_fitness": 0.9,
                "best_individual": {"id": "ind_best", "metrics": {"profit": 15.0}},
                "_stats": {"avg_fitness": 0.6},
            },
        )
        manager._on_event(event)

        assert handle.current_generation == 10
        assert handle.best_fitness == 0.9
        assert handle.best_individual_id == "ind_best"
        assert handle.best_profit == 15.0
        assert len(handle.generation_stats) == 1

    def test_new_best_updates_state(self, manager):
        handle = _make_handle(run_id="r1")
        manager._runs["r1"] = handle

        event = Event(
            type=EventType.NEW_BEST,
            run_id="r1",
            data={
                "individual": {
                    "id": "super_ind",
                    "fitness": 0.95,
                    "metrics": {"profit": 20.0},
                },
            },
        )
        manager._on_event(event)

        assert handle.best_individual_id == "super_ind"
        assert handle.best_fitness == 0.95
        assert handle.best_profit == 20.0

    def test_evolution_complete_marks_finished(self, manager):
        handle = _make_handle(run_id="r1")
        manager._runs["r1"] = handle

        event = Event(type=EventType.EVOLUTION_COMPLETE, run_id="r1", data={})
        manager._on_event(event)

        assert handle.status == RunStatus.COMPLETED
        assert handle.finished_at is not None

    def test_unknown_run_ignored(self, manager):
        # Should not raise for an event with unknown run_id
        event = Event(type=EventType.GENERATION_END, run_id="no_such", data={})
        manager._on_event(event)  # no crash


# ── Stop / Pause / Resume / Inject Tests ─────────────────────


class TestStopRun:

    def test_stop_running(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.RUNNING)
        manager._runs["r1"] = handle

        assert manager.stop_run("r1") is True
        assert handle.status == RunStatus.STOPPING
        assert handle.stop_event.is_set()

    def test_stop_paused(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.PAUSED)
        handle.pause_event.set()
        manager._runs["r1"] = handle

        assert manager.stop_run("r1") is True
        assert handle.status == RunStatus.STOPPING
        # Pause should be cleared so loop can check stop
        assert not handle.pause_event.is_set()

    def test_stop_already_stopped(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.COMPLETED)
        manager._runs["r1"] = handle

        assert manager.stop_run("r1") is False

    def test_stop_nonexistent(self, manager):
        assert manager.stop_run("nope") is False


class TestPauseRun:

    def test_pause_running(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.RUNNING)
        manager._runs["r1"] = handle

        assert manager.pause_run("r1") is True
        assert handle.status == RunStatus.PAUSED
        assert handle.pause_event.is_set()

    def test_pause_already_paused(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.PAUSED)
        manager._runs["r1"] = handle

        assert manager.pause_run("r1") is False

    def test_pause_nonexistent(self, manager):
        assert manager.pause_run("nope") is False


class TestResumeRun:

    def test_resume_paused(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.PAUSED)
        handle.pause_event.set()
        manager._runs["r1"] = handle

        assert manager.resume_run("r1") is True
        assert handle.status == RunStatus.RUNNING
        assert not handle.pause_event.is_set()

    def test_resume_running(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.RUNNING)
        manager._runs["r1"] = handle

        assert manager.resume_run("r1") is False


class TestInjectStrategy:

    def test_inject_into_running(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.RUNNING)
        manager._runs["r1"] = handle

        gene = {"individual_id": "custom_001", "indicators": []}
        assert manager.inject_strategy("r1", gene) is True
        # Check the message was enqueued (mp.Queue needs a short wait)
        msg = handle.injection_queue.get(timeout=2)
        assert msg["individual_id"] == "custom_001"

    def test_inject_into_stopped(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.COMPLETED)
        manager._runs["r1"] = handle

        assert manager.inject_strategy("r1", {}) is False


class TestSaveCheckpoint:

    def test_checkpoint_running(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.RUNNING)
        manager._runs["r1"] = handle

        assert manager.save_checkpoint("r1") is True
        msg = handle.injection_queue.get(timeout=2)
        assert msg["_command"] == "checkpoint"

    def test_checkpoint_stopped(self, manager):
        handle = _make_handle(run_id="r1", status=RunStatus.COMPLETED)
        manager._runs["r1"] = handle

        assert manager.save_checkpoint("r1") is False


class TestListRuns:

    def test_list_runs(self, manager):
        manager._runs["r1"] = _make_handle(run_id="r1")
        manager._runs["r2"] = _make_handle(run_id="r2", status=RunStatus.COMPLETED)

        summaries = manager.list_runs()
        ids = [s.run_id for s in summaries]
        assert "r1" in ids
        assert "r2" in ids

    def test_list_runs_empty(self, manager):
        assert manager.list_runs() == []

    def test_get_run(self, manager):
        handle = _make_handle(run_id="r1")
        manager._runs["r1"] = handle

        assert manager.get_run("r1") is handle
        assert manager.get_run("nope") is None

    def test_get_run_ids(self, manager):
        manager._runs["a"] = _make_handle(run_id="a")
        manager._runs["b"] = _make_handle(run_id="b")

        ids = manager.get_run_ids()
        assert set(ids) == {"a", "b"}
