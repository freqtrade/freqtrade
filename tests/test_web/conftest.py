"""
Shared fixtures for web dashboard tests.

Provides a configured FastAPI TestClient, mock EventBus, mock DataService,
and sample data factories.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from genetic_algorithm.web.config import WebConfig
from genetic_algorithm.web.event_bus import Event, EventBus, EventType, reset_event_bus
from genetic_algorithm.web.models.generation import GenerationDetail, IndividualSummary
from genetic_algorithm.web.models.run import RunDetail, RunStatus, RunSummary
from genetic_algorithm.web.models.strategy import StrategyDetail
from genetic_algorithm.web.server import create_app


# ── Factories ──────────────────────────────────────────────────────


def make_run_summary(
    run_id: str = "test_run_001",
    status: RunStatus = RunStatus.RUNNING,
    **overrides: Any,
) -> RunSummary:
    defaults = dict(
        run_id=run_id,
        status=status,
        config_name="test_config",
        current_generation=5,
        total_generations=50,
        best_fitness=0.85,
        best_profit=12.5,
        population_size=20,
        started_at=time.time() - 300,
        elapsed_seconds=300.0,
        pairs=["BTC/USDT", "ETH/USDT"],
    )
    defaults.update(overrides)
    return RunSummary(**defaults)


def make_run_detail(run_id: str = "test_run_001", **overrides: Any) -> RunDetail:
    defaults = dict(
        run_id=run_id,
        status=RunStatus.RUNNING,
        config_name="test_config",
        current_generation=5,
        total_generations=50,
        best_fitness=0.85,
        best_profit=12.5,
        population_size=20,
        started_at=time.time() - 300,
        elapsed_seconds=300.0,
        pairs=["BTC/USDT"],
        config={"genetic_algorithm": {"population_size": 20, "generations": 50}},
        generation_stats=[],
        best_individual_id="ind_abc123",
        mode="single_objective",
    )
    defaults.update(overrides)
    return RunDetail(**defaults)


def make_individual(
    id: str = "ind_001",
    fitness: float = 0.75,
    **overrides: Any,
) -> IndividualSummary:
    defaults = dict(
        id=id,
        fitness=fitness,
        raw_fitness=fitness,
        rank=1,
        crowding_distance=1.0,
        evaluated=True,
        metrics={"profit": 5.0, "sharpe_ratio": 1.2},
        profit=5.0,
        sharpe_ratio=1.2,
        win_rate=0.6,
        num_trades=25,
        max_drawdown=-0.08,
        profit_factor=1.5,
        complexity=5,
        indicators=["EMA_20", "RSI_14"],
    )
    defaults.update(overrides)
    return IndividualSummary(**defaults)


def make_generation_detail(
    run_id: str = "test_run_001",
    generation: int = 3,
    n_individuals: int = 5,
) -> GenerationDetail:
    individuals = [
        make_individual(id=f"ind_{i:03d}", fitness=0.5 + i * 0.1)
        for i in range(n_individuals)
    ]
    return GenerationDetail(
        run_id=run_id,
        generation=generation,
        individuals=individuals,
        stats={"best_fitness": 0.9, "avg_fitness": 0.65},
    )


def make_strategy_detail(
    id: str = "strat_001",
    run_id: str = "test_run_001",
    **overrides: Any,
) -> StrategyDetail:
    defaults = dict(
        id=id,
        run_id=run_id,
        generation=5,
        fitness=0.85,
        raw_fitness=0.82,
        metrics={"profit": 10.0, "sharpe_ratio": 1.5, "num_trades": 30},
        gene=None,
        quality=None,
        parent_ids=[],
        mutations=[],
        walk_forward_windows=None,
        monte_carlo=None,
    )
    defaults.update(overrides)
    return StrategyDetail(**defaults)


def make_hof_entries(n: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"hof_{i:03d}",
            "fitness": 0.9 - i * 0.05,
            "profit": 15.0 - i * 3.0,
            "sharpe_ratio": 2.0 - i * 0.3,
            "num_trades": 40 - i * 5,
            "max_drawdown": -0.05 - i * 0.02,
            "win_rate": 0.65 - i * 0.05,
            "complexity": 5 + i,
            "timeframe": "5m",
            "added_at": "2026-02-28T12:00:00",
            "config_name": "config_main",
            "run_id": f"run_{i:03d}",
        }
        for i in range(n)
    ]


def make_config_templates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "config_main",
            "path": "/path/to/config_main.yaml",
            "pairs": ["BTC/USDT"],
            "generations": 50,
            "population_size": 20,
        },
        {
            "name": "config_alt",
            "path": "/path/to/config_alt.yaml",
            "pairs": ["ETH/USDT", "BNB/USDT"],
            "generations": 100,
            "population_size": 50,
        },
    ]


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_bus():
    """Ensure each test gets a fresh EventBus."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def event_bus() -> EventBus:
    """Return a fresh EventBus instance (not the global singleton)."""
    return EventBus()


@pytest.fixture
def mock_data_service():
    """Return a MagicMock DataService with sensible defaults."""
    ds = MagicMock()
    ds.list_runs.return_value = [make_run_summary()]
    ds.get_run_detail.return_value = make_run_detail()
    ds.get_generation.return_value = make_generation_detail()
    ds.get_strategy.return_value = make_strategy_detail()
    ds.get_strategy_code.return_value = "# Generated strategy\nclass MyStrategy:\n    pass"
    ds.get_hall_of_fame.return_value = make_hof_entries()
    ds.get_config_templates.return_value = make_config_templates()
    ds.load_config_template.return_value = {
        "genetic_algorithm": {"population_size": 20, "generations": 50},
        "backtesting": {"pairs": ["BTC/USDT"], "timerange": "20250101-20250301"},
    }
    return ds


@pytest.fixture
def mock_run_manager():
    """Return a MagicMock RunManager."""
    from genetic_algorithm.web.run_manager import RunHandle

    mgr = MagicMock()
    mgr.list_runs.return_value = [make_run_summary()]
    mgr.get_run.return_value = None  # default: run not found in manager
    mgr.start_run.return_value = MagicMock(
        spec=RunHandle,
        run_id="new_run_001",
        status=RunStatus.RUNNING,
    )
    mgr.start_run.return_value.to_summary.return_value = make_run_summary(
        run_id="new_run_001", status=RunStatus.RUNNING
    )
    mgr.stop_run.return_value = True
    mgr.pause_run.return_value = True
    mgr.resume_run.return_value = True
    mgr.inject_strategy.return_value = True
    mgr.save_checkpoint.return_value = True
    return mgr


@pytest.fixture
def app(mock_data_service, mock_run_manager):
    """Create a FastAPI app with mocked services."""
    test_app = create_app(
        web_config=WebConfig(host="127.0.0.1", port=8501),
        run_manager=mock_run_manager,
    )
    # Override data_service with our mock
    test_app.state.data_service = mock_data_service
    test_app.state.run_manager = mock_run_manager
    return test_app


@pytest.fixture
def client(app) -> TestClient:
    """Return a FastAPI TestClient."""
    return TestClient(app)
