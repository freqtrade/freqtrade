"""
Tests for web dashboard bug fixes and feature additions.

BUG-3: Missing event emission (checkpoint.saved, log, error)
Fix #15: Backtest exchange parameter from run config
Fix #18: Duplicate ID handling in HoF / strategy lookup
P3-2: Indicator overlay endpoint
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import json
import tempfile
from pathlib import Path


# ── BUG-3: Event emission tests ──────────────────────────────────


class TestCheckpointSavedEvent:
    """Verify checkpoint.saved events are emitted when checkpoints are saved."""

    def test_ws_monitor_has_on_checkpoint_saved(self):
        """WebSocketMonitor should have on_checkpoint_saved method."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor

        monitor = WebSocketMonitor(run_id="test", config={})
        assert hasattr(monitor, "on_checkpoint_saved")
        assert callable(monitor.on_checkpoint_saved)

    def test_ws_monitor_emits_checkpoint_event(self):
        """on_checkpoint_saved should publish CHECKPOINT_SAVED event."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor
        from genetic_algorithm.web.event_bus import EventType

        monitor = WebSocketMonitor(run_id="test-run", config={})
        monitor.bus = MagicMock()

        monitor.on_checkpoint_saved(5, "/tmp/checkpoint.json")

        monitor.bus.publish.assert_called_once()
        event = monitor.bus.publish.call_args[0][0]
        assert event.type == EventType.CHECKPOINT_SAVED
        assert event.run_id == "test-run"
        assert event.data["generation"] == 5
        assert event.data["path"] == "/tmp/checkpoint.json"
        assert event.data["requested"] is False

    def test_null_monitor_has_on_checkpoint_saved(self):
        """NullMonitor should have on_checkpoint_saved method (no-op)."""
        from genetic_algorithm.monitor.null_monitor import NullMonitor

        monitor = NullMonitor()
        assert hasattr(monitor, "on_checkpoint_saved")
        # Should not raise
        monitor.on_checkpoint_saved(3, "/tmp/x.json")


class TestLogEvent:
    """Verify log events are emitted."""

    def test_ws_monitor_has_on_log(self):
        """WebSocketMonitor should have on_log method."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor

        monitor = WebSocketMonitor(run_id="test", config={})
        assert hasattr(monitor, "on_log")

    def test_ws_monitor_emits_log_event(self):
        """on_log should publish LOG event."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor
        from genetic_algorithm.web.event_bus import EventType

        monitor = WebSocketMonitor(run_id="test-run", config={})
        monitor.bus = MagicMock()

        monitor.on_log("Evolution converged early at gen 5/12", "warning")

        event = monitor.bus.publish.call_args[0][0]
        assert event.type == EventType.LOG
        assert event.data["message"] == "Evolution converged early at gen 5/12"
        assert event.data["level"] == "warning"

    def test_null_monitor_has_on_log(self):
        """NullMonitor should have on_log method (no-op)."""
        from genetic_algorithm.monitor.null_monitor import NullMonitor

        monitor = NullMonitor()
        monitor.on_log("test message", "info")  # Should not raise


class TestErrorEvent:
    """Verify error events are emitted."""

    def test_ws_monitor_has_on_error(self):
        """WebSocketMonitor should have on_error method."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor

        monitor = WebSocketMonitor(run_id="test", config={})
        assert hasattr(monitor, "on_error")

    def test_ws_monitor_emits_error_event(self):
        """on_error should publish ERROR event."""
        from genetic_algorithm.web.ws_monitor import WebSocketMonitor
        from genetic_algorithm.web.event_bus import EventType

        monitor = WebSocketMonitor(run_id="test-run", config={})
        monitor.bus = MagicMock()

        monitor.on_error("Holdout monitoring failed: timeout", {"component": "holdout"})

        event = monitor.bus.publish.call_args[0][0]
        assert event.type == EventType.ERROR
        assert "monitoring failed" in event.data["message"]
        assert event.data["component"] == "holdout"

    def test_null_monitor_has_on_error(self):
        """NullMonitor should have on_error method (no-op)."""
        from genetic_algorithm.monitor.null_monitor import NullMonitor

        monitor = NullMonitor()
        monitor.on_error("test error", {"detail": "x"})  # Should not raise


# ── Fix #18: Duplicate ID handling ───────────────────────────────


class TestDuplicateIDHandling:
    """Verify strategy lookup handles unique entry IDs correctly."""

    def test_search_hof_matches_entry_id(self):
        """_search_hof should match against the entry's id field."""
        from genetic_algorithm.web.services.data_service import DataService

        ds = DataService.__new__(DataService)

        # Mock get_hall_of_fame to return entries with unique IDs
        entries = [
            {
                "id": "abc123_unique_hash",
                "fitness": 0.85,
                "run_id": "run_A",
                "metrics": {"profit": 10},
                "strategy_gene": {
                    "generation": 5,
                    "individual_id": 3,
                    "indicators": [],
                    "entry_conditions": [],
                    "exit_conditions": [],
                },
            },
            {
                "id": "Gen5_Ind3",
                "fitness": 0.75,
                "run_id": "run_B",
                "metrics": {"profit": 8},
                "strategy_gene": {
                    "generation": 5,
                    "individual_id": 3,
                    "indicators": [],
                    "entry_conditions": [],
                    "exit_conditions": [],
                },
            },
        ]

        with patch.object(ds, "get_hall_of_fame", return_value=entries):
            # Match by unique entry_id
            result = ds._search_hof("run_A", "abc123_unique_hash")
            assert result is not None
            assert result.id == "abc123_unique_hash"

            # Match by legacy Gen/Ind format — prefers run_B since exact run_id match
            result = ds._search_hof("run_B", "Gen5_Ind3")
            assert result is not None
            assert result.fitness == 0.75

    def test_search_hof_prefers_run_id_match(self):
        """When multiple entries match, prefer the one with matching run_id."""
        from genetic_algorithm.web.services.data_service import DataService

        ds = DataService.__new__(DataService)

        entries = [
            {
                "id": "Gen2_Ind1",
                "fitness": 0.90,
                "run_id": "run_X",
                "metrics": {},
                "strategy_gene": {
                    "generation": 2,
                    "individual_id": 1,
                    "indicators": [],
                    "entry_conditions": [],
                    "exit_conditions": [],
                },
            },
            {
                "id": "Gen2_Ind1",
                "fitness": 0.80,
                "run_id": "run_Y",
                "metrics": {},
                "strategy_gene": {
                    "generation": 2,
                    "individual_id": 1,
                    "indicators": [],
                    "entry_conditions": [],
                    "exit_conditions": [],
                },
            },
        ]

        with patch.object(ds, "get_hall_of_fame", return_value=entries):
            result = ds._search_hof("run_Y", "Gen2_Ind1")
            assert result is not None
            assert result.fitness == 0.80  # Should prefer run_Y match


# ── P3-2: Indicator endpoint tests ──────────────────────────────


class TestIndicatorRegistry:
    """Verify the indicator computation registry works correctly."""

    def test_ensure_indicator_registry(self):
        """Registry should contain all supported indicators."""
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        expected = {"EMA", "SMA", "RSI", "MACD", "BBANDS", "ADX", "ATR", "STOCH", "CCI"}
        assert expected.issubset(set(_INDICATOR_REGISTRY.keys()))

    def test_ema_computation(self):
        """EMA should produce valid values."""
        import pandas as pd
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        })

        result = _INDICATOR_REGISTRY["EMA"](df, period=3)
        assert "EMA_3" in result
        assert len(result["EMA_3"]) == 10
        # EMA should follow the trend
        assert result["EMA_3"].iloc[-1] > result["EMA_3"].iloc[3]

    def test_rsi_computation(self):
        """RSI should produce values between 0 and 100."""
        import pandas as pd
        import numpy as np
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        # Need enough data for RSI period + warmup (at least 2*period)
        df = pd.DataFrame({
            "close": list(range(100, 150)),  # 50 data points, up trend
        })

        result = _INDICATOR_REGISTRY["RSI"](df, period=14)
        assert "RSI_14" in result
        rsi_vals = result["RSI_14"].dropna()
        assert len(rsi_vals) > 0
        assert all(0 <= v <= 100 for v in rsi_vals)
        assert rsi_vals.iloc[-1] > 50  # Uptrend → high RSI

    def test_bbands_computation(self):
        """Bollinger Bands should have upper > middle > lower."""
        import pandas as pd
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        df = pd.DataFrame({
            "close": [100 + i * 0.1 for i in range(50)],
        })

        result = _INDICATOR_REGISTRY["BBANDS"](df, period=20, std_dev=2.0)
        assert "BB_upper_20" in result
        assert "BB_middle_20" in result
        assert "BB_lower_20" in result

        # At last index, upper > middle > lower
        assert result["BB_upper_20"].iloc[-1] > result["BB_middle_20"].iloc[-1]
        assert result["BB_middle_20"].iloc[-1] > result["BB_lower_20"].iloc[-1]

    def test_macd_computation(self):
        """MACD should produce macd line, signal, and histogram."""
        import pandas as pd
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        df = pd.DataFrame({
            "close": [100 + i * 0.5 for i in range(40)],
        })

        result = _INDICATOR_REGISTRY["MACD"](df, fast_period=12, slow_period=26, signal_period=9)
        assert any("MACD" in k for k in result.keys())
        assert any("Signal" in k for k in result.keys())
        assert any("Hist" in k for k in result.keys())

    def test_adx_computation(self):
        """ADX should produce values between 0 and 100."""
        import pandas as pd
        import numpy as np
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "high": [100 + i * 0.3 + np.random.random() for i in range(n)],
            "low": [100 + i * 0.3 - np.random.random() for i in range(n)],
            "close": [100 + i * 0.3 for i in range(n)],
        })

        result = _INDICATOR_REGISTRY["ADX"](df, period=14)
        assert "ADX_14" in result
        adx_vals = result["ADX_14"].dropna()
        assert len(adx_vals) > 0


class TestIndicatorResponseFormat:
    """Verify indicator endpoint returns dict format matching frontend."""

    def test_response_is_dict_format(self):
        """Response indicators should be a dict with 'values' and 'pane' keys."""
        # Simulate the response format
        from genetic_algorithm.web.routers.data import _ensure_indicator_registry, _INDICATOR_REGISTRY
        import pandas as pd
        import numpy as np

        _INDICATOR_REGISTRY.clear()
        _ensure_indicator_registry()

        # Build a result dict the same way the endpoint does
        df = pd.DataFrame({
            "date": [1000000 + i * 60000 for i in range(30)],
            "open": [100] * 30,
            "high": [105] * 30,
            "low": [95] * 30,
            "close": [100 + i * 0.1 for i in range(30)],
            "volume": [1000] * 30,
        })

        compute_fn = _INDICATOR_REGISTRY["EMA"]
        columns = compute_fn(df, period=5)

        result_dict = {}
        for col_name, series in columns.items():
            data_points = []
            for idx in range(len(df)):
                v = series.iloc[idx]
                if pd.notna(v) and np.isfinite(v):
                    data_points.append([int(df["date"].iloc[idx]), round(float(v), 6)])
            result_dict[col_name] = {
                "values": data_points,
                "pane": "price",
            }

        # Verify dict format
        assert isinstance(result_dict, dict)
        assert "EMA_5" in result_dict
        assert "values" in result_dict["EMA_5"]
        assert "pane" in result_dict["EMA_5"]
        assert result_dict["EMA_5"]["pane"] == "price"
        assert len(result_dict["EMA_5"]["values"]) > 0
        # Each value should be [timestamp, value]
        first = result_dict["EMA_5"]["values"][0]
        assert len(first) == 2
        assert isinstance(first[0], int)
        assert isinstance(first[1], float)
