import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from adapters.ccxt_shim import metrics_exporter

# Constants for testing
TEST_RUNTIME_DIR = Path("user_data/generated/runtime_test")
TEST_ALERTS_FILE = TEST_RUNTIME_DIR / "alerts.jsonl"
TEST_METRICS_JSON = TEST_RUNTIME_DIR / "metrics.json"
TEST_METRICS_PROM = TEST_RUNTIME_DIR / "metrics.prom"
TEST_PAPER_SQLITE = TEST_RUNTIME_DIR / "paper.sqlite"


@pytest.fixture
def mock_paths(monkeypatch, tmp_path):
    d = tmp_path / "runtime"
    d.mkdir()
    monkeypatch.setattr(metrics_exporter, "RUNTIME_DIR", d)
    monkeypatch.setattr(metrics_exporter, "ALERTS_FILE", d / "alerts.jsonl")
    monkeypatch.setattr(metrics_exporter, "METRICS_JSON", d / "metrics.json")
    monkeypatch.setattr(metrics_exporter, "METRICS_PROM", d / "metrics.prom")
    monkeypatch.setattr(metrics_exporter, "PAPER_LEDGER_SQLITE", d / "paper.sqlite")
    monkeypatch.setattr(metrics_exporter, "PAPER_LEDGER_CSV", d / "paper_trades.csv")
    return d


def test_collect_metrics_missing_files(mock_paths):
    """Test collection when no files exist."""
    with patch("adapters.ccxt_shim.health_snapshot.load", return_value={}):
        with patch("adapters.ccxt_shim.health_snapshot.get_p50_latency", return_value=0):
            metrics = metrics_exporter.collect_metrics()
            assert metrics["orders_paper_total"] == 0
            assert metrics["alerts_total"] == 0
            assert metrics["latency_ms_p50"]["fetch_ticker"] == 0


def test_collect_metrics_with_data(mock_paths):
    """Test collection with mocked data."""
    # Create fake alerts
    mock_paths.joinpath("alerts.jsonl").write_text("{}\n{}\n{}\n")  # 3 lines

    # Mock health snapshot
    health_data = {
        "counters": {"policy_blocks": 5, "degraded_failures": 2},
        "circuit_breaker": {"state": "open"},
    }

    with patch("adapters.ccxt_shim.health_snapshot.load", return_value=health_data):
        with patch(
            "adapters.ccxt_shim.health_snapshot.get_p50_latency",
            side_effect=lambda m: 100 if m == "fetch_ticker" else 0,
        ):
            # Mock paper ledger via sqlite
            with patch("sqlite3.connect") as mock_conn:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [10]  # 10 trades
                mock_conn.return_value.cursor.return_value = mock_cursor
                # We need the file to "exist" for the check
                mock_paths.joinpath("paper.sqlite").touch()

                metrics = metrics_exporter.collect_metrics()

                assert metrics["orders_paper_total"] == 10
                assert metrics["alerts_total"] == 3
                assert metrics["policy_blocks_total"] == 5
                assert metrics["circuit_open_total"] == 1
                assert metrics["latency_ms_p50"]["fetch_ticker"] == 100


def test_write_prom(mock_paths):
    """Test Prometheus format writing."""
    metrics = {
        "policy_blocks_total": 5,
        "degraded_failures_total": 0,
        "circuit_open_total": 1,
        "orders_paper_total": 10,
        "orders_live_blocked_total": 5,
        "alerts_total": 3,
        "latency_ms_p50": {"fetch_ticker": 100},
    }

    metrics_exporter.write_prom(metrics)

    prom_file = mock_paths / "metrics.prom"
    assert prom_file.exists()
    content = prom_file.read_text()

    assert "policy_blocks_total 5" in content
    assert 'latency_ms_p50{method="fetch_ticker"} 100' in content
    # Sorted check
    lines = content.strip().split("\n")
    assert lines == sorted(lines)


def test_full_export_flow(mock_paths):
    """Test the full export flow."""
    with patch("adapters.ccxt_shim.health_snapshot.load", return_value={}):
        with patch("adapters.ccxt_shim.health_snapshot.get_p50_latency", return_value=0):
            metrics_exporter.export_metrics()
            assert (mock_paths / "metrics.json").exists()
            assert (mock_paths / "metrics.prom").exists()
