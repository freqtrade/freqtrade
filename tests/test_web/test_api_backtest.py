"""
Integration tests for the Backtest REST API.

Tests: start backtest, concurrent limit, get result, get trades.
Module-level state (_backtests, _active_count) requires careful cleanup.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from genetic_algorithm.web.routers import backtest as bt_mod


@pytest.fixture(autouse=True)
def _clean_backtest_state():
    """Reset module-level backtest state between tests."""
    bt_mod._backtests.clear()
    bt_mod._active_count = 0
    yield
    bt_mod._backtests.clear()
    bt_mod._active_count = 0


def _make_backtest_request() -> dict:
    return {
        "strategy_gene": {
            "indicators": [{"name": "EMA", "params": {"period": 20}}],
            "entry_conditions": [],
            "exit_conditions": [],
        },
        "timerange": "20250101-20250301",
        "pairs": ["BTC/USDT"],
        "stake_amount": 100.0,
    }


class TestStartBacktest:

    @patch.object(bt_mod, "_run_backtest")
    def test_start_backtest(self, mock_run, client):
        resp = client.post("/api/backtest", json=_make_backtest_request())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["backtest_id"].startswith("bt_")

    @patch.object(bt_mod, "_run_backtest")
    def test_concurrent_limit(self, mock_run, client, app):
        app.state.web_config.max_concurrent_backtests = 1
        # First request should succeed
        resp1 = client.post("/api/backtest", json=_make_backtest_request())
        assert resp1.status_code == 200
        # _active_count is incremented but _run_backtest is mocked so it
        # never decrements. Second request should hit 429.
        resp2 = client.post("/api/backtest", json=_make_backtest_request())
        assert resp2.status_code == 429


class TestGetBacktest:

    def test_get_backtest(self, client):
        from genetic_algorithm.web.models.strategy import BacktestResultModel

        # Manually insert a completed result
        result = BacktestResultModel(
            backtest_id="bt_test01",
            status="completed",
            progress=1.0,
            result={"total_profit": 50.0, "trades": []},
        )
        bt_mod._backtests["bt_test01"] = result

        resp = client.get("/api/backtest/bt_test01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backtest_id"] == "bt_test01"
        assert data["status"] == "completed"

    def test_get_backtest_not_found(self, client):
        resp = client.get("/api/backtest/bt_noexist")
        assert resp.status_code == 404


class TestGetBacktestTrades:

    def _insert_completed(self, n_trades: int = 10, pair: str = "BTC/USDT") -> str:
        from genetic_algorithm.web.models.strategy import BacktestResultModel

        bt_id = "bt_trades_test"
        trades = [
            {
                "pair": pair if i % 2 == 0 else "ETH/USDT",
                "profit_ratio": 0.01 * i,
                "profit_abs": 1.0 * i,
                "trade_duration": 60 * i,
            }
            for i in range(n_trades)
        ]
        result = BacktestResultModel(
            backtest_id=bt_id,
            status="completed",
            progress=1.0,
            result={"total_profit": 50.0, "trades": trades},
        )
        bt_mod._backtests[bt_id] = result
        return bt_id

    def test_get_trades(self, client):
        bt_id = self._insert_completed(10)
        resp = client.get(f"/api/backtest/{bt_id}/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["trades"]) == 10

    def test_trade_pagination(self, client):
        bt_id = self._insert_completed(20)
        resp = client.get(f"/api/backtest/{bt_id}/trades?offset=5&limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 20
        assert data["offset"] == 5
        assert data["limit"] == 3
        assert len(data["trades"]) == 3

    def test_trade_pair_filter(self, client):
        bt_id = self._insert_completed(10)
        resp = client.get(f"/api/backtest/{bt_id}/trades?pair=BTC/USDT")
        assert resp.status_code == 200
        data = resp.json()
        # Even indices are BTC/USDT (0,2,4,6,8) = 5 trades
        assert data["total"] == 5

    def test_trades_not_completed(self, client):
        from genetic_algorithm.web.models.strategy import BacktestResultModel

        bt_mod._backtests["bt_running"] = BacktestResultModel(
            backtest_id="bt_running", status="running"
        )
        resp = client.get("/api/backtest/bt_running/trades")
        assert resp.status_code == 400

    def test_trades_not_found(self, client):
        resp = client.get("/api/backtest/bt_noexist/trades")
        assert resp.status_code == 404
