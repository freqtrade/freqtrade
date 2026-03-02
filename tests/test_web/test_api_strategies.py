"""
Integration tests for the Strategies REST API.

Tests: get strategy, get strategy code, hall of fame.
"""

from __future__ import annotations

import pytest


class TestGetStrategy:

    def test_get_strategy(self, client, mock_data_service):
        resp = client.get("/api/runs/test_run_001/strategies/strat_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "strat_001"
        assert data["run_id"] == "test_run_001"
        assert data["fitness"] == 0.85

    def test_strategy_not_found(self, client, mock_data_service):
        mock_data_service.get_strategy.return_value = None
        resp = client.get("/api/runs/test_run_001/strategies/no_such_id")
        assert resp.status_code == 404


class TestGetStrategyCode:

    def test_get_code(self, client, mock_data_service):
        resp = client.get("/api/runs/test_run_001/strategies/strat_001/code")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_id"] == "strat_001"
        assert "class MyStrategy" in data["code"]

    def test_code_not_found(self, client, mock_data_service):
        mock_data_service.get_strategy_code.return_value = None
        resp = client.get("/api/runs/test_run_001/strategies/strat_001/code")
        assert resp.status_code == 404


class TestHallOfFame:

    def test_hall_of_fame(self, client, mock_data_service):
        resp = client.get("/api/hall-of-fame")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["id"] == "hof_000"

    def test_hall_of_fame_empty(self, client, mock_data_service):
        mock_data_service.get_hall_of_fame.return_value = []
        resp = client.get("/api/hall-of-fame")
        assert resp.status_code == 200
        assert resp.json() == []
