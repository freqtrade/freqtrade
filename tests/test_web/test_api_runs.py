"""
Integration tests for the Runs REST API.

Uses FastAPI TestClient with mocked RunManager and DataService.
Tests: list, detail, config, start, stop, pause, resume, checkpoint, inject.
"""

from __future__ import annotations

import pytest

from tests.test_web.conftest import make_run_detail, make_run_summary
from genetic_algorithm.web.models.run import RunStatus


class TestListRuns:

    def test_list_runs(self, client, mock_data_service):
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["run_id"] == "test_run_001"

    def test_list_runs_empty(self, client, mock_data_service):
        mock_data_service.list_runs.return_value = []
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetRun:

    def test_get_run_detail(self, client, mock_data_service):
        resp = client.get("/api/runs/test_run_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "test_run_001"
        assert data["status"] == "running"
        assert "config" in data
        assert "generation_stats" in data

    def test_get_run_not_found(self, client, mock_data_service):
        mock_data_service.get_run_detail.return_value = None
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_run_config(self, client, mock_data_service):
        resp = client.get("/api/runs/test_run_001/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "genetic_algorithm" in data


class TestStartRun:

    def test_start_run(self, client, mock_run_manager):
        config = {
            "genetic_algorithm": {"population_size": 20, "generations": 50},
            "backtesting": {"pairs": ["BTC/USDT"]},
        }
        resp = client.post("/api/runs", json={"config": config})
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "new_run_001"
        mock_run_manager.start_run.assert_called_once()

    def test_start_run_with_custom_id(self, client, mock_run_manager):
        config = {"genetic_algorithm": {"population_size": 10, "generations": 5}}
        resp = client.post(
            "/api/runs",
            json={"config": config, "run_id": "my_custom_run"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_run_manager.start_run.call_args
        assert call_kwargs.kwargs.get("run_id") == "my_custom_run" or \
               (call_kwargs.args and len(call_kwargs.args) > 1)

    def test_start_run_failure(self, client, mock_run_manager):
        mock_run_manager.start_run.side_effect = RuntimeError("Config invalid")
        config = {"genetic_algorithm": {}}
        resp = client.post("/api/runs", json={"config": config})
        assert resp.status_code == 500


class TestStopRun:

    def test_stop_run(self, client, mock_run_manager):
        resp = client.post("/api/runs/test_run_001/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopping"
        mock_run_manager.stop_run.assert_called_once_with("test_run_001")

    def test_stop_run_not_running(self, client, mock_run_manager):
        mock_run_manager.stop_run.return_value = False
        resp = client.post("/api/runs/not_running/stop")
        assert resp.status_code == 400


class TestPauseResume:

    def test_pause_run(self, client, mock_run_manager):
        resp = client.post("/api/runs/test_run_001/pause")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "paused"

    def test_pause_not_running(self, client, mock_run_manager):
        mock_run_manager.pause_run.return_value = False
        resp = client.post("/api/runs/paused_run/pause")
        assert resp.status_code == 400

    def test_resume_run(self, client, mock_run_manager):
        resp = client.post("/api/runs/test_run_001/resume")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_resume_not_paused(self, client, mock_run_manager):
        mock_run_manager.resume_run.return_value = False
        resp = client.post("/api/runs/running_run/resume")
        assert resp.status_code == 400


class TestCheckpoint:

    def test_checkpoint(self, client, mock_run_manager):
        resp = client.post("/api/runs/test_run_001/checkpoint")
        assert resp.status_code == 200
        mock_run_manager.save_checkpoint.assert_called_once_with("test_run_001")

    def test_checkpoint_failure(self, client, mock_run_manager):
        mock_run_manager.save_checkpoint.return_value = False
        resp = client.post("/api/runs/not_running/checkpoint")
        assert resp.status_code == 400


class TestInjectStrategy:

    def test_inject_strategy(self, client, mock_run_manager):
        body = {
            "strategy_gene": {"indicators": [], "entry_conditions": []},
            "source_description": "Test injection",
        }
        resp = client.post("/api/runs/test_run_001/inject", json=body)
        assert resp.status_code == 200
        mock_run_manager.inject_strategy.assert_called_once()

    def test_inject_strategy_failure(self, client, mock_run_manager):
        mock_run_manager.inject_strategy.return_value = False
        body = {
            "strategy_gene": {"indicators": []},
        }
        resp = client.post("/api/runs/not_running/inject", json=body)
        assert resp.status_code == 400
