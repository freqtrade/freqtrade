"""
Integration tests for the Config REST API.

Tests: list templates, load template, validate config.
"""

from __future__ import annotations

import pytest


class TestListTemplates:

    def test_list_templates(self, client, mock_data_service):
        resp = client.get("/api/config/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_templates_empty(self, client, mock_data_service):
        mock_data_service.get_config_templates.return_value = []
        resp = client.get("/api/config/templates")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetTemplate:

    def test_load_template(self, client, mock_data_service):
        resp = client.get("/api/config/templates/config_main")
        assert resp.status_code == 200
        data = resp.json()
        assert "genetic_algorithm" in data

    def test_template_not_found(self, client, mock_data_service):
        mock_data_service.load_config_template.return_value = None
        resp = client.get("/api/config/templates/nonexistent")
        assert resp.status_code == 404


class TestValidateConfig:

    def test_valid_config(self, client):
        config = {
            "genetic_algorithm": {
                "population_size": 20,
                "generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "elite_size": 2,
            },
            "backtesting": {
                "pairs": ["BTC/USDT"],
                "timerange": "20250101-20250301",
            },
        }
        resp = client.post("/api/config/validate", json=config)
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["warnings"] == []

    def test_missing_ga_section(self, client):
        resp = client.post("/api/config/validate", json={
            "backtesting": {"pairs": ["BTC/USDT"]},
        })
        data = resp.json()
        assert data["valid"] is False
        assert any("genetic_algorithm" in e for e in data["errors"])

    def test_missing_backtesting_section(self, client):
        resp = client.post("/api/config/validate", json={
            "genetic_algorithm": {"population_size": 20, "generations": 50},
        })
        data = resp.json()
        assert data["valid"] is False
        assert any("backtesting" in e for e in data["errors"])

    def test_invalid_population_size(self, client):
        config = {
            "genetic_algorithm": {"population_size": 1, "generations": 50},
            "backtesting": {"pairs": ["BTC/USDT"]},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is False
        assert any("population_size" in e for e in data["errors"])

    def test_invalid_mutation_rate(self, client):
        config = {
            "genetic_algorithm": {
                "population_size": 20,
                "generations": 50,
                "mutation_rate": 1.5,
            },
            "backtesting": {"pairs": ["BTC/USDT"]},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is False
        assert any("mutation_rate" in e for e in data["errors"])

    def test_elite_size_too_large(self, client):
        config = {
            "genetic_algorithm": {
                "population_size": 10,
                "generations": 50,
                "elite_size": 10,
            },
            "backtesting": {"pairs": ["BTC/USDT"]},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is False
        assert any("elite_size" in e for e in data["errors"])

    def test_empty_pairs(self, client):
        config = {
            "genetic_algorithm": {"population_size": 20, "generations": 50},
            "backtesting": {"pairs": []},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is False
        assert any("pairs" in e for e in data["errors"])

    def test_missing_timerange_warning(self, client):
        config = {
            "genetic_algorithm": {"population_size": 20, "generations": 50},
            "backtesting": {"pairs": ["BTC/USDT"]},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is True
        assert any("timerange" in w for w in data["warnings"])

    def test_fitness_weights_warning(self, client):
        config = {
            "genetic_algorithm": {"population_size": 20, "generations": 50},
            "backtesting": {"pairs": ["BTC/USDT"], "timerange": "20250101-"},
            "fitness_weights": {"profit": 0.5, "sharpe": 0.2},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is True
        assert any("fitness_weights" in w for w in data["warnings"])

    def test_fitness_weights_sum_one_no_warning(self, client):
        config = {
            "genetic_algorithm": {"population_size": 20, "generations": 50},
            "backtesting": {"pairs": ["BTC/USDT"], "timerange": "20250101-"},
            "fitness_weights": {"profit": 0.6, "sharpe": 0.4},
        }
        resp = client.post("/api/config/validate", json=config)
        data = resp.json()
        assert data["valid"] is True
        assert not any("fitness_weights" in w for w in data["warnings"])
