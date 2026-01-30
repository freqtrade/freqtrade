import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from adapters.ccxt_shim import health_snapshot

# Test Data Path
TEST_HEALTH_FILE = Path("user_data/generated/runtime/test_health.json")


@pytest.fixture
def mock_health_file():
    # Patch the global HEALTH_FILE in the module
    original_path = health_snapshot.HEALTH_FILE
    health_snapshot.HEALTH_FILE = TEST_HEALTH_FILE

    # Reset singleton
    health_snapshot.HealthSnapshot._instance = None

    yield

    # Cleanup
    if TEST_HEALTH_FILE.exists():
        TEST_HEALTH_FILE.unlink()
    if TEST_HEALTH_FILE.with_suffix(".tmp").exists():
        TEST_HEALTH_FILE.with_suffix(".tmp").unlink()

    health_snapshot.HEALTH_FILE = original_path
    health_snapshot.HealthSnapshot._instance = None


def test_singleton_pattern(mock_health_file):
    s1 = health_snapshot.HealthSnapshot.get_instance()
    s2 = health_snapshot.HealthSnapshot.get_instance()
    assert s1 is s2


def test_persistence_atomic(mock_health_file):
    instance = health_snapshot.HealthSnapshot.get_instance()
    instance.update_mode(mock=True, paper=True, live=False)

    assert TEST_HEALTH_FILE.exists()

    data = instance.load()
    assert data["runtime"]["mode"]["breeze_mock"] is True
    assert data["runtime"]["mode"]["paper_trading"] is True


def test_counters_increment(mock_health_file):
    health_snapshot.update("policy_block")
    health_snapshot.update("degraded_failure")
    health_snapshot.update("policy_block")

    data = health_snapshot.load()
    assert data["counters"]["policy_blocks"] == 2
    assert data["counters"]["degraded_failures"] == 1


def test_error_recording(mock_health_file):
    health_snapshot.update("error", {"code": "E100", "message": "Test Error"})

    data = health_snapshot.load()
    assert data["last_error"]["code"] == "E100"
    assert data["last_error"]["message"] == "Test Error"


def test_corrupt_file_recovery(mock_health_file):
    # Write garbage
    with open(TEST_HEALTH_FILE, "w") as f:
        f.write("{ invalid json")

    # Should not crash
    data = health_snapshot.load()
    assert data == {}

    # Instance should still be usable
    health_snapshot.update("policy_block")
    data = health_snapshot.load()
    assert data["counters"]["policy_blocks"] == 1
