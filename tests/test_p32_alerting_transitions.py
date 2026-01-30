import pytest
import time
from unittest.mock import patch, MagicMock
from adapters.ccxt_shim import alerts


@pytest.fixture
def alert_manager():
    # Reset singleton
    alerts.AlertManager._instance = None
    return alerts.AlertManager.get_instance()


def test_alert_trigger(alert_manager):
    with patch("adapters.ccxt_shim.alerts.logger") as mock_logger:
        alert_manager.alert("TEST_CAT", "Test Message", "HIGH")

        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        assert "[ALERT:HIGH]" in args[0]
        assert "[TEST_CAT]" in args[0]


def test_suppression(alert_manager):
    with patch("adapters.ccxt_shim.alerts.logger") as mock_logger:
        # First alert
        alert_manager.alert("SUPPRESS_CAT", "Msg 1", "HIGH")
        assert mock_logger.error.call_count == 1

        # Second alert immediately (should be suppressed)
        alert_manager.alert("SUPPRESS_CAT", "Msg 2", "HIGH")
        assert mock_logger.error.call_count == 1

        # Wait for suppression window expiration (mock time)
        with patch("time.time", return_value=time.time() + 61):
            alert_manager.alert("SUPPRESS_CAT", "Msg 3", "HIGH")
            assert mock_logger.error.call_count == 2
