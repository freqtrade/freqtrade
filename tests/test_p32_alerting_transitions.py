import pytest
from unittest.mock import patch
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
        # Control time via lambda
        current_time = [1000.0]

        def mock_now():
            return current_time[0]

        # Re-init manager with mock clock
        # We need to poke the singleton or use get_instance with args if supported (we added support)
        # But get_instance protects singleton.
        # Strategy: Force reset singleton
        alerts.AlertManager._instance = None
        am = alerts.AlertManager.get_instance(now_fn=mock_now)

        # First alert
        am.alert("SUPPRESS_CAT", "Msg 1", "HIGH")
        assert mock_logger.error.call_count == 1

        # Second alert immediately (should be suppressed)
        am.alert("SUPPRESS_CAT", "Msg 2", "HIGH")
        assert mock_logger.error.call_count == 1

        # Advance time > 60s
        current_time[0] += 61.0

        am.alert("SUPPRESS_CAT", "Msg 3", "HIGH")
        assert mock_logger.error.call_count == 2
