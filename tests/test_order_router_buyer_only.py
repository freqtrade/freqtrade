import pytest
from unittest.mock import MagicMock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.order_router import OrderRouter


def test_assert_buyer_only_buy_allowed():
    router = OrderRouter(MagicMock())
    # Entry Buy - Assert no exception
    router.assert_buyer_only("RELIANCE/INR", "buy", None)
    router.assert_buyer_only("RELIANCE/INR", "BUY", None)


def test_assert_buyer_only_sell_blocked_no_callback():
    router = OrderRouter(MagicMock())
    # Sell without callback -> BLOCK
    with pytest.raises(OperationalException, match=r"order_router_block:buyer_only"):
        router.assert_buyer_only("RELIANCE/INR", "sell", None)


def test_assert_buyer_only_sell_blocked_no_position():
    router = OrderRouter(MagicMock())
    # Mock callback returns False (No position)
    mock_cb = MagicMock(return_value=False)

    with pytest.raises(OperationalException, match=r"order_router_block:buyer_only"):
        router.assert_buyer_only("RELIANCE/INR", "sell", mock_cb)
    mock_cb.assert_called_with("RELIANCE/INR")


def test_assert_buyer_only_sell_allowed_with_position():
    router = OrderRouter(MagicMock())
    # Mock callback returns True (Position exists)
    mock_cb = MagicMock(return_value=True)

    # Should not raise
    router.assert_buyer_only("RELIANCE/INR", "sell", mock_cb)
    mock_cb.assert_called_with("RELIANCE/INR")
