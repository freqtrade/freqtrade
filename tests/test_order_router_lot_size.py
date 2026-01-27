import pytest
from unittest.mock import MagicMock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.order_router import OrderRouter


def test_resolve_lot_size_defaults():
    mock_markets = MagicMock(return_value={})
    router = OrderRouter(mock_markets)
    assert router.resolve_lot_size("UNKNOWN/INR") == 1


def test_resolve_lot_size_found():
    mock_markets = MagicMock(return_value={"NIFTY/INR": {"lot": 50}, "RELIANCE/INR": {"lot": 1}})
    router = OrderRouter(mock_markets)
    assert router.resolve_lot_size("NIFTY/INR") == 50
    assert router.resolve_lot_size("RELIANCE/INR") == 1


def test_assert_lot_size_valid():
    router = OrderRouter(MagicMock())
    # 50 lot size, amount 50, 100, 150 OK
    router.assert_lot_size("X", 50.0, 50)
    router.assert_lot_size("X", 100.0, 50)
    router.assert_lot_size("X", 5000.0, 50)

    # 1 lot size, amount 1.0, 1.5 (wait, 1.5? No, int lot size implies amounts usually int for derivatives, but for spot 1.5 might be valid if step is < 1?
    # The prompt says: "amount must be multiple of lot_size".
    # For cash lot_size=1.
    router.assert_lot_size("X", 1.0, 1)
    router.assert_lot_size("X", 15.0, 1)


def test_assert_lot_size_blocks_invalid():
    router = OrderRouter(MagicMock())
    with pytest.raises(OperationalException, match=r"order_router_block:lot_size"):
        router.assert_lot_size("X", 51.0, 50)

    with pytest.raises(OperationalException, match=r"order_router_block:lot_size"):
        router.assert_lot_size("X", 1.5, 1)  # Assuming strict integer multiples for lot size >= 1
