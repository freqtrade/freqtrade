import pytest
from unittest.mock import MagicMock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.order_router import OrderRouter


def test_track_and_assert_modify_ladder_rate_limit():
    router = OrderRouter(MagicMock())
    order_id = "ORD-LADDER"

    # First mod
    router.track_and_assert_modify(order_id, 1000.0)

    # Second mod too soon (< 2s)
    with pytest.raises(OperationalException, match=r"order_router_block:mod_ladder"):
        router.track_and_assert_modify(order_id, 1001.5)

    # Second mod OK (> 2s)
    router.track_and_assert_modify(order_id, 1002.1)

    # Third mod too soon
    with pytest.raises(OperationalException, match=r"order_router_block:mod_ladder"):
        router.track_and_assert_modify(order_id, 1003.0)


def test_track_and_assert_modify_ladder_independent_orders():
    router = OrderRouter(MagicMock())

    router.track_and_assert_modify("ORD-1", 1000.0)

    # ORD-2 can modify immediately, independent rate limit
    router.track_and_assert_modify("ORD-2", 1000.1)
