import pytest
from unittest.mock import MagicMock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.order_router import OrderRouter


def test_track_and_assert_modify_quota_limit():
    router = OrderRouter(MagicMock())
    order_id = "ORD-123"

    # 3 modifications allowed
    router.track_and_assert_modify(order_id, 1000.0)
    router.track_and_assert_modify(order_id, 1100.0)
    router.track_and_assert_modify(order_id, 1200.0)

    # 4th should block
    with pytest.raises(OperationalException, match=r"order_router_block:mod_quota"):
        router.track_and_assert_modify(order_id, 1300.0)


def test_track_and_assert_modify_state_isolation():
    router = OrderRouter(MagicMock())

    # ORD-A: 3 mods
    router.track_and_assert_modify("ORD-A", 1000.0)
    router.track_and_assert_modify("ORD-A", 1100.0)
    router.track_and_assert_modify("ORD-A", 1200.0)

    # ORD-B: Should be fresh (0 mods)
    router.track_and_assert_modify("ORD-B", 1000.0)  # OK
