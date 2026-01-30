"""
Tests for Order Slicing
"""

import pytest
from unittest.mock import MagicMock
from adapters.ccxt_shim.order_router import OrderRouter


def test_order_slicing_uniform():
    # Mock market resolution for lot size
    mock_markets = MagicMock(return_value={"REL": {"lot": 50}})
    router = OrderRouter(mock_markets)
    config = {"max_child_orders": 4}

    # 1000 qty, 4 orders -> 250 each.
    chunks = router.slice_order("REL", 1000, config)

    assert len(chunks) == 4
    assert chunks == [250, 250, 250, 250]
    assert sum(chunks) == 1000


def test_order_slicing_remainder():
    mock_markets = MagicMock(return_value={"REL": {"lot": 10}})
    router = OrderRouter(mock_markets)
    config = {"max_child_orders": 3}

    # 100 qty, 3 orders. 100//3 = 33.3.
    # Base chunk = 33. Round down multiple of 10 -> 30.
    # Chunk = 30.
    # Orders: 30, 30, 40 (remainder)

    chunks = router.slice_order("REL", 100, config)
    assert chunks == [30, 30, 40]
    assert sum(chunks) == 100


def test_order_slicing_small():
    mock_markets = MagicMock(return_value={"REL": {"lot": 50}})
    router = OrderRouter(mock_markets)
    config = {"max_child_orders": 4}

    # 60 qty. Max chunk 15. Too small < lot 50.
    # Fallback -> 50, then 10? No, should respect lot size logic?
    # Actually wait. If input is 60 and lot is 50, validate_entry would FAIL for 60.
    # But assume valid input: 100 qty (2 lots). Max 4 children.
    # 100 // 4 = 25. 25 < 50.
    # Logic in slice_order falls back to lot_size (50).
    # so we get 50, 50.

    chunks = router.slice_order("REL", 100, config)
    assert chunks == [50, 50]
    assert len(chunks) == 2  # Max child is upper bound, less is fine
