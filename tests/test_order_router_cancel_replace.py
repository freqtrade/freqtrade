import pytest
from unittest.mock import MagicMock
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


def test_edit_order_cancel_replace_flow():
    # Setup
    ex = BreezeCCXT({})
    ex.cancel_order = MagicMock()
    ex.create_order = MagicMock(return_value={"id": "NEW_ORD_ID"})
    ex.order_router = MagicMock()  # Mock router to bypass quota check

    # Execute
    res = ex.edit_order("OLD_ID", "RELIANCE/INR", "limit", "buy", 10, 2500)

    # Assert
    ex.cancel_order.assert_called_with("OLD_ID", "RELIANCE/INR")
    ex.create_order.assert_called_with("RELIANCE/INR", "limit", "buy", 10, 2500, {})
    assert res == {"id": "NEW_ORD_ID"}


def test_edit_order_enforces_router():
    ex = BreezeCCXT({})
    ex.order_router.track_and_assert_modify = MagicMock()
    ex.cancel_order = MagicMock()
    ex.create_order = MagicMock()

    ex.edit_order("ORD-1", "RELIANCE/INR", "limit", "buy", 10, 2500)

    # Verify router called
    ex.order_router.track_and_assert_modify.assert_called()
    call_args = ex.order_router.track_and_assert_modify.call_args
    assert call_args[0][0] == "ORD-1"  # Check order ID passed
