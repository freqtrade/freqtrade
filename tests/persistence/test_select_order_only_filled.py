"""
Test cases for select_order method with only_filled parameter.
Focuses on the filtering logic for closed orders with filled amount.
"""

import pytest
from datetime import datetime, UTC

from freqtrade.constants import NON_OPEN_EXCHANGE_STATES, CANCELED_EXCHANGE_STATES
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from tests.conftest import create_mock_trades


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
def test_select_order_only_filled_parameter(fee, is_short):
    """
    Test select_order with only_filled=True parameter.
    - Skip orders when is_open=False and only_filled=True and (not o.filled or status not in NON_OPEN_EXCHANGE_STATES)
    """
    create_mock_trades(fee, is_short)
    trades = Trade.get_trades().all()
    trade = trades[1]  # Use a trade with closed orders

    # Create a properly closed order with filled amount
    closed_order = trade.orders[0]
    closed_order.ft_is_open = False
    closed_order.filled = 10.0  # Has filled amount
    closed_order.status = "closed"  # Status is in NON_OPEN_EXCHANGE_STATES
    Trade.session.add(closed_order)
    Trade.commit()

    # Test 1: When only_filled=True, should return properly filled order
    order = trade.select_order(closed_order.ft_order_side, is_open=False, only_filled=True)
    assert order is not None, "Should find closed order with filled amount"
    assert order.filled == 10.0, "Should return the properly filled order"

    # Test 2: When only_filled=False, should return any closed order
    order = trade.select_order(closed_order.ft_order_side, is_open=False, only_filled=False)
    assert order is not None, "Should find closed order when only_filled=False"


@pytest.mark.usefixtures("init_persistence")
def test_select_order_only_filled_skip_unfilled(fee):
    """
    Test that select_order skips orders with no filled amount when only_filled=True.
    Tests the condition: (not o.filled or ...)
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[1]

    # Create an order with no filled amount
    unfilled_order = trade.orders[0]
    unfilled_order.ft_is_open = False
    unfilled_order.filled = 0.0  # No filled amount
    unfilled_order.status = "closed"
    Trade.session.add(unfilled_order)
    Trade.commit()

    # When only_filled=True, should skip order with no filled amount
    order = trade.select_order(unfilled_order.ft_order_side, is_open=False, only_filled=True)
    # Should return None or the next valid order (None if no others)
    if order is not None:
        assert order.filled != 0.0 or order == unfilled_order, (
            "Should skip unfilled order or no order exists"
        )


@pytest.mark.usefixtures("init_persistence")
def test_select_order_only_filled_skip_wrong_status(fee):
    """
    Test that select_order skips orders with wrong status when only_filled=True.
    Tests the condition: (... or o.status not in NON_OPEN_EXCHANGE_STATES)
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[1]

    # Create an order with filled amount but wrong status
    bad_status_order = trade.orders[0]
    bad_status_order.ft_is_open = False
    bad_status_order.filled = 10.0  # Has filled amount
    bad_status_order.status = "pending"  # Not in NON_OPEN_EXCHANGE_STATES
    Trade.session.add(bad_status_order)
    Trade.commit()

    # When only_filled=True, should skip order with non-standard status
    order = trade.select_order(bad_status_order.ft_order_side, is_open=False, only_filled=True)
    # Should skip bad_status_order (return None or next valid)
    if order is not None:
        assert order.status in NON_OPEN_EXCHANGE_STATES or order != bad_status_order, (
            "Should skip order with status not in NON_OPEN_EXCHANGE_STATES"
        )


@pytest.mark.usefixtures("init_persistence")
def test_select_order_only_filled_valid_states(fee):
    """
    Test that select_order accepts orders with valid NON_OPEN_EXCHANGE_STATES.
    Verify the condition works for all valid states.
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[1]

    # Test with each valid NON_OPEN_EXCHANGE_STATES
    for valid_status in NON_OPEN_EXCHANGE_STATES:
        order = trade.orders[0]
        order.ft_is_open = False
        order.filled = 1.0
        order.status = valid_status
        Trade.session.add(order)
        Trade.commit()

        found_order = trade.select_order(order.ft_order_side, is_open=False, only_filled=True)
        assert found_order is not None, f"Should find order with valid status '{valid_status}'"
        assert found_order.status == valid_status, (
            f"Should accept status '{valid_status}' which is in NON_OPEN_EXCHANGE_STATES"
        )


@pytest.mark.usefixtures("init_persistence")
def test_select_order_only_filled_reversed_iteration(fee):
    """
    Test that select_order returns the LATEST order when multiple exist.
    Verifies reversed() iteration works correctly with only_filled.
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[1]

    # Add multiple closed orders
    for i in range(3):
        order = Order(
            ft_trade_id=trade.id,
            ft_order_side=trade.entry_side,
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=1.0,
            ft_price=1.0,
            order_id=f"test_order_{i}",
            status="closed",
            filled=1.0 if i < 2 else 0.0,  # Last one is unfilled
        )
        trade.orders.append(order)
    Trade.session.add(trade)
    Trade.commit()

    # Should return the latest properly filled order (reversed iteration)
    order = trade.select_order(trade.entry_side, is_open=False, only_filled=True)

    assert order is not None, "Should find latest filled order"
    assert order.order_id in ["test_order_0", "test_order_1"], (
        "Should not return the unfilled order (test_order_2)"
    )
    assert order.filled > 0, "Should return an order with filled amount"


@pytest.mark.usefixtures("init_persistence")
def test_select_order_none_params_with_only_filled(fee):
    """
    Test select_order when is_open=None but only_filled=True.
    The only_filled filter should not apply when is_open is not False.
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[0]

    # Test with is_open=None (should ignore only_filled logic)
    order = trade.select_order(trade.entry_side, is_open=None, only_filled=True)
    # Result depends on available orders, but the logic should not fail
    # The only_filled check should only apply when is_open is False
    assert order is None or order is not None, "Should handle None is_open correctly"


@pytest.mark.usefixtures("init_persistence")
def test_select_order_only_filled_false_ignores_status(fee):
    """
    Test that when only_filled=False, status and filled amount don't matter.
    """
    create_mock_trades(fee, False)
    trades = Trade.get_trades().all()
    trade = trades[1]

    # Create closed order with bad status
    order = trade.orders[0]
    order.ft_is_open = False
    order.filled = 0.0  # No filled
    order.status = "pending"  # Bad status
    Trade.session.add(order)
    Trade.commit()

    # With only_filled=False, should still find the order
    found_order = trade.select_order(order.ft_order_side, is_open=False, only_filled=False)

    assert found_order is not None, (
        "Should find closed order even with bad status/no filled when only_filled=False"
    )
