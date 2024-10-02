from unittest.mock import MagicMock

import pytest

from freqtrade.enums.marginmode import MarginMode
from freqtrade.leverage.liquidation_price import update_liquidation_prices


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("margin_mode", [MarginMode.CROSS, MarginMode.ISOLATED])
def test_update_liquidation_prices(mocker, margin_mode, dry_run):
    # Heavily mocked test - Only testing the logic of the function
    # update liquidation price for trade in isolated mode
    # update liquidation price for all trades in cross mode
    exchange = MagicMock()
    exchange.margin_mode = margin_mode
    wallets = MagicMock()
    trade_mock = MagicMock()

    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock])

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )

    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.get_total.call_count == (
        0 if margin_mode == MarginMode.ISOLATED or not dry_run else 1
    )

    # Test with multiple trades
    trade_mock.reset_mock()
    trade_mock_2 = MagicMock()

    mocker.patch(
        "freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock, trade_mock_2]
    )

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )
    # Trade2 is only updated in cross mode
    assert trade_mock_2.set_liquidation_price.call_count == (
        1 if margin_mode == MarginMode.CROSS else 0
    )
    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.call_count == 0 if not dry_run else 1
