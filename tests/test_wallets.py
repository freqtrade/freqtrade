# pragma pylint: disable=missing-docstring
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import Trade
from tests.conftest import EXMS, create_mock_trades, get_patched_freqtradebot, patch_wallet


def test_sync_wallet_at_boot(mocker, default_conf):
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            "BNT": {
                "free": 1.0,
                "used": 2.0,
                "total": 3.0
            },
            "GAS": {
                "free": 0.260739,
                "used": 0.0,
                "total": 0.260739
            },
            "USDT": {
                "free": 20,
                "used": 20,
                "total": 40
            },
        })
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert len(freqtrade.wallets._wallets) == 3
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used == 0.0
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('BNT') == 1.0
    assert 'USDT' in freqtrade.wallets._wallets
    assert freqtrade.wallets._last_wallet_refresh is not None
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            "BNT": {
                "free": 1.2,
                "used": 1.9,
                "total": 3.5
            },
            "GAS": {
                "free": 0.270739,
                "used": 0.1,
                "total": 0.260439
            },
        })
    )

    freqtrade.wallets.update()

    # USDT is missing from the 2nd result - so should not be in this either.
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.2
    assert freqtrade.wallets._wallets['BNT'].used == 1.9
    assert freqtrade.wallets._wallets['BNT'].total == 3.5
    assert freqtrade.wallets._wallets['GAS'].free == 0.270739
    assert freqtrade.wallets._wallets['GAS'].used == 0.1
    assert freqtrade.wallets._wallets['GAS'].total == 0.260439
    assert freqtrade.wallets.get_free('GAS') == 0.270739
    assert freqtrade.wallets.get_used('GAS') == 0.1
    assert freqtrade.wallets.get_total('GAS') == 0.260439
    update_mock = mocker.patch('freqtrade.wallets.Wallets._update_live')
    freqtrade.wallets.update(False)
    assert update_mock.call_count == 0
    freqtrade.wallets.update()
    assert update_mock.call_count == 1

    assert freqtrade.wallets.get_free('NOCURRENCY') == 0
    assert freqtrade.wallets.get_used('NOCURRENCY') == 0
    assert freqtrade.wallets.get_total('NOCURRENCY') == 0


def test_sync_wallet_missing_data(mocker, default_conf):
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            "BNT": {
                "free": 1.0,
                "used": 2.0,
                "total": 3.0
            },
            "GAS": {
                "free": 0.260739,
                "total": 0.260739
            },
        })
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used is None
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('GAS') == 0.260739


def test_get_trade_stake_amount_no_stake_amount(default_conf, mocker) -> None:
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.wallets.get_trade_stake_amount('ETH/BTC')


@pytest.mark.parametrize("balance_ratio,capital,result1,result2", [
                        (1,    None, 50, 66.66666),
                        (0.99, None, 49.5, 66.0),
                        (0.50, None, 25, 33.3333),
    # Tests with capital ignore balance_ratio
                        (1,    100, 50, 0.0),
                        (0.99, 200, 50, 66.66666),
                        (0.99, 150, 50, 50),
                        (0.50, 50, 25, 0.0),
                        (0.50, 10, 5, 0.0),
])
def test_get_trade_stake_amount_unlimited_amount(default_conf, ticker, balance_ratio, capital,
                                                 result1, result2, limit_buy_order_open,
                                                 fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        create_order=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee
    )

    conf = deepcopy(default_conf)
    conf['stake_amount'] = UNLIMITED_STAKE_AMOUNT
    conf['dry_run_wallet'] = 100
    conf['max_open_trades'] = 2
    conf['tradable_balance_ratio'] = balance_ratio
    if capital is not None:
        conf['available_capital'] = capital

    freqtrade = get_patched_freqtradebot(mocker, conf)

    # no open trades, order amount should be 'balance / max_open_trades'
    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT')
    assert result == result1

    # create one trade, order amount should be 'balance / (max_open_trades - num_open_trades)'
    freqtrade.execute_entry('ETH/USDT', result)

    result = freqtrade.wallets.get_trade_stake_amount('LTC/USDT')
    assert result == result1

    # create 2 trades, order amount should be None
    freqtrade.execute_entry('LTC/BTC', result)

    result = freqtrade.wallets.get_trade_stake_amount('XRP/USDT')
    assert result == 0

    freqtrade.config['max_open_trades'] = 3
    freqtrade.config['dry_run_wallet'] = 200
    freqtrade.wallets.start_cap = 200
    result = freqtrade.wallets.get_trade_stake_amount('XRP/USDT')
    assert round(result, 4) == round(result2, 4)

    # set max_open_trades = None, so do not trade
    freqtrade.config['max_open_trades'] = 0
    result = freqtrade.wallets.get_trade_stake_amount('NEO/USDT')
    assert result == 0


@pytest.mark.parametrize('stake_amount,min_stake,stake_available,max_stake,trade_amount,expected', [
    (22, 11, 50, 10000, None, 22),
    (100, 11, 500, 10000, None, 100),
    (1000, 11, 500, 10000, None, 500),  # Above stake_available
    (700, 11, 1000, 400, None, 400),  # Above max_stake, below stake available
    (20, 15, 10, 10000, None, 0),  # Minimum stake > stake_available
    (9, 11, 100, 10000, None, 11),  # Below min stake
    (1, 15, 10, 10000, None, 0),  # Below min stake and min_stake > stake_available
    (20, 50, 100, 10000, None, 0),  # Below min stake and stake * 1.3 > min_stake
    (1000, None, 1000, 10000, None, 1000),  # No min-stake-amount could be determined
    (2000, 15, 2000, 3000, 1500, 1500),  # Rebuy - resulting in too high stake amount. Adjusting.
])
def test_validate_stake_amount(
    mocker,
    default_conf,
    stake_amount,
    min_stake,
    stake_available,
    max_stake,
    trade_amount,
    expected,
):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch("freqtrade.wallets.Wallets.get_available_stake_amount",
                 return_value=stake_available)
    res = freqtrade.wallets.validate_stake_amount(
        'XRP/USDT', stake_amount, min_stake, max_stake, trade_amount)
    assert res == expected


@pytest.mark.parametrize('available_capital,closed_profit,open_stakes,free,expected', [
    (None, 10, 100, 910, 1000),
    (None, 0, 0, 2500, 2500),
    (None, 500, 0, 2500, 2000),
    (None, 500, 0, 2500, 2000),
    (None, -70, 0, 1930, 2000),
    # Only available balance matters when it's set.
    (100, 0, 0, 0, 100),
    (1000, 0, 2, 5, 1000),
    (1235, 2250, 2, 5, 1235),
    (1235, -2250, 2, 5, 1235),
])
def test_get_starting_balance(mocker, default_conf, available_capital, closed_profit,
                              open_stakes, free, expected):
    if available_capital:
        default_conf['available_capital'] = available_capital
    mocker.patch("freqtrade.persistence.models.Trade.get_total_closed_profit",
                 return_value=closed_profit)
    mocker.patch("freqtrade.persistence.models.Trade.total_open_trades_stakes",
                 return_value=open_stakes)
    mocker.patch("freqtrade.wallets.Wallets.get_free", return_value=free)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert freqtrade.wallets.get_starting_balance() == expected


def test_sync_wallet_futures_live(mocker, default_conf):
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    mock_result = [
        {
            "symbol": "ETH/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 100.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 100.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 2896.41,
            "collateral": 20,
            "marginType": "isolated",
            "side": 'short',
            "percentage": None
        },
        {
            "symbol": "ADA/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 100.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 100.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 0.91,
            "collateral": 20,
            "marginType": "isolated",
            "side": 'short',
            "percentage": None
        },
        {
            # Closed position
            "symbol": "SOL/BUSD:BUSD",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 0.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 0.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 15.41,
            "collateral": 0.0,
            "marginType": "isolated",
            "side": 'short',
            "percentage": None
        }
    ]
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            "USDT": {
                "free": 900,
                "used": 100,
                "total": 1000
            },
        }),
        fetch_positions=MagicMock(return_value=mock_result)
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 2

    assert 'USDT' in freqtrade.wallets._wallets
    assert 'ETH/USDT:USDT' in freqtrade.wallets._positions
    assert freqtrade.wallets._last_wallet_refresh is not None

    # Remove ETH/USDT:USDT position
    del mock_result[0]
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._positions) == 1
    assert 'ETH/USDT:USDT' not in freqtrade.wallets._positions


def test_sync_wallet_futures_dry(mocker, default_conf, fee):
    default_conf['dry_run'] = True
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 0

    create_mock_trades(fee, is_short=None)

    freqtrade.wallets.update()

    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 4
    positions = freqtrade.wallets.get_all_positions()
    positions['ETH/BTC'].side == 'short'
    positions['ETC/BTC'].side == 'long'
    positions['XRP/BTC'].side == 'long'
    positions['LTC/BTC'].side == 'short'

    assert freqtrade.wallets.get_starting_balance() == default_conf['dry_run_wallet']
    total = freqtrade.wallets.get_total('BTC')
    free = freqtrade.wallets.get_free('BTC')
    used = freqtrade.wallets.get_used('BTC')
    assert free + used == total


def test_check_exit_amount(mocker, default_conf, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    update_mock = mocker.patch("freqtrade.wallets.Wallets.update")
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=123)

    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.amount == 123

    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert update_mock.call_count == 0
    assert total_mock.call_count == 1

    update_mock.reset_mock()
    # Reduce returned amount to below the trade amount - which should
    # trigger a wallet update and return False, triggering "order refinding"
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=100)
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert update_mock.call_count == 1
    assert total_mock.call_count == 2


def test_check_exit_amount_futures(mocker, default_conf, fee):
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=123)

    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    trade.trading_mode = 'futures'
    assert trade.amount == 123

    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert total_mock.call_count == 0

    update_mock = mocker.patch("freqtrade.wallets.Wallets.update")
    trade.amount = 150
    # Reduce returned amount to below the trade amount - which should
    # trigger a wallet update and return False, triggering "order refinding"
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert total_mock.call_count == 0
    assert update_mock.call_count == 1
