# pragma pylint: disable=missing-docstring
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from tests.conftest import get_patched_freqtradebot, patch_wallet


def test_sync_wallet_at_boot(mocker, default_conf):
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
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
    assert freqtrade.wallets._last_wallet_refresh > 0
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
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
        'freqtrade.exchange.Exchange',
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


@pytest.mark.parametrize("balance_ratio,result1,result2", [
                        (1, 50, 66.66666),
                        (0.99, 49.5, 66.0),
                        (0.50, 25, 33.3333),
])
def test_get_trade_stake_amount_unlimited_amount(default_conf, ticker, balance_ratio, result1,
                                                 result2, limit_buy_order_open,
                                                 fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee
    )

    conf = deepcopy(default_conf)
    conf['stake_amount'] = UNLIMITED_STAKE_AMOUNT
    conf['dry_run_wallet'] = 100
    conf['max_open_trades'] = 2
    conf['tradable_balance_ratio'] = balance_ratio

    freqtrade = get_patched_freqtradebot(mocker, conf)

    # no open trades, order amount should be 'balance / max_open_trades'
    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT')
    assert result == result1

    # create one trade, order amount should be 'balance / (max_open_trades - num_open_trades)'
    freqtrade.execute_buy('ETH/USDT', result)

    result = freqtrade.wallets.get_trade_stake_amount('LTC/USDT')
    assert result == result1

    # create 2 trades, order amount should be None
    freqtrade.execute_buy('LTC/BTC', result)

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
