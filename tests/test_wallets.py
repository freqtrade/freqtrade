# pragma pylint: disable=missing-docstring
from unittest.mock import MagicMock

from tests.conftest import get_patched_freqtradebot


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
        })
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used == 0.0
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('BNT') == 1.0

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
