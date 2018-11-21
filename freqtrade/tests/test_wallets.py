# pragma pylint: disable=missing-docstring
from freqtrade.tests.conftest import get_patched_freqtradebot
from unittest.mock import MagicMock


def test_sync_wallet_at_boot(mocker, default_conf):
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

    assert len(freqtrade.wallets.wallets) == 2
    assert freqtrade.wallets.wallets['BNT'].free == 1.0
    assert freqtrade.wallets.wallets['BNT'].used == 2.0
    assert freqtrade.wallets.wallets['BNT'].total == 3.0
    assert freqtrade.wallets.wallets['GAS'].free == 0.260739
    assert freqtrade.wallets.wallets['GAS'].used == 0.0
    assert freqtrade.wallets.wallets['GAS'].total == 0.260739

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

    assert len(freqtrade.wallets.wallets) == 2
    assert freqtrade.wallets.wallets['BNT'].free == 1.2
    assert freqtrade.wallets.wallets['BNT'].used == 1.9
    assert freqtrade.wallets.wallets['BNT'].total == 3.5
    assert freqtrade.wallets.wallets['GAS'].free == 0.270739
    assert freqtrade.wallets.wallets['GAS'].used == 0.1
    assert freqtrade.wallets.wallets['GAS'].total == 0.260439
