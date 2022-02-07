from unittest.mock import MagicMock  # , PropertyMock

from tests.conftest import get_patched_exchange


def test_get_maintenance_ratio_and_amt_okex(
    default_conf,
    mocker,
):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['dry_run'] = True
    api_mock.fetch_leverage_tiers = MagicMock(return_value={
        'SHIB/USDT:USDT': [
            {
                'tier': 1,
                'notionalFloor': 0,
                'notionalCap': 2000,
                'maintenanceMarginRatio': 0.01,
                'maxLeverage': 75,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.013',
                    'instId': '',
                    'maxLever': '75',
                    'maxSz': '2000',
                    'minSz': '0',
                    'mmr': '0.01',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '1',
                    'uly': 'SHIB-USDT'
                }
            },
            {
                'tier': 2,
                # TODO-lev: What about a value between 2000 and 2001?
                'notionalFloor': 2001,
                'notionalCap': 4000,
                'maintenanceMarginRatio': 0.015,
                'maxLeverage': 50,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.02',
                    'instId': '',
                    'maxLever': '50',
                    'maxSz': '4000',
                    'minSz': '2001',
                    'mmr': '0.015',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '2',
                    'uly': 'SHIB-USDT'
                }
            },
            {
                'tier': 3,
                'notionalFloor': 4001,
                'notionalCap': 8000,
                'maintenanceMarginRatio': 0.02,
                'maxLeverage': 20,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.05',
                    'instId': '',
                    'maxLever': '20',
                    'maxSz': '8000',
                    'minSz': '4001',
                    'mmr': '0.02',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '3',
                    'uly': 'SHIB-USDT'
                }
            },
        ],
        'DOGE/USDT:USDT': [
            {
                'tier': 1,
                'notionalFloor': 0,
                'notionalCap': 500,
                'maintenanceMarginRatio': 0.02,
                'maxLeverage': 75,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.013',
                    'instId': '',
                    'maxLever': '75',
                    'maxSz': '500',
                    'minSz': '0',
                    'mmr': '0.01',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '1',
                    'uly': 'DOGE-USDT'
                }
            },
            {
                'tier': 2,
                'notionalFloor': 501,
                'notionalCap': 1000,
                'maintenanceMarginRatio': 0.025,
                'maxLeverage': 50,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.02',
                    'instId': '',
                    'maxLever': '50',
                    'maxSz': '1000',
                    'minSz': '501',
                    'mmr': '0.015',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '2',
                    'uly': 'DOGE-USDT'
                }
            },
            {
                'tier': 3,
                'notionalFloor': 1001,
                'notionalCap': 2000,
                'maintenanceMarginRatio': 0.03,
                'maxLeverage': 20,
                'info': {
                    'baseMaxLoan': '',
                    'imr': '0.05',
                    'instId': '',
                    'maxLever': '20',
                    'maxSz': '2000',
                    'minSz': '1001',
                    'mmr': '0.02',
                    'optMgnFactor': '0',
                    'quoteMaxLoan': '',
                    'tier': '3',
                    'uly': 'DOGE-USDT'
                }
            },
        ]
    })
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="okex")
    assert exchange.get_maintenance_ratio_and_amt('SHIB/USDT:USDT', 2000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt('SHIB/USDT:USDT', 2001) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt('SHIB/USDT:USDT', 4001) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('SHIB/USDT:USDT', 8000) == (0.02, None)

    assert exchange.get_maintenance_ratio_and_amt('DOGE/USDT:USDT', 1) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('DOGE/USDT:USDT', 2000) == (0.03, None)


def test_get_max_pair_stake_amount_okex(default_conf, mocker):

    exchange = get_patched_exchange(mocker, default_conf, id="okex")
    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == float('inf')

    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, id="okex")
    exchange._leverage_tiers = {
        'BNB/BUSD': [
            {
                "min": 0,       # stake(before leverage) = 0
                "max": 100000,  # max stake(before leverage) = 5000
                "mmr": 0.025,
                "lev": 20,
                "maintAmt": 0.0
            },
            {
                "min": 100000,  # stake = 10000.0
                "max": 500000,  # max_stake = 50000.0
                "mmr": 0.05,
                "lev": 10,
                "maintAmt": 2500.0
            },
            {
                "min": 500000,   # stake = 100000.0
                "max": 1000000,  # max_stake = 200000.0
                "mmr": 0.1,
                "lev": 5,
                "maintAmt": 27500.0
            },
            {
                "min": 1000000,  # stake = 333333.3333333333
                "max": 2000000,  # max_stake = 666666.6666666666
                "mmr": 0.15,
                "lev": 3,
                "maintAmt": 77500.0
            },
            {
                "min": 2000000,  # stake = 1000000.0
                "max": 5000000,  # max_stake = 2500000.0
                "mmr": 0.25,
                "lev": 2,
                "maintAmt": 277500.0
            },
            {
                "min": 5000000,   # stake = 5000000.0
                "max": 30000000,  # max_stake = 30000000.0
                "mmr": 0.5,
                "lev": 1,
                "maintAmt": 1527500.0
            }
        ],
        'BNB/USDT': [
            {
                "min": 0,      # stake = 0.0
                "max": 10000,  # max_stake = 133.33333333333334
                "mmr": 0.0065,
                "lev": 75,
                "maintAmt": 0.0
            },
            {
                "min": 10000,  # stake = 200.0
                "max": 50000,  # max_stake = 1000.0
                "mmr": 0.01,
                "lev": 50,
                "maintAmt": 35.0
            },
            {
                "min": 50000,   # stake = 2000.0
                "max": 250000,  # max_stake = 10000.0
                "mmr": 0.02,
                "lev": 25,
                "maintAmt": 535.0
            },
            {
                "min": 250000,   # stake = 25000.0
                "max": 1000000,  # max_stake = 100000.0
                "mmr": 0.05,
                "lev": 10,
                "maintAmt": 8035.0
            },
            {
                "min": 1000000,  # stake = 200000.0
                "max": 2000000,  # max_stake = 400000.0
                "mmr": 0.1,
                "lev": 5,
                "maintAmt": 58035.0
            },
            {
                "min": 2000000,  # stake = 500000.0
                "max": 5000000,  # max_stake = 1250000.0
                "mmr": 0.125,
                "lev": 4,
                "maintAmt": 108035.0
            },
            {
                "min": 5000000,   # stake = 1666666.6666666667
                "max": 10000000,  # max_stake = 3333333.3333333335
                "mmr": 0.15,
                "lev": 3,
                "maintAmt": 233035.0
            },
            {
                "min": 10000000,  # stake = 5000000.0
                "max": 20000000,  # max_stake = 10000000.0
                "mmr": 0.25,
                "lev": 2,
                "maintAmt": 1233035.0
            },
            {
                "min": 20000000,  # stake = 20000000.0
                "max": 50000000,  # max_stake = 50000000.0
                "mmr": 0.5,
                "lev": 1,
                "maintAmt": 6233035.0
            },
        ],
        'BTC/USDT': [
            {
                "min": 0,      # stake = 0.0
                "max": 50000,  # max_stake = 400.0
                "mmr": 0.004,
                "lev": 125,
                "maintAmt": 0.0
            },
            {
                "min": 50000,   # stake = 500.0
                "max": 250000,  # max_stake = 2500.0
                "mmr": 0.005,
                "lev": 100,
                "maintAmt": 50.0
            },
            {
                "min": 250000,   # stake = 5000.0
                "max": 1000000,  # max_stake = 20000.0
                "mmr": 0.01,
                "lev": 50,
                "maintAmt": 1300.0
            },
            {
                "min": 1000000,  # stake = 50000.0
                "max": 7500000,  # max_stake = 375000.0
                "mmr": 0.025,
                "lev": 20,
                "maintAmt": 16300.0
            },
            {
                "min": 7500000,   # stake = 750000.0
                "max": 40000000,  # max_stake = 4000000.0
                "mmr": 0.05,
                "lev": 10,
                "maintAmt": 203800.0
            },
            {
                "min": 40000000,   # stake = 8000000.0
                "max": 100000000,  # max_stake = 20000000.0
                "mmr": 0.1,
                "lev": 5,
                "maintAmt": 2203800.0
            },
            {
                "min": 100000000,  # stake = 25000000.0
                "max": 200000000,  # max_stake = 50000000.0
                "mmr": 0.125,
                "lev": 4,
                "maintAmt": 4703800.0
            },
            {
                "min": 200000000,  # stake = 66666666.666666664
                "max": 400000000,  # max_stake = 133333333.33333333
                "mmr": 0.15,
                "lev": 3,
                "maintAmt": 9703800.0
            },
            {
                "min": 400000000,  # stake = 200000000.0
                "max": 600000000,  # max_stake = 300000000.0
                "mmr": 0.25,
                "lev": 2,
                "maintAmt": 4.97038E7
            },
            {
                "min": 600000000,   # stake = 600000000.0
                "max": 1000000000,  # max_stake = 1000000000.0
                "mmr": 0.5,
                "lev": 1,
                "maintAmt": 1.997038E8
            },
        ]
    }

    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount('BNB/USDT', 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0, 10.0) == 100000000

    exchange.get_leverage_tiers_for_pair = MagicMock(return_value=None)
    assert exchange.get_max_pair_stake_amount('TTT/USDT', 1.0) == float('inf')  # Not in tiers
