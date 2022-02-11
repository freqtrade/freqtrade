from unittest.mock import MagicMock  # , PropertyMock

from tests.conftest import get_patched_exchange


def test_get_maintenance_ratio_and_amt_okex(
    default_conf,
    mocker,
):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['dry_run'] = False
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


def test_get_max_pair_stake_amount_okex(default_conf, mocker, leverage_tiers):

    exchange = get_patched_exchange(mocker, default_conf, id="okex")
    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == float('inf')

    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, id="okex")
    exchange._leverage_tiers = leverage_tiers

    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount('BNB/USDT', 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0, 10.0) == 100000000

    assert exchange.get_max_pair_stake_amount('TTT/USDT', 1.0) == float('inf')  # Not in tiers
