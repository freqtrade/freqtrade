from unittest.mock import MagicMock, PropertyMock

from freqtrade.enums import MarginMode, TradingMode
from tests.conftest import get_patched_exchange


def test_get_maintenance_ratio_and_amt_okx(
    default_conf,
    mocker,
):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        'freqtrade.exchange.Okx',
        exchange_has=MagicMock(return_value=True),
        load_leverage_tiers=MagicMock(return_value={
            'ETH/USDT:USDT': [
                {
                    'tier': 1,
                    'minNotional': 0,
                    'maxNotional': 2000,
                    'maintenanceMarginRate': 0.01,
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
                        'uly': 'ETH-USDT'
                    }
                },
                {
                    'tier': 2,
                    'minNotional': 2001,
                    'maxNotional': 4000,
                    'maintenanceMarginRate': 0.015,
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
                        'uly': 'ETH-USDT'
                    }
                },
                {
                    'tier': 3,
                    'minNotional': 4001,
                    'maxNotional': 8000,
                    'maintenanceMarginRate': 0.02,
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
                        'uly': 'ETH-USDT'
                    }
                },
            ],
            'ADA/USDT:USDT': [
                {
                    'tier': 1,
                    'minNotional': 0,
                    'maxNotional': 500,
                    'maintenanceMarginRate': 0.02,
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
                        'uly': 'ADA-USDT'
                    }
                },
                {
                    'tier': 2,
                    'minNotional': 501,
                    'maxNotional': 1000,
                    'maintenanceMarginRate': 0.025,
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
                        'uly': 'ADA-USDT'
                    }
                },
                {
                    'tier': 3,
                    'minNotional': 1001,
                    'maxNotional': 2000,
                    'maintenanceMarginRate': 0.03,
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
                        'uly': 'ADA-USDT'
                    }
                },
            ]
        })
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="okx")
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 2000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 2001) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 4001) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 8000) == (0.02, None)

    assert exchange.get_maintenance_ratio_and_amt('ADA/USDT:USDT', 1) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('ADA/USDT:USDT', 2000) == (0.03, None)


def test_get_max_pair_stake_amount_okx(default_conf, mocker, leverage_tiers):

    exchange = get_patched_exchange(mocker, default_conf, id="okx")
    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == float('inf')

    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, id="okx")
    exchange._leverage_tiers = leverage_tiers

    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount('BNB/USDT', 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT', 1.0, 10.0) == 100000000

    assert exchange.get_max_pair_stake_amount('TTT/USDT', 1.0) == float('inf')  # Not in tiers


def test_load_leverage_tiers_okx(default_conf, mocker, markets):
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={
        'fetchLeverageTiers': False,
        'fetchMarketLeverageTiers': True,
    })
    api_mock.fetch_market_leverage_tiers = MagicMock(side_effect=[
        [
            {
                'tier': 1,
                'minNotional': 0,
                'maxNotional': 500,
                'maintenanceMarginRate': 0.02,
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
                    'uly': 'ADA-USDT'
                }
            },
            {
                'tier': 2,
                'minNotional': 501,
                'maxNotional': 1000,
                'maintenanceMarginRate': 0.025,
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
                    'uly': 'ADA-USDT'
                }
            },
            {
                'tier': 3,
                'minNotional': 1001,
                'maxNotional': 2000,
                'maintenanceMarginRate': 0.03,
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
                    'uly': 'ADA-USDT'
                }
            },
        ],
        [
            {
                'tier': 1,
                'minNotional': 0,
                'maxNotional': 2000,
                'maintenanceMarginRate': 0.01,
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
                    'uly': 'ETH-USDT'
                }
            },
            {
                'tier': 2,
                'minNotional': 2001,
                'maxNotional': 4000,
                'maintenanceMarginRate': 0.015,
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
                    'uly': 'ETH-USDT'
                }
            },
            {
                'tier': 3,
                'minNotional': 4001,
                'maxNotional': 8000,
                'maintenanceMarginRate': 0.02,
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
                    'uly': 'ETH-USDT'
                }
            },
        ]
    ])
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['stake_currency'] = 'USDT'
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange.markets = markets
    # Initialization of load_leverage_tiers happens as part of exchange init.
    assert exchange._leverage_tiers == {
        'ADA/USDT:USDT': [
            {
                'min': 0,
                'max': 500,
                'mmr': 0.02,
                'lev': 75,
                'maintAmt': None
            },
            {
                'min': 501,
                'max': 1000,
                'mmr': 0.025,
                'lev': 50,
                'maintAmt': None
            },
            {
                'min': 1001,
                'max': 2000,
                'mmr': 0.03,
                'lev': 20,
                'maintAmt': None
            },
        ],
        'ETH/USDT:USDT': [
            {
                'min': 0,
                'max': 2000,
                'mmr': 0.01,
                'lev': 75,
                'maintAmt': None
            },
            {
                'min': 2001,
                'max': 4000,
                'mmr': 0.015,
                'lev': 50,
                'maintAmt': None
            },
            {
                'min': 4001,
                'max': 8000,
                'mmr': 0.02,
                'lev': 20,
                'maintAmt': None
            },
        ],
    }
