from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.exceptions import InvalidOrderException, OperationalException
from freqtrade.exchange import Gateio
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_patched_exchange


def test_validate_order_types_gateio(default_conf, mocker):
    default_conf['exchange']['name'] = 'gateio'
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt')
    mocker.patch('freqtrade.exchange.Exchange._load_markets', return_value={})
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    mocker.patch('freqtrade.exchange.Exchange.name', 'Bittrex')
    exch = ExchangeResolver.load_exchange('gateio', default_conf, True)
    assert isinstance(exch, Gateio)

    default_conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    with pytest.raises(OperationalException,
                       match=r'Exchange .* does not support market orders.'):
        ExchangeResolver.load_exchange('gateio', default_conf, True)


def test_get_maintenance_ratio_and_amt_gateio(default_conf, mocker):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    type(api_mock).has = PropertyMock(return_value={'fetchLeverageTiers': False})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="gateio")
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(
            return_value={
                'ETH/USDT:USDT': {
                    'taker': 0.0000075,
                    'maker': -0.0000025,
                    'maintenanceMarginRate': 0.005,
                    'info': {},
                    'id': 'ETH_USDT',
                    'symbol': 'ETH/USDT:USDT',
                },
                'ADA/USDT:USDT': {
                    'taker': 0.0000075,
                    'maker': -0.0000025,
                    'maintenanceMarginRate': 0.003,
                    'info': {},
                    'id': 'ADA_USDT',
                    'symbol': 'ADA/USDT:USDT',
                },
                'DOGE/USDT:USDT': {
                    'taker': 0.0000075,
                    'maker': -0.0000025,
                    'maintenanceMarginRate': None,
                    'info': {},
                    'id': 'ADA_USDT',
                    'symbol': 'ADA/USDT:USDT',
                },
            }
        )
    )

    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT") == (0.005, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT") == (0.003, None)

    with pytest.raises(
        InvalidOrderException,
        match="Maintenance margin rate for DOGE/USDT:USDT is unavailable for Gateio",
    ):
        exchange.get_maintenance_ratio_and_amt('DOGE/USDT:USDT')

    with pytest.raises(
        InvalidOrderException,
        match="SHIB/USDT:USDT is not tradeable on Gateio futures",
    ):
        exchange.get_maintenance_ratio_and_amt('SHIB/USDT:USDT')
