from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.exceptions import OperationalException
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


@pytest.mark.parametrize('pair,mm_ratio', [
    ("ETH/USDT:USDT", 0.005),
    ("ADA/USDT:USDT", 0.003),
])
def test_get_maintenance_ratio_and_amt_gateio(default_conf, mocker, pair, mm_ratio):
    api_mock = MagicMock()
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
            }
        )
    )
    assert exchange.get_maintenance_ratio_and_amt(pair) == (mm_ratio, None)
