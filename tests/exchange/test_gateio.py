from unittest.mock import MagicMock

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


def test_fetch_stoploss_order_gateio(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, id='gateio')

    fetch_order_mock = MagicMock()
    exchange.fetch_order = fetch_order_mock

    exchange.fetch_stoploss_order('1234', 'ETH/BTC')
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][1]['order_id'] == '1234'
    assert fetch_order_mock.call_args_list[0][1]['pair'] == 'ETH/BTC'
    assert fetch_order_mock.call_args_list[0][1]['params'] == {'stop': True}


def test_cancel_stoploss_order_gateio(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, id='gateio')

    cancel_order_mock = MagicMock()
    exchange.cancel_order = cancel_order_mock

    exchange.cancel_stoploss_order('1234', 'ETH/BTC')
    assert cancel_order_mock.call_count == 1
    assert cancel_order_mock.call_args_list[0][1]['order_id'] == '1234'
    assert cancel_order_mock.call_args_list[0][1]['pair'] == 'ETH/BTC'
    assert cancel_order_mock.call_args_list[0][1]['params'] == {'stop': True}
