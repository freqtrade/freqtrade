"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.

"""

from freqtrade.resolvers.exchange_resolver import ExchangeResolver
import pytest

# Exchanges that should be tested
EXCHANGES = ['bittrex', 'binance', 'kraken', 'ftx']


@pytest.fixture
def exchange_conf(default_conf):
    default_conf['exchange']['pair_whitelist'] = []
    return default_conf


@pytest.mark.parametrize('exchange', EXCHANGES)
def test_ccxt_fetch_l2_orderbook(exchange_conf, exchange):

    exchange_conf['exchange']['name'] = exchange
    exchange_conf['exchange']['name'] = exchange

    exchange = ExchangeResolver.load_exchange(exchange, exchange_conf)
    l2 = exchange.fetch_l2_order_book('BTC/USDT')
    assert 'asks' in l2
    assert 'bids' in l2

    for val in [1, 2, 5, 25, 100]:
        l2 = exchange.fetch_l2_order_book('BTC/USDT', val)
        if not exchange._ft_has['l2_limit_range'] or val in exchange._ft_has['l2_limit_range']:
            assert len(l2['asks']) == val
            assert len(l2['bids']) == val
        else:
            next_limit = exchange.get_next_limit_in_list(val, exchange._ft_has['l2_limit_range'])
            assert len(l2['asks']) == next_limit
            assert len(l2['asks']) == next_limit


