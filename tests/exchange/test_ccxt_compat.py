"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.

"""

import pytest
from pathlib import Path
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_default_conf


# Exchanges that should be tested
EXCHANGES = {
    'bittrex': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': False
    },
    'binance': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True
    },
    'kraken': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True
    },
    'ftx': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True
    }
}


@pytest.fixture(scope="class")
def exchange_conf():
    config = get_default_conf((Path(__file__).parent / "testdata").resolve())
    config['exchange']['pair_whitelist'] = []
    return config


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange(request, exchange_conf):
    exchange_conf['exchange']['name'] = request.param
    exchange = ExchangeResolver.load_exchange(request.param, exchange_conf, validate=False)
    yield exchange, request.param


class TestCCXTExchange():

    def test_load_markets(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        markets = exchange.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)

    def test_ccxt_fetch_tickers(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        tickers = exchange.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert tickers[pair]['ask'] is not None
        assert 'bid' in tickers[pair]
        assert tickers[pair]['bid'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_ticker(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        ticker = exchange.fetch_ticker(pair)
        assert 'ask' in ticker
        assert ticker['ask'] is not None
        assert 'bid' in ticker
        assert ticker['bid'] is not None
        assert 'quoteVolume' in ticker
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert ticker['quoteVolume'] is not None

    def test_ccxt_fetch_l2_orderbook(self, exchange):
        exchange, exchangename = exchange
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

    def test_fetch_ohlcv(self, exchange):
        # TODO: Implement me
        pass

    def test_ccxt_get_fee(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        assert exchange.get_fee(pair, 'limit', 'buy') > 0 < 1
        assert exchange.get_fee(pair, 'limit', 'sell') > 0 < 1
        assert exchange.get_fee(pair, 'market', 'buy') > 0 < 1
        assert exchange.get_fee(pair, 'market', 'sell') > 0 < 1
