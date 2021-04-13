"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_default_conf


# Exchanges that should be tested
EXCHANGES = {
    'bittrex': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': False,
        'timeframe': '1h',
    },
    'binance': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
    'kraken': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
    'ftx': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
    'kucoin': {
        'pair': 'BTC/USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
}


@pytest.fixture(scope="class")
def exchange_conf():
    config = get_default_conf((Path(__file__).parent / "testdata").resolve())
    config['exchange']['pair_whitelist'] = []
    config['dry_run'] = False
    return config


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange(request, exchange_conf):
    exchange_conf['exchange']['name'] = request.param
    exchange = ExchangeResolver.load_exchange(request.param, exchange_conf, validate=True)

    yield exchange, request.param


@pytest.mark.longrun
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
        pair = EXCHANGES[exchangename]['pair']
        l2 = exchange.fetch_l2_order_book(pair)
        assert 'asks' in l2
        assert 'bids' in l2
        l2_limit_range = exchange._ft_has['l2_limit_range']
        l2_limit_range_required = exchange._ft_has['l2_limit_range_required']
        for val in [1, 2, 5, 25, 100]:
            l2 = exchange.fetch_l2_order_book(pair, val)
            if not l2_limit_range or val in l2_limit_range:
                assert len(l2['asks']) == val
                assert len(l2['bids']) == val
            else:
                next_limit = exchange.get_next_limit_in_list(
                    val, l2_limit_range, l2_limit_range_required)
                if next_limit is None or next_limit > 200:
                    # Large orderbook sizes can be a problem for some exchanges (bitrex ...)
                    assert len(l2['asks']) > 200
                    assert len(l2['asks']) > 200
                else:
                    assert len(l2['asks']) == next_limit
                    assert len(l2['asks']) == next_limit

    def test_fetch_ohlcv(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        timeframe = EXCHANGES[exchangename]['timeframe']
        pair_tf = (pair, timeframe)
        ohlcv = exchange.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exchange.klines(pair_tf))
        # assert len(exchange.klines(pair_tf)) > 200
        # Assume 90% uptime ...
        assert len(exchange.klines(pair_tf)) > exchange.ohlcv_candle_limit(timeframe) * 0.90
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(timezone.utc) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exchange.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)

    # TODO: tests fetch_trades (?)

    def test_ccxt_get_fee(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        assert 0 < exchange.get_fee(pair, 'limit', 'buy') < 1
        assert 0 < exchange.get_fee(pair, 'limit', 'sell') < 1
        assert 0 < exchange.get_fee(pair, 'market', 'buy') < 1
        assert 0 < exchange.get_fee(pair, 'market', 'sell') < 1
