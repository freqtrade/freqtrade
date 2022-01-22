"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_default_conf_usdt


# Exchanges that should be tested
EXCHANGES = {
    'bittrex': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': False,
        'timeframe': '1h',
    },
    'binance': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
        'futures': True,
    },
    'kraken': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
    'ftx': {
        'pair': 'BTC/USD',
        'stake_currency': 'USD',
        'hasQuoteVolume': True,
        'timeframe': '5m',
        'futures_pair': 'BTC/USD:USD',
        'futures': True,
    },
    'kucoin': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
    'gateio': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
        'futures': True,
        'futures_pair': 'BTC/USDT:USDT',
    },
    'okex': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '5m',
        'futures_pair': 'BTC/USDT:USDT',
        'futures': True,
    },
    'bitvavo': {
        'pair': 'BTC/EUR',
        'stake_currency': 'EUR',
        'hasQuoteVolume': True,
        'timeframe': '5m',
    },
}


@pytest.fixture(scope="class")
def exchange_conf():
    config = get_default_conf_usdt((Path(__file__).parent / "testdata").resolve())
    config['exchange']['pair_whitelist'] = []
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''
    config['dry_run'] = False
    return config


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange(request, exchange_conf):
    exchange_conf['exchange']['name'] = request.param
    exchange_conf['stake_currency'] = EXCHANGES[request.param]['stake_currency']
    exchange = ExchangeResolver.load_exchange(request.param, exchange_conf, validate=True)

    yield exchange, request.param


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange_futures(request, exchange_conf, class_mocker):
    if not EXCHANGES[request.param].get('futures') is True:
        yield None, request.param
    else:
        exchange_conf = deepcopy(exchange_conf)
        exchange_conf['exchange']['name'] = request.param
        exchange_conf['trading_mode'] = 'futures'
        exchange_conf['collateral'] = 'cross'
        exchange_conf['stake_currency'] = EXCHANGES[request.param]['stake_currency']

        # TODO-lev: This mock should no longer be necessary once futures are enabled.
        class_mocker.patch(
            'freqtrade.exchange.exchange.Exchange.validate_trading_mode_and_collateral')
        class_mocker.patch(
            'freqtrade.exchange.binance.Binance.fill_leverage_brackets')

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
        assert exchange.market_is_spot(markets[pair])

    def test_load_markets_futures(self, exchange_futures):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return
        pair = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        markets = exchange.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)

        assert exchange.market_is_future(markets[pair])

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

        pair_tf = (pair, timeframe, CandleType.SPOT)

        ohlcv = exchange.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exchange.klines(pair_tf))
        # assert len(exchange.klines(pair_tf)) > 200
        # Assume 90% uptime ...
        assert len(exchange.klines(pair_tf)) > exchange.ohlcv_candle_limit(timeframe) * 0.90
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(timezone.utc) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exchange.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return

        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        timeframe_ff = exchange._ft_has.get('funding_fee_timeframe',
                                            exchange._ft_has['mark_ohlcv_timeframe'])
        pair_tf = (pair, timeframe_ff, CandleType.FUNDING_RATE)

        funding_ohlcv = exchange.refresh_latest_ohlcv(
            [pair_tf],
            since_ms=since,
            drop_incomplete=False)

        assert isinstance(funding_ohlcv, dict)
        rate = funding_ohlcv[pair_tf]

        this_hour = timeframe_to_prev_date(timeframe_ff)
        hour1 = timeframe_to_prev_date(timeframe_ff, this_hour - timedelta(minutes=1))
        hour2 = timeframe_to_prev_date(timeframe_ff, hour1 - timedelta(minutes=1))
        hour3 = timeframe_to_prev_date(timeframe_ff, hour2 - timedelta(minutes=1))
        val0 = rate[rate['date'] == this_hour].iloc[0]['open']
        val1 = rate[rate['date'] == hour1].iloc[0]['open']
        val2 = rate[rate['date'] == hour2].iloc[0]['open']
        val3 = rate[rate['date'] == hour3].iloc[0]['open']

        # Test For last 4 hours
        # Avoids random test-failure when funding-fees are 0 for a few hours.
        assert val0 != 0.0 or val1 != 0.0 or val2 != 0.0 or val3 != 0.0
        # We expect funding rates to be different from 0.0 - or moving around.
        assert (
            rate['open'].max() != 0.0 or rate['open'].min() != 0.0 or
            (rate['open'].min() != rate['open'].max())
        )

    def test_ccxt_fetch_mark_price_history(self, exchange_futures):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        pair_tf = (pair, '1h', CandleType.MARK)

        mark_ohlcv = exchange.refresh_latest_ohlcv(
            [pair_tf],
            since_ms=since,
            drop_incomplete=False)

        assert isinstance(mark_ohlcv, dict)
        expected_tf = '1h'
        mark_candles = mark_ohlcv[pair_tf]

        this_hour = timeframe_to_prev_date(expected_tf)
        prev_hour = timeframe_to_prev_date(expected_tf, this_hour - timedelta(minutes=1))

        assert mark_candles[mark_candles['date'] == prev_hour].iloc[0]['open'] != 0.0
        assert mark_candles[mark_candles['date'] == this_hour].iloc[0]['open'] != 0.0

    def test_ccxt__calculate_funding_fees(self, exchange_futures):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since = datetime.now(timezone.utc) - timedelta(days=5)

        funding_fee = exchange._calculate_funding_fees(pair, 20, open_date=since)

        assert isinstance(funding_fee, float)
        # assert funding_fee > 0

    # TODO: tests fetch_trades (?)

    def test_ccxt_get_fee(self, exchange):
        exchange, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        threshold = 0.01
        assert 0 < exchange.get_fee(pair, 'limit', 'buy') < threshold
        assert 0 < exchange.get_fee(pair, 'limit', 'sell') < threshold
        assert 0 < exchange.get_fee(pair, 'market', 'buy') < threshold
        assert 0 < exchange.get_fee(pair, 'market', 'sell') < threshold
