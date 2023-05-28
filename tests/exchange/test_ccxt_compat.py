"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import pytest

from freqtrade.constants import Config
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange import Exchange, timeframe_to_msecs
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import EXMS, get_default_conf_usdt


EXCHANGE_FIXTURE_TYPE = Tuple[Exchange, str]

# Exchanges that should be tested
EXCHANGES = {
    'bittrex': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': False,
        'timeframe': '1h',
        'leverage_tiers_public': False,
        'leverage_in_spot_market': False,
    },
    'binance': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'use_ci_proxy': True,
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'futures': True,
        'futures_pair': 'BTC/USDT:USDT',
        'hasQuoteVolumeFutures': True,
        'leverage_tiers_public': False,
        'leverage_in_spot_market': False,
        'private_methods': [
            'fapiPrivateGetPositionSideDual',
            'fapiPrivateGetMultiAssetsMargin'
        ],
        'sample_order': [{
            "symbol": "SOLUSDT",
            "orderId": 3551312894,
            "orderListId": -1,
            "clientOrderId": "x-R4DD3S8297c73a11ccb9dc8f2811ba",
            "transactTime": 1674493798550,
            "price": "15.50000000",
            "origQty": "1.10000000",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "workingTime": 1674493798550,
            "fills": [],
            "selfTradePreventionMode": "NONE",
        }]
    },
    'binanceus': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'futures': False,
        'sample_order': [{
            "symbol": "SOLUSDT",
            "orderId": 3551312894,
            "orderListId": -1,
            "clientOrderId": "x-R4DD3S8297c73a11ccb9dc8f2811ba",
            "transactTime": 1674493798550,
            "price": "15.50000000",
            "origQty": "1.10000000",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "workingTime": 1674493798550,
            "fills": [],
            "selfTradePreventionMode": "NONE",
        }]
    },
    'kraken': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'leverage_tiers_public': False,
        'leverage_in_spot_market': True,
    },
    'kucoin': {
        'pair': 'XRP/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'leverage_tiers_public': False,
        'leverage_in_spot_market': True,
        'sample_order': [
            {'id': '63d6742d0adc5570001d2bbf7'},  # create order
            {
                'id': '63d6742d0adc5570001d2bbf7',
                'symbol': 'SOL-USDT',
                'opType': 'DEAL',
                'type': 'limit',
                'side': 'buy',
                'price': '15.5',
                'size': '1.1',
                'funds': '0',
                'dealFunds': '17.05',
                'dealSize': '1.1',
                'fee': '0.000065252',
                'feeCurrency': 'USDT',
                'stp': '',
                'stop': '',
                'stopTriggered': False,
                'stopPrice': '0',
                'timeInForce': 'GTC',
                'postOnly': False,
                'hidden': False,
                'iceberg': False,
                'visibleSize': '0',
                'cancelAfter': 0,
                'channel': 'API',
                'clientOid': '0a053870-11bf-41e5-be61-b272a4cb62e1',
                'remark': None,
                'tags': 'partner:ccxt',
                'isActive': False,
                'cancelExist': False,
                'createdAt': 1674493798550,
                'tradeType': 'TRADE'
            }],
    },
    'gate': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'futures': True,
        'futures_pair': 'BTC/USDT:USDT',
        'hasQuoteVolumeFutures': True,
        'leverage_tiers_public': True,
        'leverage_in_spot_market': True,
        'sample_order': [
            {
                "id": "276266139423",
                "text": "apiv4",
                "create_time": "1674493798",
                "update_time": "1674493798",
                "create_time_ms": "1674493798550",
                "update_time_ms": "1674493798550",
                "status": "closed",
                "currency_pair": "SOL_USDT",
                "type": "limit",
                "account": "spot",
                "side": "buy",
                "amount": "1.1",
                "price": "15.5",
                "time_in_force": "gtc",
                "iceberg": "0",
                "left": "0",
                "fill_price": "17.05",
                "filled_total": "17.05",
                "avg_deal_price": "15.5",
                "fee": "0.0000018",
                "fee_currency": "SOL",
                "point_fee": "0",
                "gt_fee": "0",
                "gt_maker_fee": "0",
                "gt_taker_fee": "0.0015",
                "gt_discount": True,
                "rebated_fee": "0",
                "rebated_fee_currency": "USDT"
            },
            {
                # market order
                'id': '276401180529',
                'text': 'apiv4',
                'create_time': '1674493798',
                'update_time': '1674493798',
                'create_time_ms': '1674493798550',
                'update_time_ms': '1674493798550',
                'status': 'cancelled',
                'currency_pair': 'SOL_USDT',
                'type': 'market',
                'account': 'spot',
                'side': 'buy',
                'amount': '17.05',
                'price': '0',
                'time_in_force': 'ioc',
                'iceberg': '0',
                'left': '0.0000000016228',
                'fill_price': '17.05',
                'filled_total': '17.05',
                'avg_deal_price': '15.5',
                'fee': '0',
                'fee_currency': 'SOL',
                'point_fee': '0.0199999999967544',
                'gt_fee': '0',
                'gt_maker_fee': '0',
                'gt_taker_fee': '0',
                'gt_discount': False,
                'rebated_fee': '0',
                'rebated_fee_currency': 'USDT'
            }
        ],
    },
    'okx': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'futures': True,
        'futures_pair': 'BTC/USDT:USDT',
        'hasQuoteVolumeFutures': False,
        'leverage_tiers_public': True,
        'leverage_in_spot_market': True,
        'private_methods': ['fetch_accounts'],
    },
    'bybit': {
        'pair': 'BTC/USDT',
        'stake_currency': 'USDT',
        'hasQuoteVolume': True,
        'use_ci_proxy': True,
        'timeframe': '1h',
        'futures_pair': 'BTC/USDT:USDT',
        'futures': True,
        'leverage_tiers_public': True,
        'leverage_in_spot_market': True,
        'sample_order': [
            {
                "orderId": "1274754916287346280",
                "orderLinkId": "1666798627015730",
                "symbol": "SOLUSDT",
                "createTime": "1674493798550",
                "orderPrice": "15.5",
                "orderQty": "1.1",
                "orderType": "LIMIT",
                "side": "BUY",
                "status": "NEW",
                "timeInForce": "GTC",
                "accountId": "5555555",
                "execQty": "0",
                "orderCategory": "0"
            }
        ]
    },
    'huobi': {
        'pair': 'ETH/BTC',
        'stake_currency': 'BTC',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'futures': False,
    },
    'bitvavo': {
        'pair': 'BTC/EUR',
        'stake_currency': 'EUR',
        'hasQuoteVolume': True,
        'timeframe': '1h',
        'leverage_tiers_public': False,
        'leverage_in_spot_market': False,
    },
}


@pytest.fixture(scope="class")
def exchange_conf():
    config = get_default_conf_usdt((Path(__file__).parent / "testdata").resolve())
    config['exchange']['pair_whitelist'] = []
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''
    config['dry_run'] = False
    config['entry_pricing']['use_order_book'] = True
    config['exit_pricing']['use_order_book'] = True
    return config


def set_test_proxy(config: Config, use_proxy: bool) -> Config:
    # Set proxy to test in CI.
    import os
    if use_proxy and (proxy := os.environ.get('CI_WEB_PROXY')):
        config1 = deepcopy(config)
        config1['exchange']['ccxt_config'] = {
            "aiohttp_proxy": proxy,
            'proxies': {
                'https': proxy,
                'http': proxy,
            }
        }
        return config1

    return config


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange(request, exchange_conf):
    exchange_conf = set_test_proxy(
        exchange_conf, EXCHANGES[request.param].get('use_ci_proxy', False))
    exchange_conf['exchange']['name'] = request.param
    exchange_conf['stake_currency'] = EXCHANGES[request.param]['stake_currency']
    exchange = ExchangeResolver.load_exchange(exchange_conf, validate=True)

    yield exchange, request.param


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange_futures(request, exchange_conf, class_mocker):
    if EXCHANGES[request.param].get('futures') is not True:
        yield None, request.param
    else:
        exchange_conf = set_test_proxy(
            exchange_conf, EXCHANGES[request.param].get('use_ci_proxy', False))
        exchange_conf = deepcopy(exchange_conf)
        exchange_conf['exchange']['name'] = request.param
        exchange_conf['trading_mode'] = 'futures'
        exchange_conf['margin_mode'] = 'isolated'
        exchange_conf['stake_currency'] = EXCHANGES[request.param]['stake_currency']

        class_mocker.patch(
            'freqtrade.exchange.binance.Binance.fill_leverage_tiers')
        class_mocker.patch(f'{EXMS}.fetch_trading_fees')
        class_mocker.patch('freqtrade.exchange.okx.Okx.additional_exchange_init')
        class_mocker.patch('freqtrade.exchange.binance.Binance.additional_exchange_init')
        class_mocker.patch('freqtrade.exchange.bybit.Bybit.additional_exchange_init')
        class_mocker.patch(f'{EXMS}.load_cached_leverage_tiers', return_value=None)
        class_mocker.patch(f'{EXMS}.cache_leverage_tiers')

        exchange = ExchangeResolver.load_exchange(
            exchange_conf, validate=True, load_leverage_tiers=True)

        yield exchange, request.param


@pytest.mark.longrun
class TestCCXTExchange():

    def test_load_markets(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        markets = exch.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exch.market_is_spot(markets[pair])

    def test_has_validations(self, exchange: EXCHANGE_FIXTURE_TYPE):

        exch, exchangename = exchange

        exch.validate_ordertypes({
            'entry': 'limit',
            'exit': 'limit',
            'stoploss': 'limit',
            })

        if exchangename == 'gate':
            # gate doesn't have market orders on spot
            return
        exch.validate_ordertypes({
            'entry': 'market',
            'exit': 'market',
            'stoploss': 'market',
            })

    def test_load_markets_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
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

    def test_ccxt_order_parse(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchange_name = exchange
        if orders := EXCHANGES[exchange_name].get('sample_order'):
            for order in orders:
                po = exch._api.parse_order(order)
                assert isinstance(po['id'], str)
                assert po['id'] is not None
                if len(order.keys()) < 5:
                    # Kucoin case
                    assert po['status'] == 'closed'
                    continue
                assert po['timestamp'] == 1674493798550
                assert isinstance(po['datetime'], str)
                assert isinstance(po['timestamp'], int)
                assert isinstance(po['price'], float)
                assert po['price'] == 15.5
                if po['average'] is not None:
                    assert isinstance(po['average'], float)
                    assert po['average'] == 15.5
                assert po['symbol'] == 'SOL/USDT'
                assert isinstance(po['amount'], float)
                assert po['amount'] == 1.1
                assert isinstance(po['status'], str)
        else:
            pytest.skip(f"No sample order available for exchange {exchange_name}")

    def test_ccxt_fetch_tickers(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        tickers = exch.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert tickers[pair]['ask'] is not None
        assert 'bid' in tickers[pair]
        assert tickers[pair]['bid'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_tickers_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange_futures
        if not exch or exchangename in ('gate'):
            # exchange_futures only returns values for supported exchanges
            return

        pair = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)

        tickers = exch.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert tickers[pair]['ask'] is not None
        assert 'bid' in tickers[pair]
        assert tickers[pair]['bid'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolumeFutures'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_ticker(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']

        ticker = exch.fetch_ticker(pair)
        assert 'ask' in ticker
        assert ticker['ask'] is not None
        assert 'bid' in ticker
        assert ticker['bid'] is not None
        assert 'quoteVolume' in ticker
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert ticker['quoteVolume'] is not None

    def test_ccxt_fetch_l2_orderbook(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        l2 = exch.fetch_l2_order_book(pair)
        assert 'asks' in l2
        assert 'bids' in l2
        assert len(l2['asks']) >= 1
        assert len(l2['bids']) >= 1
        l2_limit_range = exch._ft_has['l2_limit_range']
        l2_limit_range_required = exch._ft_has['l2_limit_range_required']
        if exchangename == 'gate':
            # TODO: Gate is unstable here at the moment, ignoring the limit partially.
            return
        for val in [1, 2, 5, 25, 50, 100]:
            if val > 50 and exchangename == 'bybit':
                continue
            l2 = exch.fetch_l2_order_book(pair, val)
            if not l2_limit_range or val in l2_limit_range:
                if val > 50:
                    # Orderbooks are not always this deep.
                    assert val - 5 < len(l2['asks']) <= val
                    assert val - 5 < len(l2['bids']) <= val
                else:
                    assert len(l2['asks']) == val
                    assert len(l2['bids']) == val
            else:
                next_limit = exch.get_next_limit_in_list(
                    val, l2_limit_range, l2_limit_range_required)
                if next_limit is None:
                    assert len(l2['asks']) > 100
                    assert len(l2['asks']) > 100
                elif next_limit > 200:
                    # Large orderbook sizes can be a problem for some exchanges (bitrex ...)
                    assert len(l2['asks']) > 200
                    assert len(l2['asks']) > 200
                else:
                    assert len(l2['asks']) == next_limit
                    assert len(l2['asks']) == next_limit

    def test_ccxt_fetch_ohlcv(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        timeframe = EXCHANGES[exchangename]['timeframe']

        pair_tf = (pair, timeframe, CandleType.SPOT)

        ohlcv = exch.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        # assert len(exch.klines(pair_tf)) > 200
        # Assume 90% uptime ...
        assert len(exch.klines(pair_tf)) > exch.ohlcv_candle_limit(
            timeframe, CandleType.SPOT) * 0.90
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(timezone.utc) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exch.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)

    def ccxt__async_get_candle_history(self, exchange, exchangename, pair, timeframe, candle_type):

        timeframe_ms = timeframe_to_msecs(timeframe)
        now = timeframe_to_prev_date(
                timeframe, datetime.now(timezone.utc))
        for offset in (360, 120, 30, 10, 5, 2):
            since = now - timedelta(days=offset)
            since_ms = int(since.timestamp() * 1000)

            res = exchange.loop.run_until_complete(exchange._async_get_candle_history(
                pair=pair,
                timeframe=timeframe,
                since_ms=since_ms,
                candle_type=candle_type
            )
            )
            assert res
            assert res[0] == pair
            assert res[1] == timeframe
            assert res[2] == candle_type
            candles = res[3]
            factor = 0.9
            candle_count = exchange.ohlcv_candle_limit(timeframe, candle_type, since_ms) * factor
            candle_count1 = (now.timestamp() * 1000 - since_ms) // timeframe_ms * factor
            assert len(candles) >= min(candle_count, candle_count1), \
                f"{len(candles)} < {candle_count} in {timeframe}, Offset: {offset} {factor}"
            assert candles[0][0] == since_ms or (since_ms + timeframe_ms)

    def test_ccxt__async_get_candle_history(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exc, exchangename = exchange
        if exchangename in ('bittrex'):
            # For some weired reason, this test returns random lengths for bittrex.
            pytest.skip("Exchange doesn't provide stable ohlcv history")

        if not exc._ft_has['ohlcv_has_history']:
            pytest.skip("Exchange does not support candle history")
        pair = EXCHANGES[exchangename]['pair']
        timeframe = EXCHANGES[exchangename]['timeframe']
        self.ccxt__async_get_candle_history(
            exc, exchangename, pair, timeframe, CandleType.SPOT)

    def test_ccxt__async_get_candle_history_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        timeframe = EXCHANGES[exchangename]['timeframe']
        self.ccxt__async_get_candle_history(
            exchange, exchangename, pair, timeframe, CandleType.FUTURES)

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
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

    def test_ccxt_fetch_mark_price_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
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

    def test_ccxt__calculate_funding_fees(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, exchangename = exchange_futures
        if not exchange:
            # exchange_futures only returns values for supported exchanges
            return
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since = datetime.now(timezone.utc) - timedelta(days=5)

        funding_fee = exchange._fetch_and_calculate_funding_fees(
            pair, 20, is_short=False, open_date=since)

        assert isinstance(funding_fee, float)
        # assert funding_fee > 0

    # TODO: tests fetch_trades (?)

    def test_ccxt_get_fee(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        threshold = 0.01
        assert 0 < exch.get_fee(pair, 'limit', 'buy') < threshold
        assert 0 < exch.get_fee(pair, 'limit', 'sell') < threshold
        assert 0 < exch.get_fee(pair, 'market', 'buy') < threshold
        assert 0 < exch.get_fee(pair, 'market', 'sell') < threshold

    def test_ccxt_get_max_leverage_spot(self, exchange: EXCHANGE_FIXTURE_TYPE):
        spot, spot_name = exchange
        if spot:
            leverage_in_market_spot = EXCHANGES[spot_name].get('leverage_in_spot_market')
            if leverage_in_market_spot:
                spot_pair = EXCHANGES[spot_name].get('pair', EXCHANGES[spot_name]['pair'])
                spot_leverage = spot.get_max_leverage(spot_pair, 20)
                assert (isinstance(spot_leverage, float) or isinstance(spot_leverage, int))
                assert spot_leverage >= 1.0

    def test_ccxt_get_max_leverage_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if futures:
            leverage_tiers_public = EXCHANGES[futures_name].get('leverage_tiers_public')
            if leverage_tiers_public:
                futures_pair = EXCHANGES[futures_name].get(
                    'futures_pair',
                    EXCHANGES[futures_name]['pair']
                )
                futures_leverage = futures.get_max_leverage(futures_pair, 20)
                assert (isinstance(futures_leverage, float) or isinstance(futures_leverage, int))
                assert futures_leverage >= 1.0

    def test_ccxt_get_contract_size(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if futures:
            futures_pair = EXCHANGES[futures_name].get(
                'futures_pair',
                EXCHANGES[futures_name]['pair']
            )
            contract_size = futures.get_contract_size(futures_pair)
            assert (isinstance(contract_size, float) or isinstance(contract_size, int))
            assert contract_size >= 0.0

    def test_ccxt_load_leverage_tiers(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if futures and EXCHANGES[futures_name].get('leverage_tiers_public'):
            leverage_tiers = futures.load_leverage_tiers()
            futures_pair = EXCHANGES[futures_name].get(
                'futures_pair',
                EXCHANGES[futures_name]['pair']
            )
            assert (isinstance(leverage_tiers, dict))
            assert futures_pair in leverage_tiers
            pair_tiers = leverage_tiers[futures_pair]
            assert len(pair_tiers) > 0
            oldLeverage = float('inf')
            oldMaintenanceMarginRate = oldminNotional = oldmaxNotional = -1
            for tier in pair_tiers:
                for key in [
                    'maintenanceMarginRate',
                    'minNotional',
                    'maxNotional',
                    'maxLeverage'
                ]:
                    assert key in tier
                    assert tier[key] >= 0.0
                assert tier['maxNotional'] > tier['minNotional']
                assert tier['maxLeverage'] <= oldLeverage
                assert tier['maintenanceMarginRate'] >= oldMaintenanceMarginRate
                assert tier['minNotional'] > oldminNotional
                assert tier['maxNotional'] > oldmaxNotional
                oldLeverage = tier['maxLeverage']
                oldMaintenanceMarginRate = tier['maintenanceMarginRate']
                oldminNotional = tier['minNotional']
                oldmaxNotional = tier['maxNotional']

    def test_ccxt_dry_run_liquidation_price(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if futures and EXCHANGES[futures_name].get('leverage_tiers_public'):

            futures_pair = EXCHANGES[futures_name].get(
                'futures_pair',
                EXCHANGES[futures_name]['pair']
            )

            liquidation_price = futures.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
            )
            assert (isinstance(liquidation_price, float))
            assert liquidation_price >= 0.0

            liquidation_price = futures.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
            )
            assert (isinstance(liquidation_price, float))
            assert liquidation_price >= 0.0

    def test_ccxt_get_max_pair_stake_amount(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if futures:
            futures_pair = EXCHANGES[futures_name].get(
                'futures_pair',
                EXCHANGES[futures_name]['pair']
            )
            max_stake_amount = futures.get_max_pair_stake_amount(futures_pair, 40000)
            assert (isinstance(max_stake_amount, float))
            assert max_stake_amount >= 0.0

    def test_private_method_presence(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        for method in EXCHANGES[exchangename].get('private_methods', []):
            assert hasattr(exch._api, method)
