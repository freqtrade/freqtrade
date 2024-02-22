"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""

from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange import timeframe_to_msecs
from freqtrade.util import dt_floor_day, dt_now, dt_ts
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE, EXCHANGES


@pytest.mark.longrun
class TestCCXTExchange:

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
        pair = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        markets = exchange.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)

        assert exchange.market_is_future(markets[pair])

    def test_ccxt_order_parse(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchange_name = exchange
        if orders := EXCHANGES[exchange_name].get('sample_order'):
            pair = 'SOL/USDT'
            for order in orders:
                market = exch._api.markets[pair]
                po = exch._api.parse_order(order, market)
                assert isinstance(po['id'], str)
                assert po['id'] is not None
                if len(order.keys()) < 5:
                    # Kucoin case
                    assert po['status'] is None
                    continue
                assert po['timestamp'] == 1674493798550
                assert isinstance(po['datetime'], str)
                assert isinstance(po['timestamp'], int)
                assert isinstance(po['price'], float)
                assert po['price'] == 15.5
                if po['average'] is not None:
                    assert isinstance(po['average'], float)
                    assert po['average'] == 15.5
                assert po['symbol'] == pair
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
        orderbook_max_entries = EXCHANGES[exchangename].get('orderbook_max_entries')
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
            if orderbook_max_entries and val > orderbook_max_entries:
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

    def test_ccxt_fetch_ohlcv_startdate(self, exchange: EXCHANGE_FIXTURE_TYPE):
        """
        Test that pair data starts at the provided startdate
        """
        exch, exchangename = exchange
        pair = EXCHANGES[exchangename]['pair']
        timeframe = '1d'

        pair_tf = (pair, timeframe, CandleType.SPOT)
        # last 5 days ...
        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=6))
        ohlcv = exch.refresh_latest_ohlcv([pair_tf], since_ms=since_ms)
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(timezone.utc) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exch.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)
        assert exch.klines(pair_tf)['date'].astype(int).iloc[0] // 1e6 == since_ms

    def ccxt__async_get_candle_history(
            self, exchange, exchangename, pair, timeframe, candle_type, factor=0.9):

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
            candle_count = exchange.ohlcv_candle_limit(timeframe, candle_type, since_ms) * factor
            candle_count1 = (now.timestamp() * 1000 - since_ms) // timeframe_ms * factor
            assert len(candles) >= min(candle_count, candle_count1), \
                f"{len(candles)} < {candle_count} in {timeframe}, Offset: {offset} {factor}"
            # Check if first-timeframe is either the start, or start + 1
            assert candles[0][0] == since_ms or (since_ms + timeframe_ms)

    def test_ccxt__async_get_candle_history(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exc, exchangename = exchange

        if not exc._ft_has['ohlcv_has_history']:
            pytest.skip("Exchange does not support candle history")
        pair = EXCHANGES[exchangename]['pair']
        timeframe = EXCHANGES[exchangename]['timeframe']
        self.ccxt__async_get_candle_history(
            exc, exchangename, pair, timeframe, CandleType.SPOT)

    @pytest.mark.parametrize('candle_type', [
        CandleType.FUTURES,
        CandleType.FUNDING_RATE,
        CandleType.MARK,
        ])
    def test_ccxt__async_get_candle_history_futures(
            self, exchange_futures: EXCHANGE_FIXTURE_TYPE, candle_type):
        exchange, exchangename = exchange_futures
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        timeframe = EXCHANGES[exchangename]['timeframe']
        if candle_type == CandleType.FUNDING_RATE:
            timeframe = exchange._ft_has.get('funding_fee_timeframe',
                                             exchange._ft_has['mark_ohlcv_timeframe'])
        self.ccxt__async_get_candle_history(
            exchange,
            exchangename,
            pair=pair,
            timeframe=timeframe,
            candle_type=candle_type,
        )

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, exchangename = exchange_futures

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
        pair = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since = datetime.now(timezone.utc) - timedelta(days=5)

        funding_fee = exchange._fetch_and_calculate_funding_fees(
            pair, 20, is_short=False, open_date=since)

        assert isinstance(funding_fee, float)
        # assert funding_fee > 0

    def test_ccxt__async_get_trade_history(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename = exchange
        if not (lookback := EXCHANGES[exchangename].get('trades_lookback_hours')):
            pytest.skip('test_fetch_trades not enabled for this exchange')
        pair = EXCHANGES[exchangename]['pair']
        since = int((datetime.now(timezone.utc) - timedelta(hours=lookback)).timestamp() * 1000)
        res = exch.loop.run_until_complete(
            exch._async_get_trade_history(pair, since, None, None)
        )
        assert len(res) == 2
        res_pair, res_trades = res
        assert res_pair == pair
        assert isinstance(res_trades, list)
        assert res_trades[0][0] >= since
        assert len(res_trades) > 1200

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
        futures_pair = EXCHANGES[futures_name].get(
            'futures_pair',
            EXCHANGES[futures_name]['pair']
        )
        contract_size = futures.get_contract_size(futures_pair)
        assert (isinstance(contract_size, float) or isinstance(contract_size, int))
        assert contract_size >= 0.0

    def test_ccxt_load_leverage_tiers(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, futures_name = exchange_futures
        if EXCHANGES[futures_name].get('leverage_tiers_public'):
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
        if EXCHANGES[futures_name].get('leverage_tiers_public'):

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
