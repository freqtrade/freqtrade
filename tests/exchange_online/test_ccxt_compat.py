"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""

from datetime import UTC, datetime, timedelta

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange import Exchange, timeframe_to_msecs
from freqtrade.util import dt_floor_day, dt_now, dt_ts
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE


@pytest.mark.longrun
class TestCCXTExchange:
    def test_load_markets(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]
        markets = exch.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exch.market_is_spot(markets[pair])

    def test_has_validations(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, _ = exchange

        exch.validate_ordertypes(
            {
                "entry": "limit",
                "exit": "limit",
                "stoploss": "limit",
            }
        )

        if exchangename == "gate":
            # gate doesn't have market orders on spot
            return
        exch.validate_ordertypes(
            {
                "entry": "market",
                "exit": "market",
                "stoploss": "market",
            }
        )

    def test_ohlcv_limit(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        expected_count = exchange_params.get("candle_count")
        if not expected_count:
            pytest.skip("No expected candle count for exchange")

        assert exch.ohlcv_candle_limit("1m", CandleType.SPOT) == expected_count

    def test_ohlcv_limit_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange_futures
        expected_count = exchange_params.get(
            "candle_count_futures", exchange_params.get("candle_count")
        )
        if not expected_count:
            pytest.skip("No expected candle count for exchange")

        assert exch.ohlcv_candle_limit("1m", CandleType.FUTURES) == expected_count

    def test_load_markets_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, _, exchange_params = exchange_futures
        pair = exchange_params["pair"]
        pair1 = exchange_params.get("futures_pair", pair)
        alternative_pairs = exchange_params.get("futures_alt_pairs", [])
        markets = exchange.markets
        for pair in [pair1] + alternative_pairs:
            assert pair in markets, f"Futures pair {pair} not found in markets"
            assert isinstance(markets[pair], dict)

            assert exchange.market_is_future(markets[pair])

    def test_ccxt_order_parse(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, exchange_params = exchange
        if orders := exchange_params.get("sample_order"):
            for order in orders:
                pair = order["pair"]
                exchange_response: dict = order["exchange_response"]

                market = exch._api.markets[pair]
                po = exch._api.parse_order(exchange_response, market)
                expected = order["expected"]
                assert isinstance(po["id"], str)
                assert po["id"] is not None
                if len(exchange_response.keys()) < 5:
                    # Kucoin case
                    assert po["status"] is None
                    continue
                assert po["timestamp"] == expected["timestamp"]
                assert isinstance(po["datetime"], str)
                assert isinstance(po["timestamp"], int)
                assert isinstance(po["price"], float)
                assert po["price"] == expected["price"]
                if po["status"] == "closed":
                    # Filled orders should have average assigned.
                    assert isinstance(po["average"], float)
                    assert po["average"] == 15.5
                assert po["symbol"] == pair
                assert isinstance(po["amount"], float)
                assert po["amount"] == expected["amount"]
                assert isinstance(po["status"], str)
        else:
            pytest.skip(f"No sample order available for exchange {exchangename}")

    def test_ccxt_my_trades_parse(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, exchange_params = exchange
        if trades := exchange_params.get("sample_my_trades"):
            pair = "SOL/USDT"
            for trade in trades:
                po = exch._api.parse_trade(trade)
                assert po["symbol"] == pair
                assert isinstance(po["id"], str)
                assert isinstance(po["side"], str)
                assert isinstance(po["amount"], float)
                assert isinstance(po["price"], float)
                assert isinstance(po["datetime"], str)
                assert isinstance(po["timestamp"], int)

                if fees := po.get("fees"):
                    assert isinstance(fees, list)
                    for fee in fees:
                        assert isinstance(fee, dict)
                        assert isinstance(fee["cost"], float)
                        assert isinstance(fee["currency"], str)

        else:
            pytest.skip(f"No sample Trades available for exchange {exchangename}")

    def test_ccxt_balances_parse(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, exchange_params = exchange
        if balance_response := exchange_params.get("sample_balances"):
            balances = exch._api.parse_balance(balance_response["exchange_response"])
            expected = balance_response["expected"]
            for currency, balance in expected.items():
                assert currency in balances
                assert isinstance(balance, dict)
                assert balance == balances[currency]
            pass
        else:
            pytest.skip(f"No sample Balances available for exchange {exchangename}")

    def test_ccxt_fetch_tickers(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]

        tickers = exch.get_tickers()
        assert pair in tickers
        assert "ask" in tickers[pair]
        assert "bid" in tickers[pair]
        if exchange_params.get("tickers_have_bid_ask"):
            assert tickers[pair]["bid"] is not None
            assert tickers[pair]["ask"] is not None
        assert "quoteVolume" in tickers[pair]
        if exchange_params.get("hasQuoteVolume"):
            assert tickers[pair]["quoteVolume"] is not None

    def test_ccxt_fetch_tickers_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, exchange_params = exchange_futures
        if not exch or exchangename in ("gate"):
            # exchange_futures only returns values for supported exchanges
            return

        pair = exchange_params["pair"]
        pair = exchange_params.get("futures_pair", pair)

        tickers = exch.get_tickers()
        assert pair in tickers
        assert "ask" in tickers[pair]
        assert tickers[pair]["ask"] is not None
        assert "bid" in tickers[pair]
        assert tickers[pair]["bid"] is not None
        assert "quoteVolume" in tickers[pair]
        if exchange_params.get("hasQuoteVolumeFutures"):
            assert tickers[pair]["quoteVolume"] is not None

    def test_ccxt_fetch_ticker(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]

        ticker = exch.fetch_ticker(pair)
        assert "ask" in ticker
        assert "bid" in ticker
        if exchange_params.get("tickers_have_bid_ask"):
            assert ticker["ask"] is not None
            assert ticker["bid"] is not None
        assert "quoteVolume" in ticker
        if exchange_params.get("hasQuoteVolume"):
            assert ticker["quoteVolume"] is not None

    def test_ccxt_fetch_l2_orderbook(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, exchange_params = exchange
        pair = exchange_params["pair"]
        l2 = exch.fetch_l2_order_book(pair)
        orderbook_max_entries = exchange_params.get("orderbook_max_entries")
        assert "asks" in l2
        assert "bids" in l2
        assert len(l2["asks"]) >= 1
        assert len(l2["bids"]) >= 1
        l2_limit_range = exch._ft_has["l2_limit_range"]
        l2_limit_range_required = exch._ft_has["l2_limit_range_required"]
        if exchangename == "gate":
            # TODO: Gate is unstable here at the moment, ignoring the limit partially.
            return
        for val in [1, 2, 5, 25, 50, 100]:
            if orderbook_max_entries and val > orderbook_max_entries:
                continue
            l2 = exch.fetch_l2_order_book(pair, val)
            if not l2_limit_range or val in l2_limit_range:
                if val > 50:
                    # Orderbooks are not always this deep.
                    assert val - 5 < len(l2["asks"]) <= val
                    assert val - 5 < len(l2["bids"]) <= val
                else:
                    assert len(l2["asks"]) == val
                    assert len(l2["bids"]) == val
            else:
                next_limit = exch.get_next_limit_in_list(
                    val, l2_limit_range, l2_limit_range_required
                )
                if next_limit is None:
                    assert len(l2["asks"]) > 100
                    assert len(l2["asks"]) > 100
                elif next_limit > 200:
                    # Large orderbook sizes can be a problem for some exchanges (bitrex ...)
                    assert len(l2["asks"]) > 200
                    assert len(l2["asks"]) > 200
                else:
                    assert len(l2["asks"]) == next_limit
                    assert len(l2["asks"]) == next_limit

    def test_ccxt_fetch_ohlcv(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]
        timeframe = exchange_params["timeframe"]

        pair_tf = (pair, timeframe, CandleType.SPOT)

        ohlcv = exch.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        # assert len(exch.klines(pair_tf)) > 200
        # Assume 90% uptime ...
        assert (
            len(exch.klines(pair_tf)) > exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) * 0.90
        )
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(UTC) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exch.klines(pair_tf).iloc[-1]["date"] >= timeframe_to_prev_date(timeframe, now)

    def test_ccxt_fetch_ohlcv_startdate(self, exchange: EXCHANGE_FIXTURE_TYPE):
        """
        Test that pair data starts at the provided startdate
        """
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]
        timeframe = "1d"

        pair_tf = (pair, timeframe, CandleType.SPOT)
        # last 5 days ...
        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=6))
        ohlcv = exch.refresh_latest_ohlcv([pair_tf], since_ms=since_ms)
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        # Check if last-timeframe is within the last 2 intervals
        now = datetime.now(UTC) - timedelta(minutes=(timeframe_to_minutes(timeframe) * 2))
        assert exch.klines(pair_tf).iloc[-1]["date"] >= timeframe_to_prev_date(timeframe, now)
        assert exch.klines(pair_tf)["date"].astype(int).iloc[0] // 1e6 == since_ms

    def _ccxt__async_get_candle_history(
        self, exchange, pair: str, timeframe: str, candle_type: CandleType, factor: float = 0.9
    ):
        timeframe_ms = timeframe_to_msecs(timeframe)
        timeframe_ms_8h = timeframe_to_msecs("8h")
        now = timeframe_to_prev_date(timeframe, datetime.now(UTC))
        for offset_days in (360, 120, 30, 10, 5, 2):
            since = now - timedelta(days=offset_days)
            since_ms = int(since.timestamp() * 1000)

            res = exchange.loop.run_until_complete(
                exchange._async_get_candle_history(
                    pair=pair, timeframe=timeframe, since_ms=since_ms, candle_type=candle_type
                )
            )
            assert res
            assert res[0] == pair
            assert res[1] == timeframe
            assert res[2] == candle_type
            candles = res[3]
            candle_count = exchange.ohlcv_candle_limit(timeframe, candle_type, since_ms) * factor
            candle_count1 = (now.timestamp() * 1000 - since_ms) // timeframe_ms * factor
            # funding fees can be 1h or 8h - depending on pair and time.
            candle_count2 = (now.timestamp() * 1000 - since_ms) // timeframe_ms_8h * factor
            min_value = min(
                candle_count,
                candle_count1,
                candle_count2 if candle_type == CandleType.FUNDING_RATE else candle_count1,
            )
            assert len(candles) >= min_value, (
                f"{len(candles)} < {candle_count} in {timeframe} {offset_days=} {factor=}"
            )
            # Check if first-timeframe is either the start, or start + 1
            assert candles[0][0] == since_ms or (since_ms + timeframe_ms)

    def test_ccxt__async_get_candle_history(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exc, _, exchange_params = exchange

        if not exc._ft_has["ohlcv_has_history"]:
            pytest.skip("Exchange does not support candle history")
        pair = exchange_params["pair"]
        timeframe = exchange_params["timeframe"]
        self._ccxt__async_get_candle_history(exc, pair, timeframe, CandleType.SPOT)

    @pytest.mark.parametrize(
        "candle_type",
        [
            CandleType.FUTURES,
            CandleType.FUNDING_RATE,
            CandleType.INDEX,
            CandleType.PREMIUMINDEX,
            CandleType.MARK,
        ],
    )
    def test_ccxt__async_get_candle_history_futures(
        self, exchange_futures: EXCHANGE_FIXTURE_TYPE, candle_type: CandleType
    ):
        exchange, _, exchange_params = exchange_futures
        pair = exchange_params.get("futures_pair", exchange_params["pair"])
        timeframe = exchange_params["timeframe"]
        if candle_type == CandleType.FUNDING_RATE:
            timeframe = exchange._ft_has.get(
                "funding_fee_timeframe", exchange._ft_has["mark_ohlcv_timeframe"]
            )
        else:
            # never skip funding rate!
            if not exchange.check_candle_type_support(candle_type):
                pytest.skip(f"Exchange does not support candle type {candle_type}")
        self._ccxt__async_get_candle_history(
            exchange,
            pair=pair,
            timeframe=timeframe,
            candle_type=candle_type,
        )

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, _, exchange_params = exchange_futures

        pair = exchange_params.get("futures_pair", exchange_params["pair"])
        since = int((datetime.now(UTC) - timedelta(days=5)).timestamp() * 1000)
        timeframe_ff = exchange._ft_has.get(
            "funding_fee_timeframe", exchange._ft_has["mark_ohlcv_timeframe"]
        )
        timeframe_ff_8h = "8h"
        pair_tf = (pair, timeframe_ff, CandleType.FUNDING_RATE)

        funding_ohlcv = exchange.refresh_latest_ohlcv(
            [pair_tf], since_ms=since, drop_incomplete=False
        )

        assert isinstance(funding_ohlcv, dict)
        rate = funding_ohlcv[pair_tf]

        this_hour = timeframe_to_prev_date(timeframe_ff)
        hour1 = timeframe_to_prev_date(timeframe_ff, this_hour - timedelta(minutes=1))
        hour2 = timeframe_to_prev_date(timeframe_ff, hour1 - timedelta(minutes=1))
        hour3 = timeframe_to_prev_date(timeframe_ff, hour2 - timedelta(minutes=1))
        # Alternative 8h timeframe - funding fee timeframe is not stable.
        h8_this_hour = timeframe_to_prev_date(timeframe_ff_8h)
        h8_hour1 = timeframe_to_prev_date(timeframe_ff_8h, h8_this_hour - timedelta(minutes=1))
        h8_hour2 = timeframe_to_prev_date(timeframe_ff_8h, h8_hour1 - timedelta(minutes=1))
        h8_hour3 = timeframe_to_prev_date(timeframe_ff_8h, h8_hour2 - timedelta(minutes=1))
        row0 = rate.iloc[-1]
        row1 = rate.iloc[-2]
        row2 = rate.iloc[-3]
        row3 = rate.iloc[-4]

        assert row0["date"] == this_hour or row0["date"] == h8_this_hour
        assert row1["date"] == hour1 or row1["date"] == h8_hour1
        assert row2["date"] == hour2 or row2["date"] == h8_hour2
        assert row3["date"] == hour3 or row3["date"] == h8_hour3

        # Test For last 4 hours
        # Avoids random test-failure when funding-fees are 0 for a few hours.
        assert (
            row0["open"] != 0.0 or row1["open"] != 0.0 or row2["open"] != 0.0 or row3["open"] != 0.0
        )
        # We expect funding rates to be different from 0.0 - or moving around.
        assert (
            rate["open"].max() != 0.0
            or rate["open"].min() != 0.0
            or (rate["open"].min() != rate["open"].max())
        )

    def test_ccxt_fetch_mark_price_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, _, exchange_params = exchange_futures
        pair = exchange_params.get("futures_pair", exchange_params["pair"])
        since = int((datetime.now(UTC) - timedelta(days=5)).timestamp() * 1000)
        candle_type = CandleType.from_string(
            exchange.get_option("mark_ohlcv_price", default=CandleType.MARK)
        )
        pair_tf = (pair, "1h", candle_type)

        mark_ohlcv = exchange.refresh_latest_ohlcv([pair_tf], since_ms=since, drop_incomplete=False)

        assert isinstance(mark_ohlcv, dict)
        expected_tf = "1h"
        mark_candles = mark_ohlcv[pair_tf]

        this_hour = timeframe_to_prev_date(expected_tf)
        prev_hour = timeframe_to_prev_date(expected_tf, this_hour - timedelta(minutes=1))

        # Mark price must be available for the currently open candle (as well as older candles,
        # even though the test only asserts the last two).
        # This is a requirement to have funding fee calculations available correctly and timely
        # right as the funding fee applies (e.g. at 08:00).
        assert mark_candles[mark_candles["date"] == prev_hour].iloc[0]["open"] != 0.0
        assert mark_candles[mark_candles["date"] == this_hour].iloc[0]["open"] != 0.0

    def test_ccxt__calculate_funding_fees(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exchange, _, exchange_params = exchange_futures
        pair = exchange_params.get("futures_pair", exchange_params["pair"])
        since = datetime.now(UTC) - timedelta(days=5)

        funding_fee = exchange._fetch_and_calculate_funding_fees(
            pair, 20, is_short=False, open_date=since
        )

        assert isinstance(funding_fee, float)
        assert funding_fee != 0

    def test_ccxt__async_get_trade_history(self, exchange: EXCHANGE_FIXTURE_TYPE, mocker):
        exch, exchangename, exchange_params = exchange
        if not (lookback := exchange_params.get("trades_lookback_hours")):
            pytest.skip("test_fetch_trades not enabled for this exchange")
        pair = exchange_params["pair"]
        since = int((datetime.now(UTC) - timedelta(hours=lookback)).timestamp() * 1000)
        nvspy = mocker.spy(exch, "_get_trade_pagination_next_value")
        res = exch.loop.run_until_complete(exch._async_get_trade_history(pair, since, None, None))
        assert len(res) == 2
        res_pair, res_trades = res
        assert res_pair == pair
        assert isinstance(res_trades, list)
        assert res_trades[0][0] >= since
        assert len(res_trades) > 1200
        assert nvspy.call_count > 5
        if exchangename == "kraken":
            # for Kraken, the pagination value is added to the last trade result by ccxt.
            # We therefore expect that the last row has one additional field

            # Pick a random spy call
            trades_orig = nvspy.call_args_list[2][0][0]
            assert len(trades_orig[-1].get("info")) > len(trades_orig[-2].get("info"))

    def _ccxt_get_fee(self, exch: Exchange, pair: str):
        threshold = 0.01
        assert 0 < exch.get_fee(pair, "limit", "buy") < threshold
        assert 0 < exch.get_fee(pair, "limit", "sell") < threshold
        assert 0 < exch.get_fee(pair, "market", "buy") < threshold
        assert 0 < exch.get_fee(pair, "market", "sell") < threshold

    def test_ccxt_get_fee_spot(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        pair = exchange_params["pair"]
        self._ccxt_get_fee(exch, pair)

    def test_ccxt_get_fee_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange_futures
        pair = exchange_params.get("futures_pair", exchange_params["pair"])
        self._ccxt_get_fee(exch, pair)

    def test_ccxt_get_max_leverage_spot(self, exchange: EXCHANGE_FIXTURE_TYPE):
        spot, _, exchange_params = exchange
        if spot:
            leverage_in_market_spot = exchange_params.get("leverage_in_spot_market")
            if leverage_in_market_spot:
                spot_pair = exchange_params.get("pair", exchange_params["pair"])
                spot_leverage = spot.get_max_leverage(spot_pair, 20)
                assert isinstance(spot_leverage, float) or isinstance(spot_leverage, int)
                assert spot_leverage >= 1.0

    def test_ccxt_get_max_leverage_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, _, exchange_params = exchange_futures
        leverage_tiers_public = exchange_params.get("leverage_tiers_public")
        if leverage_tiers_public:
            futures_pair = exchange_params.get("futures_pair", exchange_params["pair"])
            futures_leverage = futures.get_max_leverage(futures_pair, 20)
            assert isinstance(futures_leverage, float) or isinstance(futures_leverage, int)
            assert futures_leverage >= 1.0

    def test_ccxt_get_contract_size(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, _, exchange_params = exchange_futures
        futures_pair = exchange_params.get("futures_pair", exchange_params["pair"])
        contract_size = futures.get_contract_size(futures_pair)
        assert isinstance(contract_size, float) or isinstance(contract_size, int)
        assert contract_size >= 0.0

    def test_ccxt_load_leverage_tiers(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, _, exchange_params = exchange_futures
        if exchange_params.get("leverage_tiers_public"):
            leverage_tiers = futures.load_leverage_tiers()
            futures_pair = exchange_params.get("futures_pair", exchange_params["pair"])
            assert isinstance(leverage_tiers, dict)
            assert futures_pair in leverage_tiers
            pair_tiers = leverage_tiers[futures_pair]
            assert len(pair_tiers) > 0
            oldLeverage = float("inf")
            oldMaintenanceMarginRate = oldminNotional = oldmaxNotional = -1
            for tier in pair_tiers:
                for key in ["maintenanceMarginRate", "minNotional", "maxNotional", "maxLeverage"]:
                    assert key in tier
                    # maxNotional can be None (no limit)
                    assert tier[key] is None or tier[key] >= 0.0
                assert tier["maxNotional"] is None or tier["maxNotional"] > tier["minNotional"]
                assert tier["maxLeverage"] <= oldLeverage
                assert tier["maintenanceMarginRate"] >= oldMaintenanceMarginRate
                assert tier["minNotional"] > oldminNotional
                assert tier["maxNotional"] is None or tier["maxNotional"] > oldmaxNotional
                oldLeverage = tier["maxLeverage"]
                oldMaintenanceMarginRate = tier["maintenanceMarginRate"]
                oldminNotional = tier["minNotional"]
                oldmaxNotional = tier["maxNotional"]

    def test_ccxt_dry_run_liquidation_price(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, _, exchange_params = exchange_futures
        if exchange_params.get("leverage_tiers_public"):
            futures_pair = exchange_params.get("futures_pair", exchange_params["pair"])

            liquidation_price = futures.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
                open_trades=[],
            )
            assert isinstance(liquidation_price, float)
            assert liquidation_price >= 0.0

            liquidation_price = futures.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
                open_trades=[],
            )
            assert isinstance(liquidation_price, float)
            assert liquidation_price >= 0.0

    def test_ccxt_get_max_pair_stake_amount(self, exchange_futures: EXCHANGE_FIXTURE_TYPE):
        futures, _, exchange_params = exchange_futures
        futures_pair = exchange_params.get("futures_pair", exchange_params["pair"])
        max_stake_amount = futures.get_max_pair_stake_amount(futures_pair, 40000)
        assert isinstance(max_stake_amount, float)
        assert max_stake_amount >= 0.0

    def test_private_method_presence(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, _, exchange_params = exchange
        for method in exchange_params.get("private_methods", []):
            assert hasattr(exch._api, method)

    def test_ccxt_bitget_ohlcv_candle_limit(self, exchange: EXCHANGE_FIXTURE_TYPE):
        exch, exchangename, _ = exchange
        if exchangename != "bitget":
            pytest.skip("This test is only for the Bitget exchange")

        timeframes = ("1m", "5m", "1h")

        for timeframe in timeframes:
            assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 200

            start_time = dt_ts(dt_now() - timedelta(days=17))
            assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
            start_time = dt_ts(dt_now() - timedelta(days=48))
            length = 200 if timeframe in ("1m", "5m") else 1000
            assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200

            start_time = dt_ts(dt_now() - timedelta(days=61))
            length = 200
            assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
            assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
