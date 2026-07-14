"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests aim to test ccxt compatibility, specifically regarding websockets.
"""

import logging
from datetime import timedelta
from time import sleep

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_utils import timeframe_to_prev_date
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import log_has_re
from tests.exchange_online.conftest import EXCHANGE_WS_FIXTURE_TYPE


@pytest.mark.longrun
@pytest.mark.timeout(3 * 60)
class TestCCXTExchangeWs:
    def test_ccxt_watch_ohlcv(self, exchange_ws: EXCHANGE_WS_FIXTURE_TYPE, caplog, mocker):
        exch, _exchangename, pair = exchange_ws

        assert exch._ws_async is not None
        timeframe = "1m"
        pair_tf = (pair, timeframe, CandleType.SPOT)
        m_hist = mocker.spy(exch, "_async_get_historic_ohlcv")
        m_cand = mocker.spy(exch, "_async_get_candle_history")

        while True:
            # Don't start the test if we are too close to the end of the minute.
            if dt_now().second < 50 and dt_now().second > 1:
                break
            sleep(1)

        caplog.set_level(logging.DEBUG)
        res = exch.refresh_latest_ohlcv([pair_tf])
        assert m_cand.call_count == 1

        # Currently open candle
        next_candle = timeframe_to_prev_date(timeframe, dt_now())
        # Currently closed candle
        curr_candle = timeframe_to_prev_date(timeframe, next_candle - timedelta(seconds=1))

        assert pair_tf in exch._exchange_ws._klines_watching
        assert pair_tf in exch._exchange_ws._klines_scheduled
        assert res[pair_tf] is not None
        df1 = res[pair_tf]
        assert df1.iloc[-1]["date"] == curr_candle, (
            f"Expected {curr_candle}, got {df1.iloc[-1]['date']} for {pair_tf}, now: {dt_now()}"
        )

        # Wait until the next candle (might be up to 1 minute).
        while True:
            caplog.clear()
            res = exch.refresh_latest_ohlcv([pair_tf])
            df2 = res[pair_tf]
            assert df2 is not None
            if df2.iloc[-1]["date"] == next_candle:
                break
            assert df2.iloc[-1]["date"] == curr_candle
            sleep(1)

        assert m_hist.call_count == 0
        # shouldn't have tried fetch_ohlcv a second time.
        assert m_cand.call_count == 1
        assert log_has_re(r"watch result.*", caplog)

    def test_ccxt_watch_orderbook(self, exchange_ws: EXCHANGE_WS_FIXTURE_TYPE, caplog, mocker):
        exch, _exchangename, pair = exchange_ws

        assert exch._ws_async is not None
        if not exch._has_watch_orderbook:
            pytest.skip(f"{_exchangename} does not support watch_order_book.")

        # Spy on the REST fallback - it must stop being called once the ws cache is warm.
        m_rest = mocker.spy(exch._api, "fetch_l2_order_book")

        # First call schedules the websocket subscription. The cache is still cold,
        # so this call is served from the REST endpoint.
        ob = exch.fetch_l2_order_book(pair)
        assert ob is not None
        assert pair in exch._exchange_ws._ob_watching

        # Wait for the websocket to populate the local orderbook cache with actual levels.
        # (ccxt.pro creates the book object immediately on watch but it's empty)
        # time-limited by the class-level @pytest.mark.timeout.
        while True:
            cached = exch._exchange_ws.get_orderbook(pair)
            if cached.get("bids") and cached.get("asks"):
                break
            sleep(1)

        # Now that the cache is warm, further calls must be served from the websocket
        # cache without hitting the REST endpoint again.
        m_rest.reset_mock()
        caplog.clear()
        ob = exch.fetch_l2_order_book(pair)

        assert m_rest.call_count == 0
        assert log_has_re(r"using orderbook for .*", caplog)

        # Validate the returned orderbook structure.
        assert ob is not None
        assert ob["bids"] and ob["asks"]
        # Best bid must be below the best ask.
        assert ob["bids"][0][0] < ob["asks"][0][0]
        # Bids are sorted descending, asks ascending.
        assert ob["bids"][0][0] >= ob["bids"][-1][0]
        assert ob["asks"][0][0] <= ob["asks"][-1][0]
