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
from freqtrade.loggers.set_log_levels import set_loggers
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

        res = exch.refresh_latest_ohlcv([pair_tf])
        assert m_cand.call_count == 1

        # Currently open candle
        next_candle = timeframe_to_prev_date(timeframe, dt_now())
        now = next_candle - timedelta(seconds=1)
        # Currently closed candle
        curr_candle = timeframe_to_prev_date(timeframe, now)

        assert pair_tf in exch._exchange_ws._klines_watching
        assert pair_tf in exch._exchange_ws._klines_scheduled
        assert res[pair_tf] is not None
        df1 = res[pair_tf]
        caplog.set_level(logging.DEBUG)
        set_loggers(1)
        assert df1.iloc[-1]["date"] == curr_candle

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
