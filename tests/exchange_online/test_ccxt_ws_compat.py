"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests aim to test ccxt compatibility, specifically regarding websockets.
"""

import logging
from datetime import timedelta

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_utils import (timeframe_to_minutes, timeframe_to_next_date,
                                               timeframe_to_prev_date)
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import log_has_re
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE, EXCHANGES


@pytest.mark.longrun
class TestCCXTExchangeWs:

    def test_ccxt_ohlcv(self, exchange_ws: EXCHANGE_FIXTURE_TYPE, caplog):
        exch, exchangename = exchange_ws

        assert exch._ws_async is not None
        pair = EXCHANGES[exchangename]['pair']
        timeframe = '1m'
        pair_tf = (pair, timeframe, CandleType.SPOT)

        res = exch.refresh_latest_ohlcv([pair_tf])
        now = dt_now() - timedelta(minutes=(timeframe_to_minutes(timeframe) * 1.1))
        # Currently closed candle
        curr_candle = timeframe_to_prev_date(timeframe, now)
        # Currently open candle
        next_candle = timeframe_to_next_date(timeframe, now)
        assert pair_tf in exch._exchange_ws._klines_watching
        assert pair_tf in exch._exchange_ws._klines_scheduled
        assert res[pair_tf] is not None
        df1 = res[pair_tf]
        caplog.set_level(logging.DEBUG)
        assert df1.iloc[-1]['date'] == curr_candle

        # Wait until the next candle (might be up to 1 minute).
        while True:
            caplog.clear()
            res = exch.refresh_latest_ohlcv([pair_tf])
            df2 = res[pair_tf]
            assert df2 is not None
            if df2.iloc[-1]['date'] == next_candle:
                break
            assert df2.iloc[-1]['date'] == curr_candle

        assert log_has_re(r"watch result.*", caplog)

