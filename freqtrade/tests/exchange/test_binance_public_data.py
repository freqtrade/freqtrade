import asyncio
import datetime
import io
import re
import sys
import zipfile
from datetime import timedelta

import aiohttp
import pandas as pd
import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange.binance_public_data import (
    BadHttpStatus,
    Http404,
    binance_vision_trades_zip_url,
    binance_vision_zip_name,
    download_archive_ohlcv,
    download_archive_trades,
    get_daily_ohlcv,
    get_daily_trades,
)
from freqtrade.util.datetime_helpers import dt_ts, dt_utc
from ft_client.test_client.test_rest_client import log_has_re


@pytest.fixture(scope="module")
def event_loop_policy(request):
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    else:
        return asyncio.DefaultEventLoopPolicy()


class MockResponse:
    """AioHTTP response mock"""

    def __init__(self, content, status, reason=""):
        self._content = content
        self.status = status
        self.reason = reason

    async def read(self):
        return self._content

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


# spot klines archive csv file format, the futures/um klines don't have the header line
#
# open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore  # noqa: E501
# 1698364800000,34161.6,34182.5,33977.4,34024.2,409953,1698368399999,1202.97118037,15095,192220,564.12041453,0  # noqa: E501
# 1698368400000,34024.2,34060.1,33776.4,33848.4,740960,1698371999999,2183.75671155,23938,368266,1085.17080793,0  # noqa: E501
# 1698372000000,33848.5,34150.0,33815.1,34094.2,390376,1698375599999,1147.73267094,13854,231446,680.60405822,0  # noqa: E501


def make_response_from_url(start_date, end_date):
    def make_daily_df(date, timeframe):
        start = dt_utc(date.year, date.month, date.day)
        end = start + timedelta(days=1)
        date_col = pd.date_range(start, end, freq=timeframe.replace("m", "min"), inclusive="left")
        cols = (
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
            "taker_buy_quote_volume,ignore"
        )
        df = pd.DataFrame(columns=cols.split(","), dtype=float)
        df["open_time"] = date_col.astype("int64") // 10**6
        df["open"] = df["high"] = df["low"] = df["close"] = df["volume"] = 1.0
        return df

    def make_daily_zip(asset_type_url_segment, symbol, timeframe, date) -> bytes:
        df = make_daily_df(date, timeframe)
        if asset_type_url_segment == "spot":
            header = True
        elif asset_type_url_segment == "futures/um":
            header = None
        else:
            raise ValueError
        csv = df.to_csv(index=False, header=header)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr(binance_vision_zip_name(symbol, timeframe, date), csv)
        return zip_buffer.getvalue()

    def make_response(url):
        pattern = (
            r"https://data.binance.vision/data/(?P<asset_type_url_segment>spot|futures/um)"
            r"/daily/klines/(?P<symbol>.*?)/(?P<timeframe>.*?)/(?P=symbol)-(?P=timeframe)-"
            r"(?P<date>\d{4}-\d{2}-\d{2}).zip"
        )
        m = re.match(pattern, url)
        if not m:
            return MockResponse(content="", status=404)

        date = datetime.datetime.strptime(m["date"], "%Y-%m-%d").date()
        if date < start_date or date > end_date:
            return MockResponse(content="", status=404)

        zip_file = make_daily_zip(m["asset_type_url_segment"], m["symbol"], m["timeframe"], date)
        return MockResponse(content=zip_file, status=200)

    return make_response


@pytest.mark.parametrize(
    "candle_type,pair,since,until,first_date,last_date,stop_on_404",
    [
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23),
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59, 59),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23),
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 5),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 3, 23),
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2019, 12, 25),
            dt_utc(2020, 1, 5),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 3, 23),
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2019, 1, 1),
            dt_utc(2019, 1, 5),
            None,
            None,
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2021, 1, 1),
            dt_utc(2021, 1, 5),
            None,
            None,
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 2),
            None,
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 3, 23),
            False,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 5),
            dt_utc(2020, 1, 1),
            None,
            None,
            False,
        ),
        (
            CandleType.FUTURES,
            "BTC/USDT:USDT",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59, 59),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23),
            False,
        ),
        (
            CandleType.INDEX,
            "N/A",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59, 59),
            None,
            None,
            False,
        ),
        # stop_on_404 = True
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2019, 12, 25),
            dt_utc(2020, 1, 5),
            None,
            None,
            True,
        ),
        (
            CandleType.SPOT,
            "BTC/USDT",
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 5),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 3, 23),
            True,
        ),
        (
            CandleType.FUTURES,
            "BTC/USDT:USDT",
            dt_utc(2019, 12, 25),
            dt_utc(2020, 1, 5),
            None,
            None,
            True,
        ),
    ],
)
async def test_download_archive_ohlcv(
    mocker, candle_type, pair, since, until, first_date, last_date, stop_on_404
):
    history_start = dt_utc(2020, 1, 1).date()
    history_end = dt_utc(2020, 1, 3).date()
    timeframe = "1h"

    since_ms = dt_ts(since)
    until_ms = dt_ts(until)

    mocker.patch(
        "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
        side_effect=make_response_from_url(history_start, history_end),
    )
    markets = {"BTC/USDT": {"id": "BTCUSDT"}, "BTC/USDT:USDT": {"id": "BTCUSDT"}}

    df = await download_archive_ohlcv(
        candle_type,
        pair,
        timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        markets=markets,
        stop_on_404=stop_on_404,
    )

    if df.empty:
        assert first_date is None and last_date is None
    else:
        assert candle_type in [CandleType.SPOT, CandleType.FUTURES]
        assert df["date"].iloc[0] == first_date
        assert df["date"].iloc[-1] == last_date


async def test_download_archive_ohlcv_exception(mocker):
    timeframe = "1h"
    pair = "BTC/USDT"

    since_ms = dt_ts(dt_utc(2020, 1, 1))
    until_ms = dt_ts(dt_utc(2020, 1, 2))

    markets = {"BTC/USDT": {"id": "BTCUSDT"}, "BTC/USDT:USDT": {"id": "BTCUSDT"}}
    mocker.patch(
        "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get", side_effect=RuntimeError
    )

    df = await download_archive_ohlcv(
        CandleType.SPOT, pair, timeframe, since_ms=since_ms, until_ms=until_ms, markets=markets
    )

    assert df.empty


async def test_get_daily_ohlcv(mocker, testdatadir):
    symbol = "BTCUSDT"
    timeframe = "1h"
    date = dt_utc(2024, 10, 28).date()
    first_date = dt_utc(2024, 10, 28)
    last_date = dt_utc(2024, 10, 28, 23)

    async with aiohttp.ClientSession() as session:
        spot_path = (
            testdatadir / "binance/binance_public_data/spot-klines-BTCUSDT-1h-2024-10-28.zip"
        )
        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(spot_path.read_bytes(), 200),
        )
        df = await get_daily_ohlcv(symbol, timeframe, CandleType.SPOT, date, session)
        assert get.call_count == 1
        assert df["date"].iloc[0] == first_date
        assert df["date"].iloc[-1] == last_date

        futures_path = (
            testdatadir / "binance/binance_public_data/futures-um-klines-BTCUSDT-1h-2024-10-28.zip"
        )
        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(futures_path.read_bytes(), 200),
        )
        df = await get_daily_ohlcv(symbol, timeframe, CandleType.FUTURES, date, session)
        assert get.call_count == 1
        assert df["date"].iloc[0] == first_date
        assert df["date"].iloc[-1] == last_date

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"", 404),
        )
        with pytest.raises(Http404):
            df = await get_daily_ohlcv(
                symbol, timeframe, CandleType.SPOT, date, session, retry_delay=0
            )
        assert get.call_count == 1

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"", 500),
        )
        mocker.patch("asyncio.sleep")
        with pytest.raises(BadHttpStatus):
            df = await get_daily_ohlcv(symbol, timeframe, CandleType.SPOT, date, session)
        assert get.call_count == 4  # 1 + 3 default retries

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"nop", 200),
        )
        with pytest.raises(zipfile.BadZipFile):
            df = await get_daily_ohlcv(symbol, timeframe, CandleType.SPOT, date, session)
        assert get.call_count == 4  # 1 + 3 default retries


async def test_download_archive_trades(mocker, caplog):
    pair = "BTC/USDT"

    since_ms = dt_ts(dt_utc(2020, 1, 1))
    until_ms = dt_ts(dt_utc(2020, 1, 2))
    markets = {"BTC/USDT": {"id": "BTCUSDT"}, "BTC/USDT:USDT": {"id": "BTCUSDT"}}

    mocker.patch("freqtrade.exchange.binance_public_data.get_daily_trades", return_value=[[2, 3]])

    pair1, res = await download_archive_trades(
        CandleType.SPOT, pair, since_ms=since_ms, until_ms=until_ms, markets=markets
    )
    assert pair1 == pair
    assert res == [[2, 3], [2, 3]]

    mocker.patch(
        "freqtrade.exchange.binance_public_data.get_daily_trades",
        side_effect=Http404("xxx", dt_utc(2020, 1, 1), "http://example.com/something"),
    )

    pair1, res = await download_archive_trades(
        CandleType.SPOT, pair, since_ms=since_ms, until_ms=until_ms, markets=markets
    )

    assert pair1 == pair
    assert res == []
    # exit on day 1
    assert log_has_re("Fast download is unavailable", caplog)

    # Test fail on day 2
    caplog.clear()
    mocker.patch(
        "freqtrade.exchange.binance_public_data.get_daily_trades",
        side_effect=[
            [[2, 3]],
            [[2, 3]],
            Http404("xxx", dt_utc(2020, 1, 2), "http://example.com/something"),
            [[2, 3]],
        ],
    )
    # Download 3 days
    until_ms = dt_ts(dt_utc(2020, 1, 3))

    pair1, res = await download_archive_trades(
        CandleType.SPOT, pair, since_ms=since_ms, until_ms=until_ms, markets=markets
    )

    assert pair1 == pair
    assert res == [[2, 3], [2, 3]]
    assert log_has_re(r"Binance fast download .*stopped", caplog)


async def test_download_archive_trades_exception(mocker, caplog):
    pair = "BTC/USDT"

    since_ms = dt_ts(dt_utc(2020, 1, 1))
    until_ms = dt_ts(dt_utc(2020, 1, 2))

    markets = {"BTC/USDT": {"id": "BTCUSDT"}, "BTC/USDT:USDT": {"id": "BTCUSDT"}}
    mocker.patch(
        "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get", side_effect=RuntimeError
    )

    pair1, res = await download_archive_trades(
        CandleType.SPOT, pair, since_ms=since_ms, until_ms=until_ms, markets=markets
    )

    assert pair1 == pair
    assert res == []
    mocker.patch(
        "freqtrade.exchange.binance_public_data._download_archive_trades", side_effect=RuntimeError
    )

    await download_archive_trades(
        CandleType.SPOT, pair, since_ms=since_ms, until_ms=until_ms, markets=markets
    )
    assert pair1 == pair
    assert res == []
    assert log_has_re("An exception occurred during fast trades download", caplog)


async def test_binance_vision_trades_zip_url():
    url = binance_vision_trades_zip_url("BTCUSDT", CandleType.SPOT, dt_utc(2023, 10, 27))
    assert (
        url == "https://data.binance.vision/data/spot/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2023-10-27.zip"
    )

    url = binance_vision_trades_zip_url("BTCUSDT", CandleType.FUTURES, dt_utc(2023, 10, 28))
    assert (
        url == "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2023-10-28.zip"
    )


async def test_get_daily_trades(mocker, testdatadir):
    symbol = "PEPEUSDT"
    symbol_futures = "APEUSDT"
    date = dt_utc(2024, 10, 28).date()
    first_date = 1729987202368
    last_date = 1730073596350

    async with aiohttp.ClientSession() as session:
        spot_path = (
            testdatadir / "binance/binance_public_data/spot-PEPEUSDT-aggTrades-2024-10-27.zip"
        )
        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(spot_path.read_bytes(), 200),
        )
        res = await get_daily_trades(symbol, CandleType.SPOT, date, session)
        assert get.call_count == 1
        assert res[0][0] == first_date
        assert res[-1][0] == last_date

        futures_path = (
            testdatadir / "binance/binance_public_data/futures-APEUSDT-aggTrades-2024-10-18.zip"
        )
        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(futures_path.read_bytes(), 200),
        )
        res_fut = await get_daily_trades(symbol_futures, CandleType.FUTURES, date, session)
        assert get.call_count == 1
        assert res_fut[0][0] == 1729209603958
        assert res_fut[-1][0] == 1729295981272

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"", 404),
        )
        with pytest.raises(Http404):
            await get_daily_trades(symbol, CandleType.SPOT, date, session, retry_delay=0)
        assert get.call_count == 1

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"", 500),
        )
        mocker.patch("asyncio.sleep")
        with pytest.raises(BadHttpStatus):
            await get_daily_trades(symbol, CandleType.SPOT, date, session)
        assert get.call_count == 4  # 1 + 3 default retries

        get = mocker.patch(
            "freqtrade.exchange.binance_public_data.aiohttp.ClientSession.get",
            return_value=MockResponse(b"nop", 200),
        )
        with pytest.raises(zipfile.BadZipFile):
            await get_daily_trades(symbol, CandleType.SPOT, date, session)
        assert get.call_count == 4  # 1 + 3 default retries
