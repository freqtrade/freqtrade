import pytest

from freqtrade.candle_columns import (
    ALL_CANDLE_VALUE_COLUMNS,
    OHLCV_COLUMNS,
    candle_type_is_ohlcv,
    get_candle_agg_dict,
    get_candle_columns,
    get_candle_dtypes,
)
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType


def test_default_dataframe_columns_reexport():
    # Re-exported from constants for backwards compatibility
    assert DEFAULT_DATAFRAME_COLUMNS is OHLCV_COLUMNS
    assert DEFAULT_DATAFRAME_COLUMNS == ["date", "open", "high", "low", "close", "volume"]


@pytest.mark.parametrize(
    "candle_type,expected",
    [
        (CandleType.SPOT, OHLCV_COLUMNS),
        (CandleType.FUTURES, OHLCV_COLUMNS),
        (CandleType.MARK, OHLCV_COLUMNS),
        (CandleType.INDEX, OHLCV_COLUMNS),
        (CandleType.PREMIUMINDEX, OHLCV_COLUMNS),
        (CandleType.FUNDING_RATE, ["date", "funding_rate"]),
        (CandleType.OPEN_INTEREST, ["date", "open_interest_amount", "open_interest_value"]),
        # DataProvider and informative pairs pass plain strings / None for spot
        ("", OHLCV_COLUMNS),
        (None, OHLCV_COLUMNS),
        ("mark", OHLCV_COLUMNS),
        ("funding_rate", ["date", "funding_rate"]),
        ("open_interest", ["date", "open_interest_amount", "open_interest_value"]),
    ],
)
def test_get_candle_columns(candle_type, expected):
    columns = get_candle_columns(candle_type)
    assert columns == expected
    # "date" is mandatory for every candle type
    assert columns[0] == "date"


def test_get_candle_dtypes():
    assert get_candle_dtypes(CandleType.SPOT) == {
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float",
    }
    assert get_candle_dtypes(CandleType.FUNDING_RATE) == {"funding_rate": "float"}
    assert get_candle_dtypes(CandleType.OPEN_INTEREST) == {
        "open_interest_amount": "float",
        "open_interest_value": "float",
    }
    # date is never cast
    assert "date" not in get_candle_dtypes(CandleType.SPOT)


def test_get_candle_agg_dict():
    assert get_candle_agg_dict(CandleType.SPOT) == {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "max",
    }
    assert get_candle_agg_dict(CandleType.FUNDING_RATE) == {"funding_rate": "first"}
    assert get_candle_agg_dict(CandleType.OPEN_INTEREST) == {
        "open_interest_amount": "first",
        "open_interest_value": "first",
    }
    # Callers must not be able to mutate the shared default
    get_candle_agg_dict(CandleType.SPOT)["open"] = "last"
    assert get_candle_agg_dict(CandleType.SPOT)["open"] == "first"


def test_candle_type_is_ohlcv():
    assert candle_type_is_ohlcv(CandleType.SPOT)
    assert candle_type_is_ohlcv(CandleType.MARK)
    assert candle_type_is_ohlcv("")
    assert candle_type_is_ohlcv(None)
    assert not candle_type_is_ohlcv(CandleType.FUNDING_RATE)
    assert not candle_type_is_ohlcv(CandleType.OPEN_INTEREST)


def test_all_candle_value_columns():
    assert ALL_CANDLE_VALUE_COLUMNS == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "open_interest_amount",
        "open_interest_value",
    }
    assert "date" not in ALL_CANDLE_VALUE_COLUMNS
