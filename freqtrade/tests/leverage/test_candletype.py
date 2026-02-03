import pytest

from freqtrade.enums import CandleType


@pytest.mark.parametrize(
    "candle_type,expected",
    [
        ("", CandleType.SPOT),
        ("spot", CandleType.SPOT),
        (CandleType.SPOT, CandleType.SPOT),
        (CandleType.FUTURES, CandleType.FUTURES),
        (CandleType.INDEX, CandleType.INDEX),
        (CandleType.MARK, CandleType.MARK),
        ("futures", CandleType.FUTURES),
        ("mark", CandleType.MARK),
        ("premiumIndex", CandleType.PREMIUMINDEX),
    ],
)
def test_CandleType_from_string(candle_type, expected):
    assert CandleType.from_string(candle_type) == expected


@pytest.mark.parametrize(
    "candle_type,expected",
    [
        ("futures", CandleType.FUTURES),
        ("spot", CandleType.SPOT),
        ("margin", CandleType.SPOT),
    ],
)
def test_CandleType_get_default(candle_type, expected):
    assert CandleType.get_default(candle_type) == expected
