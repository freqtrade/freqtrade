import pytest

from freqtrade.enums import CandleType


@pytest.mark.parametrize('input,expected', [
    ('', CandleType.SPOT_),
    ('spot', CandleType.SPOT),
    (CandleType.SPOT, CandleType.SPOT),
    (CandleType.FUTURES, CandleType.FUTURES),
    (CandleType.INDEX, CandleType.INDEX),
    (CandleType.MARK, CandleType.MARK),
    ('futures', CandleType.FUTURES),
    ('mark', CandleType.MARK),
    ('premiumIndex', CandleType.PREMIUMINDEX),
])
def test_candle_type_from_string(input, expected):
    assert CandleType.from_string(input) == expected
