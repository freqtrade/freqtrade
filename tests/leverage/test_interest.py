import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.leverage import interest
from freqtrade.util import FtPrecise


ten_mins = FtPrecise(1 / 6)
five_hours = FtPrecise(5.0)
twentyfive_hours = FtPrecise(25.0)


@pytest.mark.parametrize('exchange,interest_rate,hours,expected', [
    ('binance', 0.0005, ten_mins, 0.00125),
    ('binance', 0.00025, ten_mins, 0.000625),
    ('binance', 0.00025, five_hours, 0.003125),
    ('binance', 0.00025, twentyfive_hours, 0.015625),
    # Kraken
    ('kraken', 0.0005, ten_mins, 0.06),
    ('kraken', 0.00025, ten_mins, 0.03),
    ('kraken', 0.00025, five_hours, 0.045),
    ('kraken', 0.00025, twentyfive_hours, 0.12),
])
def test_interest(exchange, interest_rate, hours, expected):
    borrowed = FtPrecise(60.0)

    assert pytest.approx(float(interest(
        exchange_name=exchange,
        borrowed=borrowed,
        rate=FtPrecise(interest_rate),
        hours=hours
    ))) == expected


def test_interest_exception():
    with pytest.raises(OperationalException, match=r"Leverage not available on .* with freqtrade"):
        interest(
            exchange_name='bitmex',
            borrowed=FtPrecise(60.0),
            rate=FtPrecise(0.0005),
            hours=ten_mins
        )
