from decimal import Decimal
from math import isclose

import pytest

from freqtrade.leverage import interest


ten_mins = Decimal(1/6)
five_hours = Decimal(5.0)
twentyfive_hours = Decimal(25.0)


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
    # FTX
    ('ftx', 0.0005, ten_mins, 0.00125),
    ('ftx', 0.00025, ten_mins, 0.000625),
    ('ftx', 0.00025, five_hours, 0.003125),
    ('ftx', 0.00025, twentyfive_hours, 0.015625),
])
def test_interest(exchange, interest_rate, hours, expected):
    borrowed = Decimal(60.0)

    assert isclose(interest(
        exchange_name=exchange,
        borrowed=borrowed,
        rate=Decimal(interest_rate),
        hours=hours
    ), expected)
