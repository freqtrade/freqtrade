from decimal import Decimal
from math import isclose

import pytest

from freqtrade.enums import Collateral, TradingMode
from freqtrade.leverage import interest, liquidation_price


# from freqtrade.exceptions import OperationalException
binance = "binance"
kraken = "kraken"
ftx = "ftx"
other = "bittrex"


def test_liquidation_price():

    spot = TradingMode.SPOT
    margin = TradingMode.MARGIN
    futures = TradingMode.FUTURES

    cross = Collateral.CROSS
    isolated = Collateral.ISOLATED

    # NONE
    assert liquidation_price(exchange_name=other, trading_mode=spot) is None
    assert liquidation_price(exchange_name=other, trading_mode=margin,
                             collateral=cross) is None
    assert liquidation_price(exchange_name=other, trading_mode=margin,
                             collateral=isolated) is None
    assert liquidation_price(
        exchange_name=other, trading_mode=futures, collateral=cross) is None
    assert liquidation_price(exchange_name=other, trading_mode=futures,
                             collateral=isolated) is None

    # Binance
    assert liquidation_price(exchange_name=binance, trading_mode=spot) is None
    assert liquidation_price(exchange_name=binance, trading_mode=spot,
                             collateral=cross) is None
    assert liquidation_price(exchange_name=binance, trading_mode=spot,
                             collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # TODO-lev: Replace 1.0 with real value
    # assert liquidation_price(
    #   exchange_name=binance,
    #   trading_mode=margin,
    #   collateral=cross
    # ) == 1.0
    # assert liquidation_price(
    #   exchange_name=binance,
    #   trading_mode=margin,
    #   collateral=isolated
    # ) == 1.0
    # assert liquidation_price(
    #   exchange_name=binance,
    #   trading_mode=futures,
    #   collateral=cross
    # ) == 1.0

    # Binance supports isolated margin, but freqtrade likely won't for a while on Binance
    # liquidation_price(exchange_name=binance, trading_mode=margin, collateral=isolated)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # Kraken
    assert liquidation_price(exchange_name=kraken, trading_mode=spot) is None
    assert liquidation_price(exchange_name=kraken, trading_mode=spot, collateral=cross) is None
    assert liquidation_price(exchange_name=kraken, trading_mode=spot,
                             collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # assert liquidation_price(kraken, trading_mode=margin, collateral=cross) == 1.0
    # assert liquidation_price(kraken, trading_mode=margin, collateral=isolated) == 1.0

    # liquidation_price(kraken, trading_mode=futures, collateral=cross)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # liquidation_price(kraken, trading_mode=futures, collateral=isolated)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # FTX
    assert liquidation_price(ftx, trading_mode=spot) is None
    assert liquidation_price(ftx, trading_mode=spot, collateral=cross) is None
    assert liquidation_price(ftx, trading_mode=spot, collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # assert liquidation_price(ftx, trading_mode=margin, collateral=cross) == 1.0
    # assert liquidation_price(ftx, trading_mode=margin, collateral=isolated) == 1.0

    # liquidation_price(ftx, trading_mode=futures, collateral=cross)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # liquidation_price(ftx, trading_mode=futures, collateral=isolated)
    # assert exception thrown #TODO-lev: Check that exception is thrown


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
    # TODO-lev: - implement FTX tests
    # ('ftx', Decimal(0.0005), ten_mins, 0.06),
    # ('ftx', Decimal(0.0005), five_hours, 0.045),
])
def test_interest(exchange, interest_rate, hours, expected):
    borrowed = Decimal(60.0)

    assert isclose(interest(
        exchange_name=exchange,
        borrowed=borrowed,
        rate=Decimal(interest_rate),
        hours=hours
    ), expected)
