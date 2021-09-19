from math import isclose

import pytest

from freqtrade.enums import Collateral, TradingMode
from freqtrade.leverage import liquidation_price


# from freqtrade.exceptions import OperationalException

spot = TradingMode.SPOT
margin = TradingMode.MARGIN
futures = TradingMode.FUTURES

cross = Collateral.CROSS
isolated = Collateral.ISOLATED


@pytest.mark.parametrize('exchange_name,open_rate,is_short,leverage,trading_mode,collateral', [
    # Bittrex
    ('bittrex', "2.0", False, "3.0", spot, None),
    ('bittrex', "2.0", False, "1.0", spot, cross),
    ('bittrex', "2.0", True, "3.0", spot, isolated),
    # Binance
    ('binance', "2.0", False, "3.0", spot, None),
    ('binance', "2.0", False, "1.0", spot, cross),
    ('binance', "2.0", True, "3.0", spot, isolated),
    # Kraken
    ('kraken', "2.0", False, "3.0", spot, None),
    ('kraken', "2.0", True, "3.0", spot, cross),
    ('kraken', "2.0", False, "1.0", spot, isolated),
    # FTX
    ('ftx', "2.0", True, "3.0", spot, None),
    ('ftx', "2.0", False, "3.0", spot, cross),
    ('ftx', "2.0", False, "3.0", spot, isolated),
])
def test_liquidation_price_is_none(
    exchange_name,
    open_rate,
    is_short,
    leverage,
    trading_mode,
    collateral
):
    assert liquidation_price(
        exchange_name,
        open_rate,
        is_short,
        leverage,
        trading_mode,
        collateral,
        1535443.01,
        71200.81144,
        -56354.57,
        135365.00,
        3683.979,
        1456.84,
        0.10,
    ) is None


@pytest.mark.parametrize('exchange_name,open_rate,is_short,leverage,trading_mode,collateral', [
    # Bittrex
    ('bittrex', "2.0", False, "3.0", margin, cross),
    ('bittrex', "2.0", False, "3.0", margin, isolated),
    ('bittrex', "2.0", False, "3.0", futures, cross),
    ('bittrex', "2.0", False, "3.0", futures, isolated),
    # Binance
    # Binance supports isolated margin, but freqtrade likely won't for a while on Binance
    ('binance', "2.0", True, "3.0", margin, isolated),
    # Kraken
    ('kraken', "2.0", False, "1.0", margin, isolated),
    ('kraken', "2.0", False, "1.0", futures, isolated),
    # FTX
    ('ftx', "2.0", False, "3.0", margin, isolated),
    ('ftx', "2.0", False, "3.0", futures, isolated),
])
def test_liquidation_price_exception_thrown(
    exchange_name,
    open_rate,
    is_short,
    leverage,
    trading_mode,
    collateral,
    result
):
    # TODO-lev assert exception is thrown
    return  # Here to avoid indent error, remove when implemented


@pytest.mark.parametrize(
    'exchange_name, open_rate, is_short, leverage, trading_mode, collateral, wallet_balance, '
    'mm_ex_1, upnl_ex_1, maintenance_amt, position, entry_price, '
    'mm_rate, expected',
    [
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.ISOLATED, 1535443.01, 0.0,
         0.0, 135365.00, 3683.979, 1456.84, 0.10, 1114.78),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.ISOLATED, 1535443.01, 0.0,
         0.0, 16300.000, 109.488, 32481.980, 0.025, 18778.73),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.CROSS, 1535443.01, 71200.81144,
         -56354.57, 135365.00, 3683.979, 1456.84, 0.10, 1153.26),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.CROSS, 1535443.01, 356512.508,
         -448192.89, 16300.000, 109.488, 32481.980, 0.025, 26316.89)
    ])
def test_liquidation_price(exchange_name, open_rate, is_short, leverage, trading_mode, collateral,
                           wallet_balance, mm_ex_1, upnl_ex_1,
                           maintenance_amt, position, entry_price, mm_rate,
                           expected):
    assert isclose(round(liquidation_price(
        exchange_name=exchange_name,
        open_rate=open_rate,
        is_short=is_short,
        leverage=leverage,
        trading_mode=trading_mode,
        collateral=collateral,
        wallet_balance=wallet_balance,
        mm_ex_1=mm_ex_1,
        upnl_ex_1=upnl_ex_1,
        maintenance_amt=maintenance_amt,
        position=position,
        entry_price=entry_price,
        mm_rate=mm_rate
    ), 2), expected)
