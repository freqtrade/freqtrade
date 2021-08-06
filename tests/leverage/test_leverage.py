# from decimal import Decimal

from freqtrade.enums import Collateral, TradingMode
from freqtrade.leverage import liquidation_price


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
