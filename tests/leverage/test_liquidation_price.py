from freqtrade.enums import Collateral, LiqFormula, TradingMode


# from freqtrade.exceptions import OperationalException


def test_liquidation_formula():

    spot = TradingMode.SPOT
    margin = TradingMode.MARGIN
    futures = TradingMode.FUTURES

    cross = Collateral.CROSS
    isolated = Collateral.ISOLATED

    # NONE
    assert LiqFormula.NONE(trading_mode=spot) is None
    assert LiqFormula.NONE(trading_mode=margin, collateral=cross) is None
    assert LiqFormula.NONE(trading_mode=margin, collateral=isolated) is None
    assert LiqFormula.NONE(trading_mode=futures, collateral=cross) is None
    assert LiqFormula.NONE(trading_mode=futures, collateral=isolated) is None

    # Binance
    assert LiqFormula.BINANCE(trading_mode=spot) is None
    assert LiqFormula.BINANCE(trading_mode=spot, collateral=cross) is None
    assert LiqFormula.BINANCE(trading_mode=spot, collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # TODO-lev: Replace 1.0 with real value
    # assert LiqFormula.BINANCE(trading_mode=margin, collateral=cross) == 1.0
    # assert LiqFormula.BINANCE(trading_mode=margin, collateral=isolated) == 1.0
    # assert LiqFormula.BINANCE(trading_mode=futures, collateral=cross) == 1.0
    # Binance supports isolated margin, but freqtrade likely won't for a while on Binance
    # assert LiqFormula.BINANCE(trading_mode=margin, collateral=isolated) == 1.0

    # Kraken
    assert LiqFormula.KRAKEN(trading_mode=spot) is None
    assert LiqFormula.KRAKEN(trading_mode=spot, collateral=cross) is None
    assert LiqFormula.KRAKEN(trading_mode=spot, collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # assert LiqFormula.KRAKEN(trading_mode=margin, collateral=cross) == 1.0
    # assert LiqFormula.KRAKEN(trading_mode=margin, collateral=isolated) == 1.0

    # LiqFormula.KRAKEN(trading_mode=futures, collateral=cross)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # LiqFormula.KRAKEN(trading_mode=futures, collateral=isolated)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # FTX
    assert LiqFormula.FTX(trading_mode=spot) is None
    assert LiqFormula.FTX(trading_mode=spot, collateral=cross) is None
    assert LiqFormula.FTX(trading_mode=spot, collateral=isolated) is None
    # TODO-lev: Uncomment these assertions and make them real calculation tests
    # assert LiqFormula.FTX(trading_mode=margin, collateral=cross) == 1.0
    # assert LiqFormula.FTX(trading_mode=margin, collateral=isolated) == 1.0

    # LiqFormula.KRAKEN(trading_mode=futures, collateral=cross)
    # assert exception thrown #TODO-lev: Check that exception is thrown

    # LiqFormula.KRAKEN(trading_mode=futures, collateral=isolated)
    # assert exception thrown #TODO-lev: Check that exception is thrown
