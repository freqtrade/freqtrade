from freqtrade.enums import LiqFormula, TradingMode


# from freqtrade.exceptions import OperationalException


def test_liquidation_formula():
    spot = TradingMode.SPOT
    # cross_margin = TradingMode.CROSS_MARGIN
    # isolated_margin = TradingMode.ISOLATED_MARGIN
    # cross_futures = TradingMode.CROSS_FUTURES
    # isolated_futures = TradingMode.ISOLATED_FUTURES

    assert LiqFormula.BINANCE(
        trading_mode=spot
    ) is None
    # TODO-mg: Uncomment these assertions and make them real calculation tests
    # assert LiqFormula.BINANCE(
    #   trading_mode=cross_margin
    # ) == 1.0  #Replace 1.0 with real value
    # assert LiqFormula.BINANCE(
    #   trading_mode=isolated_margin
    # ) == 1.0
    # assert LiqFormula.BINANCE(
    #   trading_mode=cross_futures
    # ) == 1.0
    # assert LiqFormula.BINANCE(
    #   trading_mode=isolated_futures
    # ) == 1.0

    assert LiqFormula.KRAKEN(
        trading_mode=spot
    ) is None
    # TODO-mg: Uncomment these assertions and make them real calculation tests
    # assert LiqFormula.KRAKEN(
    #   trading_mode=cross_margin
    # ) == 1.0
    # LiqFormula.KRAKEN(
    #   trading_mode=isolated_margin
    # )
    # asset exception thrown #TODO-mg: Check that exception is thrown
    # assert LiqFormula.KRAKEN(
    #   trading_mode=cross_futures
    # ) == 1.0
    # LiqFormula.KRAKEN(
    #   trading_mode=isolated_futures
    # )
    # asset exception thrown #TODO-mg: Check that exception is thrown

    assert LiqFormula.FTX(
        trading_mode=spot
    ) is None
    # TODO-mg: Uncomment these assertions and make them real calculation tests
    # assert LiqFormula.FTX(
    #   trading_mode=cross_margin
    # ) == 1.0
    # assert LiqFormula.FTX(
    #   trading_mode=isolated_margin
    # ) == 1.0
    # assert LiqFormula.FTX(
    #   trading_mode=cross_futures
    # ) == 1.0
    # assert LiqFormula.FTX(
    #   trading_mode=isolated_futures
    # ) == 1.0
