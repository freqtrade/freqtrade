# from decimal import Decimal

# from freqtrade.enums import Collateral, TradingMode
# from freqtrade.leverage import interest
# from freqtrade.exceptions import OperationalException
# binance = "binance"
# kraken = "kraken"
# ftx = "ftx"
# other = "bittrex"


def test_interest():
    return
    # Binance
    # assert interest(binance, borrowed=60, rate=0.0005,
    #                    hours = 1/6) == round(0.0008333333333333334, 8)
    # TODO-lev: The below tests
    # assert interest(binance, borrowed=60, rate=0.00025, hours=5.0) == 1.0

    # # Kraken
    # assert interest(kraken, borrowed=60, rate=0.0005, hours=1.0) == 1.0
    # assert interest(kraken, borrowed=60, rate=0.00025, hours=5.0) == 1.0

    # # FTX
    # assert interest(ftx, borrowed=60, rate=0.0005, hours=1.0) == 1.0
    # assert interest(ftx, borrowed=60, rate=0.00025, hours=5.0) == 1.0
