from decimal import Decimal

from freqtrade.leverage import interest


# from freqtrade.exceptions import OperationalException
binance = "binance"
kraken = "kraken"
ftx = "ftx"
other = "bittrex"


def test_interest():

    borrowed = Decimal(60.0)
    interest_rate = Decimal(0.0005)
    interest_rate_2 = Decimal(0.00025)
    ten_mins = Decimal(1/6)
    five_hours = Decimal(5.0)

    # Binance
    assert float(interest(
        exchange_name=binance,
        borrowed=borrowed,
        rate=interest_rate,
        hours=ten_mins
    )) == 0.00125

    assert float(interest(
        exchange_name=binance,
        borrowed=borrowed,
        rate=interest_rate_2,
        hours=five_hours
    )) == 0.003125

    # Kraken
    assert float(interest(
        exchange_name=kraken,
        borrowed=borrowed,
        rate=interest_rate,
        hours=ten_mins
    )) == 0.06

    assert float(interest(
        exchange_name=kraken,
        borrowed=borrowed,
        rate=interest_rate_2,
        hours=five_hours
    )) == 0.045

    # FTX
    # TODO-lev
    # assert float(interest(
    #     exchange_name=ftx,
    #     borrowed=borrowed,
    #     rate=interest_rate,
    #     hours=ten_mins
    # )) == 0.00125

    # assert float(interest(
    #     exchange_name=ftx,
    #     borrowed=borrowed,
    #     rate=interest_rate_2,
    #     hours=five_hours
    # )) == 0.003125
