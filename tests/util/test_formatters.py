from freqtrade.util import decimals_per_coin, fmt_coin, round_value


def test_decimals_per_coin():
    assert decimals_per_coin("USDT") == 3
    assert decimals_per_coin("EUR") == 3
    assert decimals_per_coin("BTC") == 8
    assert decimals_per_coin("ETH") == 5


def test_fmt_coin():
    assert fmt_coin(222.222222, "USDT") == "222.222 USDT"
    assert fmt_coin(222.2, "USDT", keep_trailing_zeros=True) == "222.200 USDT"
    assert fmt_coin(222.2, "USDT") == "222.2 USDT"
    assert fmt_coin(222.12745, "EUR") == "222.127 EUR"
    assert fmt_coin(0.1274512123, "BTC") == "0.12745121 BTC"
    assert fmt_coin(0.1274512123, "ETH") == "0.12745 ETH"

    assert fmt_coin(222.222222, "USDT", False) == "222.222"
    assert fmt_coin(222.2, "USDT", False) == "222.2"
    assert fmt_coin(222.00, "USDT", False) == "222"
    assert fmt_coin(222.12745, "EUR", False) == "222.127"
    assert fmt_coin(0.1274512123, "BTC", False) == "0.12745121"
    assert fmt_coin(0.1274512123, "ETH", False) == "0.12745"
    assert fmt_coin(222.2, "USDT", False, True) == "222.200"


def test_round_value():
    assert round_value(222.222222, 3) == "222.222"
    assert round_value(222.2, 3) == "222.2"
    assert round_value(222.00, 3) == "222"
    assert round_value(222.12745, 3) == "222.127"
    assert round_value(0.1274512123, 8) == "0.12745121"
    assert round_value(0.1274512123, 5) == "0.12745"
    assert round_value(222.2, 3, True) == "222.200"
    assert round_value(222.2, 0, True) == "222"
