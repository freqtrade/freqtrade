from datetime import timedelta

from freqtrade.util import (
    decimals_per_coin,
    fmt_coin,
    fmt_coin2,
    format_duration,
    format_pct,
    round_value,
)


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
    assert fmt_coin(float("nan"), "USDT", False, True) == "N/A"


def test_fmt_coin2():
    assert fmt_coin2(222.222222, "USDT") == "222.222222 USDT"
    assert fmt_coin2(222.2, "XRP", 3, keep_trailing_zeros=True) == "222.200 XRP"
    assert fmt_coin2(222.2, "USDT") == "222.2 USDT"
    assert fmt_coin2(222.12745, "EUR") == "222.12745 EUR"
    assert fmt_coin2(0.1274512123, "BTC") == "0.12745121 BTC"
    assert fmt_coin2(0.1274512123, "ETH") == "0.12745121 ETH"
    assert fmt_coin2(0.00001245, "PEPE") == "0.00001245 PEPE"
    assert fmt_coin2(float("nan"), "PEPE") == "N/A PEPE"


def test_round_value():
    assert round_value(222.222222, 3) == "222.222"
    assert round_value(222.2, 3) == "222.2"
    assert round_value(222.00, 3) == "222"
    assert round_value(222.12745, 3) == "222.127"
    assert round_value(0.1274512123, 8) == "0.12745121"
    assert round_value(0.1274512123, 5) == "0.12745"
    assert round_value(222.2, 3, True) == "222.200"
    assert round_value(222.2, 0, True) == "222"
    assert round_value(float("nan"), 0, True) == "N/A"
    assert round_value(float("nan"), 10, True) == "N/A"
    assert round_value(None, 10, True) == "N/A"
    assert round_value(None, 1, True) == "N/A"


def test_format_duration():
    assert format_duration(timedelta(minutes=5)) == "0d 00:05"
    assert format_duration(timedelta(minutes=75)) == "0d 01:15"
    assert format_duration(timedelta(minutes=1440)) == "1d 00:00"
    assert format_duration(timedelta(minutes=1445)) == "1d 00:05"
    assert format_duration(timedelta(minutes=11445)) == "7d 22:45"
    assert format_duration(timedelta(minutes=101445)) == "70d 10:45"


def test_format_pct():
    assert format_pct(0.1234) == "12.34%"
    assert format_pct(0.1) == "10.00%"
    assert format_pct(0.0) == "0.00%"
    assert format_pct(-0.0567) == "-5.67%"
    assert format_pct(-1.5567) == "-155.67%"
    assert format_pct(None) == "N/A"
    assert format_pct(float("nan")) == "N/A"
