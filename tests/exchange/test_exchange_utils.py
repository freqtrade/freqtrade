# pragma pylint: disable=missing-docstring, protected-access, invalid-name
from datetime import datetime, timedelta, timezone

import pytest
from ccxt import DECIMAL_PLACES, ROUND, ROUND_UP, TICK_SIZE, TRUNCATE

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import (amount_to_precision, date_minus_candles, price_to_precision,
                                timeframe_to_prev_date)
from freqtrade.exchange.check_exchange import check_exchange
from tests.conftest import log_has_re


def test_check_exchange(default_conf, caplog) -> None:
    # Test an officially supported by Freqtrade team exchange
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.get('exchange').update({'name': 'BITTREX'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is officially supported by the Freqtrade development team\.",
                      caplog)
    caplog.clear()

    # Test an officially supported by Freqtrade team exchange
    default_conf.get('exchange').update({'name': 'binance'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"binance\" is officially supported by the Freqtrade development team\.",
        caplog)
    caplog.clear()

    # Test an officially supported by Freqtrade team exchange
    default_conf.get('exchange').update({'name': 'binanceus'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"binanceus\" is officially supported by the Freqtrade development team\.",
        caplog)
    caplog.clear()

    # Test an officially supported by Freqtrade team exchange - with remapping
    default_conf.get('exchange').update({'name': 'okex'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"okex\" is officially supported by the Freqtrade development team\.",
        caplog)
    caplog.clear()
    # Test an available exchange, supported by ccxt
    default_conf.get('exchange').update({'name': 'huobipro'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is known to the the ccxt library, available for the bot, "
                      r"but not officially supported "
                      r"by the Freqtrade development team\. .*", caplog)
    caplog.clear()

    # Test a 'bad' exchange, which known to have serious problems
    default_conf.get('exchange').update({'name': 'bitmex'})
    with pytest.raises(OperationalException,
                       match=r"Exchange .* will not work with Freqtrade\..*"):
        check_exchange(default_conf)
    caplog.clear()

    # Test a 'bad' exchange with check_for_bad=False
    default_conf.get('exchange').update({'name': 'bitmex'})
    assert check_exchange(default_conf, False)
    assert log_has_re(r"Exchange .* is known to the the ccxt library, available for the bot, "
                      r"but not officially supported "
                      r"by the Freqtrade development team\. .*", caplog)
    caplog.clear()

    # Test an invalid exchange
    default_conf.get('exchange').update({'name': 'unknown_exchange'})
    with pytest.raises(
        OperationalException,
        match=r'Exchange "unknown_exchange" is not known to the ccxt library '
              r'and therefore not available for the bot.*'
    ):
        check_exchange(default_conf)

    # Test no exchange...
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.PLOT
    assert check_exchange(default_conf)

    # Test no exchange...
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.UTIL_EXCHANGE
    with pytest.raises(OperationalException,
                       match=r'This command requires a configured exchange.*'):
        check_exchange(default_conf)


def test_date_minus_candles():

    date = datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)

    assert date_minus_candles("5m", 3, date) == date - timedelta(minutes=15)
    assert date_minus_candles("5m", 5, date) == date - timedelta(minutes=25)
    assert date_minus_candles("1m", 6, date) == date - timedelta(minutes=6)
    assert date_minus_candles("1h", 3, date) == date - timedelta(hours=3, minutes=25)
    assert date_minus_candles("1h", 3) == timeframe_to_prev_date('1h') - timedelta(hours=3)



@pytest.mark.parametrize("amount,precision_mode,precision,expected", [
    (2.34559, 2, 4, 2.3455),
    (2.34559, 2, 5, 2.34559),
    (2.34559, 2, 3, 2.345),
    (2.9999, 2, 3, 2.999),
    (2.9909, 2, 3, 2.990),
    (2.9909, 2, 0, 2),
    (29991.5555, 2, 0, 29991),
    (29991.5555, 2, -1, 29990),
    (29991.5555, 2, -2, 29900),
    # Tests for Tick-size
    (2.34559, 4, 0.0001, 2.3455),
    (2.34559, 4, 0.00001, 2.34559),
    (2.34559, 4, 0.001, 2.345),
    (2.9999, 4, 0.001, 2.999),
    (2.9909, 4, 0.001, 2.990),
    (2.9909, 4, 0.005, 2.99),
    (2.9999, 4, 0.005, 2.995),
])
def test_amount_to_precision(amount, precision_mode, precision, expected,):
    """
    Test rounds down
    """
    # digits counting mode
    # DECIMAL_PLACES = 2
    # SIGNIFICANT_DIGITS = 3
    # TICK_SIZE = 4

    assert amount_to_precision(amount, precision, precision_mode) == expected


@pytest.mark.parametrize("price,precision_mode,precision,expected,rounding_mode", [
    # Tests for DECIMAL_PLACES, ROUND_UP
    (2.34559, 2, 4, 2.3456, ROUND_UP),
    (2.34559, 2, 5, 2.34559, ROUND_UP),
    (2.34559, 2, 3, 2.346, ROUND_UP),
    (2.9999, 2, 3, 3.000, ROUND_UP),
    (2.9909, 2, 3, 2.991, ROUND_UP),
    # Tests for DECIMAL_PLACES, ROUND
    (2.345600000000001, DECIMAL_PLACES, 4, 2.3456, ROUND),
    (2.345551, DECIMAL_PLACES, 4, 2.3456, ROUND),
    (2.49, DECIMAL_PLACES, 0, 2., ROUND),
    (2.51, DECIMAL_PLACES, 0, 3., ROUND),
    (5.1, DECIMAL_PLACES, -1, 10., ROUND),
    (4.9, DECIMAL_PLACES, -1, 0., ROUND),
    # Tests for TICK_SIZE, ROUND_UP
    (2.34559, TICK_SIZE, 0.0001, 2.3456, ROUND_UP),
    (2.34559, TICK_SIZE, 0.00001, 2.34559, ROUND_UP),
    (2.34559, TICK_SIZE, 0.001, 2.346, ROUND_UP),
    (2.9999, TICK_SIZE, 0.001, 3.000, ROUND_UP),
    (2.9909, TICK_SIZE, 0.001, 2.991, ROUND_UP),
    (2.9909, TICK_SIZE, 0.005, 2.995, ROUND_UP),
    (2.9973, TICK_SIZE, 0.005, 3.0, ROUND_UP),
    (2.9977, TICK_SIZE, 0.005, 3.0, ROUND_UP),
    (234.43, TICK_SIZE, 0.5, 234.5, ROUND_UP),
    (234.53, TICK_SIZE, 0.5, 235.0, ROUND_UP),
    (0.891534, TICK_SIZE, 0.0001, 0.8916, ROUND_UP),
    (64968.89, TICK_SIZE, 0.01, 64968.89, ROUND_UP),
    (0.000000003483, TICK_SIZE, 1e-12, 0.000000003483, ROUND_UP),
    # Tests for TICK_SIZE, ROUND
    (2.49, TICK_SIZE, 1., 2., ROUND),
    (2.51, TICK_SIZE, 1., 3., ROUND),
    (2.000000051, TICK_SIZE, 0.0000001, 2.0000001, ROUND),
    (2.000000049, TICK_SIZE, 0.0000001, 2., ROUND),
    (2.9909, TICK_SIZE, 0.005, 2.990, ROUND),
    (2.9973, TICK_SIZE, 0.005, 2.995, ROUND),
    (2.9977, TICK_SIZE, 0.005, 3.0, ROUND),
    (234.24, TICK_SIZE, 0.5, 234., ROUND),
    (234.26, TICK_SIZE, 0.5, 234.5, ROUND),
    # Tests for TRUNCATTE
    (2.34559, 2, 4, 2.3455, TRUNCATE),
    (2.34559, 2, 5, 2.34559, TRUNCATE),
    (2.34559, 2, 3, 2.345, TRUNCATE),
    (2.9999, 2, 3, 2.999, TRUNCATE),
    (2.9909, 2, 3, 2.990, TRUNCATE),
])
def test_price_to_precision(price, precision_mode, precision, expected, rounding_mode):
    assert price_to_precision(
        price, precision, precision_mode, rounding_mode=rounding_mode) == expected
