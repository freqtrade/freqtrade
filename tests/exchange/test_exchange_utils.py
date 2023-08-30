# pragma pylint: disable=missing-docstring, protected-access, invalid-name
from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import date_minus_candles, timeframe_to_prev_date
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
