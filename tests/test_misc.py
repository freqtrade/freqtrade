# pragma pylint: disable=missing-docstring,C0103

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.misc import (decimals_per_coin, file_dump_json, file_load_json, format_ms_time,
                            pair_to_filename, parse_db_uri_for_logging, plural, render_template,
                            render_template_with_fallback, round_coin_value, safe_value_fallback,
                            safe_value_fallback2, shorten_date)


def test_decimals_per_coin():
    assert decimals_per_coin('USDT') == 3
    assert decimals_per_coin('EUR') == 3
    assert decimals_per_coin('BTC') == 8
    assert decimals_per_coin('ETH') == 5


def test_round_coin_value():
    assert round_coin_value(222.222222, 'USDT') == '222.222 USDT'
    assert round_coin_value(222.2, 'USDT') == '222.200 USDT'
    assert round_coin_value(222.12745, 'EUR') == '222.127 EUR'
    assert round_coin_value(0.1274512123, 'BTC') == '0.12745121 BTC'
    assert round_coin_value(0.1274512123, 'ETH') == '0.12745 ETH'

    assert round_coin_value(222.222222, 'USDT', False) == '222.222'
    assert round_coin_value(222.2, 'USDT', False) == '222.200'
    assert round_coin_value(222.12745, 'EUR', False) == '222.127'
    assert round_coin_value(0.1274512123, 'BTC', False) == '0.12745121'
    assert round_coin_value(0.1274512123, 'ETH', False) == '0.12745'


def test_shorten_date() -> None:
    str_data = '1 day, 2 hours, 3 minutes, 4 seconds ago'
    str_shorten_data = '1 d, 2 h, 3 min, 4 sec ago'
    assert shorten_date(str_data) == str_shorten_data


def test_file_dump_json(mocker) -> None:
    file_open = mocker.patch('freqtrade.misc.open', MagicMock())
    json_dump = mocker.patch('rapidjson.dump', MagicMock())
    file_dump_json(Path('somefile'), [1, 2, 3])
    assert file_open.call_count == 1
    assert json_dump.call_count == 1
    file_open = mocker.patch('freqtrade.misc.gzip.open', MagicMock())
    json_dump = mocker.patch('rapidjson.dump', MagicMock())
    file_dump_json(Path('somefile'), [1, 2, 3], True)
    assert file_open.call_count == 1
    assert json_dump.call_count == 1


def test_file_load_json(mocker, testdatadir) -> None:

    # 7m .json does not exist
    ret = file_load_json(testdatadir / 'UNITTEST_BTC-7m.json')
    assert not ret
    # 1m json exists (but no .gz exists)
    ret = file_load_json(testdatadir / 'UNITTEST_BTC-1m.json')
    assert ret
    # 8 .json is empty and will fail if it's loaded. .json.gz is a copy of 1.json
    ret = file_load_json(testdatadir / 'UNITTEST_BTC-8m.json')
    assert ret


@pytest.mark.parametrize("pair,expected_result", [
    ("ETH/BTC", 'ETH_BTC'),
    ("Fabric Token/ETH", 'Fabric_Token_ETH'),
    ("ETHH20", 'ETHH20'),
    (".XBTBON2H", '_XBTBON2H'),
    ("ETHUSD.d", 'ETHUSD_d'),
    ("ADA-0327", 'ADA_0327'),
    ("BTC-USD-200110", 'BTC_USD_200110'),
    ("F-AKRO/USDT", 'F_AKRO_USDT'),
    ("LC+/ETH", 'LC__ETH'),
    ("CMT@18/ETH", 'CMT_18_ETH'),
    ("LBTC:1022/SAI", 'LBTC_1022_SAI'),
    ("$PAC/BTC", '_PAC_BTC'),
    ("ACC_OLD/BTC", 'ACC_OLD_BTC'),
])
def test_pair_to_filename(pair, expected_result):
    pair_s = pair_to_filename(pair)
    assert pair_s == expected_result


def test_format_ms_time() -> None:
    # Date 2018-04-10 18:02:01
    date_in_epoch_ms = 1523383321000
    date = format_ms_time(date_in_epoch_ms)
    assert type(date) is str
    res = datetime.datetime(2018, 4, 10, 18, 2, 1, tzinfo=datetime.timezone.utc)
    assert date == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')
    res = datetime.datetime(2017, 12, 13, 8, 2, 1, tzinfo=datetime.timezone.utc)
    # Date 2017-12-13 08:02:01
    date_in_epoch_ms = 1513152121000
    assert format_ms_time(date_in_epoch_ms) == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')


def test_safe_value_fallback():
    dict1 = {'keya': None, 'keyb': 2, 'keyc': 5, 'keyd': None}
    assert safe_value_fallback(dict1, 'keya', 'keyb') == 2
    assert safe_value_fallback(dict1, 'keyb', 'keya') == 2

    assert safe_value_fallback(dict1, 'keyb', 'keyc') == 2
    assert safe_value_fallback(dict1, 'keya', 'keyc') == 5

    assert safe_value_fallback(dict1, 'keyc', 'keyb') == 5

    assert safe_value_fallback(dict1, 'keya', 'keyd') is None

    assert safe_value_fallback(dict1, 'keyNo', 'keyNo') is None
    assert safe_value_fallback(dict1, 'keyNo', 'keyNo', 55) == 55


def test_safe_value_fallback2():
    dict1 = {'keya': None, 'keyb': 2, 'keyc': 5, 'keyd': None}
    dict2 = {'keya': 20, 'keyb': None, 'keyc': 6, 'keyd': None}
    assert safe_value_fallback2(dict1, dict2, 'keya', 'keya') == 20
    assert safe_value_fallback2(dict2, dict1, 'keya', 'keya') == 20

    assert safe_value_fallback2(dict1, dict2, 'keyb', 'keyb') == 2
    assert safe_value_fallback2(dict2, dict1, 'keyb', 'keyb') == 2

    assert safe_value_fallback2(dict1, dict2, 'keyc', 'keyc') == 5
    assert safe_value_fallback2(dict2, dict1, 'keyc', 'keyc') == 6

    assert safe_value_fallback2(dict1, dict2, 'keyd', 'keyd') is None
    assert safe_value_fallback2(dict2, dict1, 'keyd', 'keyd') is None
    assert safe_value_fallback2(dict2, dict1, 'keyd', 'keyd', 1234) == 1234

    assert safe_value_fallback2(dict1, dict2, 'keyNo', 'keyNo') is None
    assert safe_value_fallback2(dict2, dict1, 'keyNo', 'keyNo') is None
    assert safe_value_fallback2(dict2, dict1, 'keyNo', 'keyNo', 1234) == 1234


def test_plural() -> None:
    assert plural(0, "page") == "pages"
    assert plural(0.0, "page") == "pages"
    assert plural(1, "page") == "page"
    assert plural(1.0, "page") == "page"
    assert plural(2, "page") == "pages"
    assert plural(2.0, "page") == "pages"
    assert plural(-1, "page") == "page"
    assert plural(-1.0, "page") == "page"
    assert plural(-2, "page") == "pages"
    assert plural(-2.0, "page") == "pages"
    assert plural(0.5, "page") == "pages"
    assert plural(1.5, "page") == "pages"
    assert plural(-0.5, "page") == "pages"
    assert plural(-1.5, "page") == "pages"

    assert plural(0, "ox", "oxen") == "oxen"
    assert plural(0.0, "ox", "oxen") == "oxen"
    assert plural(1, "ox", "oxen") == "ox"
    assert plural(1.0, "ox", "oxen") == "ox"
    assert plural(2, "ox", "oxen") == "oxen"
    assert plural(2.0, "ox", "oxen") == "oxen"
    assert plural(-1, "ox", "oxen") == "ox"
    assert plural(-1.0, "ox", "oxen") == "ox"
    assert plural(-2, "ox", "oxen") == "oxen"
    assert plural(-2.0, "ox", "oxen") == "oxen"
    assert plural(0.5, "ox", "oxen") == "oxen"
    assert plural(1.5, "ox", "oxen") == "oxen"
    assert plural(-0.5, "ox", "oxen") == "oxen"
    assert plural(-1.5, "ox", "oxen") == "oxen"


def test_render_template_fallback(mocker):
    from jinja2.exceptions import TemplateNotFound
    with pytest.raises(TemplateNotFound):
        val = render_template(
            templatefile='subtemplates/indicators_does-not-exist.j2',)

    val = render_template_with_fallback(
        templatefile='subtemplates/indicators_does-not-exist.j2',
        templatefallbackfile='subtemplates/indicators_minimal.j2',
    )
    assert isinstance(val, str)
    assert 'if self.dp' in val


def test_parse_db_uri_for_logging() -> None:
    postgresql_conn_uri = "postgresql+psycopg2://scott123:scott123@host/dbname"
    mariadb_conn_uri = "mariadb+mariadbconnector://app_user:Password123!@127.0.0.1:3306/company"
    mysql_conn_uri = "mysql+pymysql://user:pass@some_mariadb/dbname?charset=utf8mb4"
    sqlite_conn_uri = "sqlite:////freqtrade/user_data/tradesv3.sqlite"
    censored_pwd = "*****"

    def get_pwd(x): return x.split(':')[2].split('@')[0]

    assert get_pwd(parse_db_uri_for_logging(postgresql_conn_uri)) == censored_pwd
    assert get_pwd(parse_db_uri_for_logging(mariadb_conn_uri)) == censored_pwd
    assert get_pwd(parse_db_uri_for_logging(mysql_conn_uri)) == censored_pwd
    assert sqlite_conn_uri == parse_db_uri_for_logging(sqlite_conn_uri)
