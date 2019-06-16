from freqtrade.utils import setup_configuration, start_list_exchanges
from freqtrade.tests.conftest import get_args, log_has, log_has_re
from freqtrade.state import RunMode

import re


def test_setup_configuration():
    args = [
        '--config', 'config.json.example',
    ]

    config = setup_configuration(get_args(args), RunMode.OTHER)
    assert "exchange" in config
    assert config['exchange']['key'] == ''
    assert config['exchange']['secret'] == ''


def test_list_exchanges(capsys):

    args = [
        "list-exchanges",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"Exchanges supported by ccxt and available.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)

    # Test with --one-column
    args = [
        "list-exchanges",
        "--one-column",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert not re.match(r"Exchanges supported by ccxt and available.*", captured.out)
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)
