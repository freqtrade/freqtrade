import re
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade import OperationalException
from freqtrade.state import RunMode
from freqtrade.utils import (setup_utils_configuration, start_create_userdir,
                             start_download_data, start_list_exchanges,
                             start_list_markets, start_list_timeframes,
                             start_new_hyperopt, start_new_strategy,
                             start_test_pairlist, start_trading,
                             start_hyperopt_list, start_hyperopt_show)
from tests.conftest import (get_args, log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


def test_setup_utils_configuration():
    args = [
        'list-exchanges', '--config', 'config.json.example',
    ]

    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert "exchange" in config
    assert config['dry_run'] is True
    assert config['exchange']['key'] == ''
    assert config['exchange']['secret'] == ''


def test_start_trading_fail(mocker):

    mocker.patch("freqtrade.worker.Worker.run", MagicMock(side_effect=OperationalException))

    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(return_value=None))

    exitmock = mocker.patch("freqtrade.worker.Worker.exit", MagicMock())
    args = [
        'trade',
        '-c', 'config.json.example'
    ]
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 1

    exitmock.reset_mock()

    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(side_effect=OperationalException))
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 0


def test_list_exchanges(capsys):

    args = [
        "list-exchanges",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"Exchanges available for Freqtrade.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)

    # Test with --one-column
    args = [
        "list-exchanges",
        "--one-column",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)

    # Test with --all
    args = [
        "list-exchanges",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"All exchanges supported by the ccxt library.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)
    assert re.match(r".*bitmex,.*", captured.out)

    # Test with --one-column --all
    args = [
        "list-exchanges",
        "--one-column",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)
    assert re.search(r"^bitmex$", captured.out, re.MULTILINE)


def test_list_timeframes(mocker, capsys):

    api_mock = MagicMock()
    api_mock.timeframes = {'1m': 'oneMin',
                           '5m': 'fiveMin',
                           '30m': 'thirtyMin',
                           '1h': 'hour',
                           '1d': 'day',
                           }
    patch_exchange(mocker, api_mock=api_mock)
    args = [
        "list-timeframes",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_timeframes(pargs)

    # Test with --config config.json.example
    args = [
        "list-timeframes",
        '--config', 'config.json.example',
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Bittrex`: "
                    "1m, 5m, 30m, 1h, 1d",
                    captured.out)

    # Test with --exchange bittrex
    args = [
        "list-timeframes",
        "--exchange", "bittrex",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Bittrex`: "
                    "1m, 5m, 30m, 1h, 1d",
                    captured.out)

    api_mock.timeframes = {'1m': '1m',
                           '5m': '5m',
                           '15m': '15m',
                           '30m': '30m',
                           '1h': '1h',
                           '6h': '6h',
                           '12h': '12h',
                           '1d': '1d',
                           '3d': '3d',
                           }
    patch_exchange(mocker, api_mock=api_mock, id='binance')
    # Test with --exchange binance
    args = [
        "list-timeframes",
        "--exchange", "binance",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Binance`: "
                    "1m, 5m, 15m, 30m, 1h, 6h, 12h, 1d, 3d",
                    captured.out)

    # Test with --one-column
    args = [
        "list-timeframes",
        '--config', 'config.json.example',
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)

    # Test with --exchange binance --one-column
    args = [
        "list-timeframes",
        "--exchange", "binance",
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)


def test_list_markets(mocker, markets, capsys):

    api_mock = MagicMock()
    api_mock.markets = markets
    patch_exchange(mocker, api_mock=api_mock)

    # Test with no --config
    args = [
        "list-markets",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_markets(pargs, False)

    # Test with --config config.json.example
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 9 active markets: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="binance")
    # Test with --exchange
    args = [
        "list-markets",
        "--exchange", "binance"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_markets(pargs, False)
    captured = capsys.readouterr()
    assert re.match("\nExchange Binance has 9 active markets:\n",
                    captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="bittrex")
    # Test with --all: all markets
    args = [
        "list-markets", "--all",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 11 markets: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand: active pairs
    args = [
        "list-pairs",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 8 active pairs: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, NEO/BTC, TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand with --all: all pairs
    args = [
        "list-pairs", "--all",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 10 pairs: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # active markets, base=ETH, LTC
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "ETH", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 5 active markets with ETH, LTC as base currencies: "
            "ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 3 active markets with LTC as base currency: "
            "LTC/BTC, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, quote=USDT, USD
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--quote", "USDT", "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 3 active markets with USDT, USD as quote currencies: "
            "ETH/USDT, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, quote=USDT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--quote", "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 2 active markets with USDT as quote currency: "
            "ETH/USDT, XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC, quote=USDT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active market with LTC as base currency and "
            "with USDT as quote currency: XLTCUSDT.\n"
            in captured.out)

    # active pairs, base=LTC, quote=USDT
    args = [
        "list-pairs",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active pair with LTC as base currency and "
            "with USD as quote currency: LTC/USD.\n"
            in captured.out)

    # active markets, base=LTC, quote=USDT, NONEXISTENT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USDT", "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active market with LTC as base currency and "
            "with USDT, NONEXISTENT as quote currencies: XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC, quote=NONEXISTENT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 0 active markets with LTC as base currency and "
            "with NONEXISTENT as quote currency.\n"
            in captured.out)

    # Test tabular output
    args = [
        "list-markets",
        '--config', 'config.json.example',
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 9 active markets:\n"
            in captured.out)

    # Test tabular output, no markets found
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "NONEXISTENT",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 0 active markets with LTC as base currency and "
            "with NONEXISTENT as quote currency.\n"
            in captured.out)

    # Test --print-json
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-json"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ('["BLK/BTC","ETH/BTC","ETH/USDT","LTC/BTC","LTC/USD","NEO/BTC",'
            '"TKN/BTC","XLTCUSDT","XRP/BTC"]'
            in captured.out)

    # Test --print-csv
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-csv"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Id,Symbol,Base,Quote,Active,Is pair" in captured.out)
    assert ("blkbtc,BLK/BTC,BLK,BTC,True,True" in captured.out)
    assert ("USD-LTC,LTC/USD,LTC,USD,True,True" in captured.out)

    # Test --one-column
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--one-column"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert re.search(r"^BLK/BTC$", captured.out, re.MULTILINE)
    assert re.search(r"^LTC/USD$", captured.out, re.MULTILINE)


def test_create_datadir_failed(caplog):

    args = [
        "create-userdir",
    ]
    with pytest.raises(SystemExit):
        start_create_userdir(get_args(args))
    assert log_has("`create-userdir` requires --userdir to be set.", caplog)


def test_create_datadir(caplog, mocker):
    cud = mocker.patch("freqtrade.utils.create_userdata_dir", MagicMock())
    csf = mocker.patch("freqtrade.utils.copy_sample_files", MagicMock())
    args = [
        "create-userdir",
        "--userdir",
        "/temp/freqtrade/test"
    ]
    start_create_userdir(get_args(args))

    assert cud.call_count == 1
    assert csf.call_count == 1
    assert len(caplog.record_tuples) == 0


def test_start_new_strategy(mocker, caplog):
    wt_mock = mocker.patch.object(Path, "write_text", MagicMock())
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    args = [
        "new-strategy",
        "--strategy",
        "CoolNewStrategy"
    ]
    start_new_strategy(get_args(args))

    assert wt_mock.call_count == 1
    assert "CoolNewStrategy" in wt_mock.call_args_list[0][0][0]
    assert log_has_re("Writing strategy to .*", caplog)


def test_start_new_strategy_DefaultStrat(mocker, caplog):
    args = [
        "new-strategy",
        "--strategy",
        "DefaultStrategy"
    ]
    with pytest.raises(OperationalException,
                       match=r"DefaultStrategy is not allowed as name\."):
        start_new_strategy(get_args(args))


def test_start_new_strategy_no_arg(mocker, caplog):
    args = [
        "new-strategy",
    ]
    with pytest.raises(OperationalException,
                       match="`new-strategy` requires --strategy to be set."):
        start_new_strategy(get_args(args))


def test_start_new_hyperopt(mocker, caplog):
    wt_mock = mocker.patch.object(Path, "write_text", MagicMock())
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    args = [
        "new-hyperopt",
        "--hyperopt",
        "CoolNewhyperopt"
    ]
    start_new_hyperopt(get_args(args))

    assert wt_mock.call_count == 1
    assert "CoolNewhyperopt" in wt_mock.call_args_list[0][0][0]
    assert log_has_re("Writing hyperopt to .*", caplog)


def test_start_new_hyperopt_DefaultHyperopt(mocker, caplog):
    args = [
        "new-hyperopt",
        "--hyperopt",
        "DefaultHyperopt"
    ]
    with pytest.raises(OperationalException,
                       match=r"DefaultHyperopt is not allowed as name\."):
        start_new_hyperopt(get_args(args))


def test_start_new_hyperopt_no_arg(mocker, caplog):
    args = [
        "new-hyperopt",
    ]
    with pytest.raises(OperationalException,
                       match="`new-hyperopt` requires --hyperopt to be set."):
        start_new_hyperopt(get_args(args))


def test_download_data_keyboardInterrupt(mocker, caplog, markets):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                           MagicMock(side_effect=KeyboardInterrupt))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
    ]
    with pytest.raises(SystemExit):
        start_download_data(get_args(args))

    assert dl_mock.call_count == 1


def test_download_data_no_markets(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                           MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker, id='binance')
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20"
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_args[1]['timerange'].starttype == "date"
    assert log_has("Pairs [ETH/BTC,XRP/BTC] not available on exchange Binance.", caplog)


def test_download_data_no_exchange(mocker, caplog):
    mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                 MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_download_data(pargs)


def test_download_data_no_pairs(mocker, caplog):

    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                 MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange",
        "binance",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"Downloading data requires a list of pairs\..*"):
        start_download_data(pargs)


def test_download_data_trades(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_trades_data',
                           MagicMock(return_value=[]))
    convert_mock = mocker.patch('freqtrade.utils.convert_trades_to_ohlcv',
                                MagicMock(return_value=[]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange", "kraken",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20",
        "--dl-trades"
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_args[1]['timerange'].starttype == "date"
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 1


def test_start_test_pairlist(mocker, caplog, markets, tickers, default_conf, capsys):
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )

    default_conf['pairlists'] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
        },
        {"method": "PrecisionFilter"},
        {"method": "PriceFilter", "low_price_ratio": 0.02},
    ]

    patched_configuration_load_config_file(mocker, default_conf)
    args = [
        'test-pairlist',
        '-c', 'config.json.example'
    ]

    start_test_pairlist(get_args(args))

    assert log_has_re(r"^Using resolved pairlist VolumePairList.*", caplog)
    assert log_has_re(r"^Using resolved pairlist PrecisionFilter.*", caplog)
    assert log_has_re(r"^Using resolved pairlist PriceFilter.*", caplog)
    captured = capsys.readouterr()
    assert re.match(r"Pairs for .*", captured.out)
    assert re.match("['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'XRP/BTC']", captured.out)


def test_hyperopt_list(mocker, capsys, hyperopt_results):
    mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.load_previous_results',
        MagicMock(return_value=hyperopt_results)
    )

    args = [
        "hyperopt-list",
        "--no-details"
    ]
    start_hyperopt_list(get_args(args))
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 4/12", " 5/12",
                         " 6/12", " 7/12", " 8/12", " 9/12", " 10/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--best",
        "--no-details"
    ]
    start_hyperopt_list(get_args(args))
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 5/12", " 10/12"])
    assert all(x not in captured.out
               for x in [" 2/12", " 3/12", " 4/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details"
    ]
    start_hyperopt_list(get_args(args))
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12", " 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])


def test_hyperopt_show(mocker, capsys, hyperopt_results):
    mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.load_previous_results',
        MagicMock(return_value=hyperopt_results)
    )

    args = [
        "hyperopt-show",
    ]
    start_hyperopt_show(get_args(args))
    captured = capsys.readouterr()
    assert " 12/12" in captured.out

    args = [
        "hyperopt-show",
        "--best"
    ]
    start_hyperopt_show(get_args(args))
    captured = capsys.readouterr()
    assert " 10/12" in captured.out

    args = [
        "hyperopt-show",
        "-n", "1"
    ]
    start_hyperopt_show(get_args(args))
    captured = capsys.readouterr()
    assert " 1/12" in captured.out

    args = [
        "hyperopt-show",
        "--best",
        "-n", "2"
    ]
    start_hyperopt_show(get_args(args))
    captured = capsys.readouterr()
    assert " 5/12" in captured.out

    args = [
        "hyperopt-show",
        "--best",
        "-n", "-1"
    ]
    start_hyperopt_show(get_args(args))
    captured = capsys.readouterr()
    assert " 10/12" in captured.out
