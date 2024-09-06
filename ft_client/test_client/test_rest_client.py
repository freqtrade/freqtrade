import re
from unittest.mock import ANY, MagicMock

import pytest
from requests.exceptions import ConnectionError

from freqtrade_client import FtRestClient
from freqtrade_client.ft_client import add_arguments, main_exec


def log_has_re(line, logs):
    """Check if line matches some caplog's message."""
    return any(re.match(line, message) for message in logs.messages)


def get_rest_client():
    client = FtRestClient("http://localhost:8080", "freqtrader", "password")
    client._session = MagicMock()
    request_mock = MagicMock()
    client._session.request = request_mock
    return client, request_mock


def test_FtRestClient_init():
    client = FtRestClient("http://localhost:8080", "freqtrader", "password")
    assert client is not None
    assert client._serverurl == "http://localhost:8080"
    assert client._session is not None
    assert client._session.auth is not None
    assert client._session.auth == ("freqtrader", "password")


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_FtRestClient_call(method):
    client, mock = get_rest_client()
    client._call(method, "/dummytest")
    assert mock.call_count == 1

    getattr(client, f"_{method.lower()}")("/dummytest")
    assert mock.call_count == 2


def test_FtRestClient_call_invalid(caplog):
    client, _ = get_rest_client()
    with pytest.raises(ValueError):
        client._call("PUTTY", "/dummytest")

    client._session.request = MagicMock(side_effect=ConnectionError())
    client._call("GET", "/dummytest")

    assert log_has_re("Connection error", caplog)


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("start", [], {}),
        ("stop", [], {}),
        ("stopbuy", [], {}),
        ("reload_config", [], {}),
        ("balance", [], {}),
        ("count", [], {}),
        ("entries", [], {}),
        ("exits", [], {}),
        ("mix_tags", [], {}),
        ("locks", [], {}),
        ("lock_add", ["XRP/USDT", "2024-01-01 20:00:00Z", "*", "rand"], {}),
        ("delete_lock", [2], {}),
        ("daily", [], {}),
        ("daily", [15], {}),
        ("weekly", [], {}),
        ("weekly", [15], {}),
        ("monthly", [], {}),
        ("monthly", [12], {}),
        ("edge", [], {}),
        ("profit", [], {}),
        ("stats", [], {}),
        ("performance", [], {}),
        ("status", [], {}),
        ("version", [], {}),
        ("show_config", [], {}),
        ("ping", [], {}),
        ("logs", [], {}),
        ("logs", [55], {}),
        ("trades", [], {}),
        ("trades", [5], {}),
        ("trades", [5, 5], {}),  # With offset
        ("trade", [1], {}),
        ("delete_trade", [1], {}),
        ("cancel_open_order", [1], {}),
        ("whitelist", [], {}),
        ("blacklist", [], {}),
        ("blacklist", ["XRP/USDT"], {}),
        ("blacklist", ["XRP/USDT", "BTC/USDT"], {}),
        ("forcebuy", ["XRP/USDT"], {}),
        ("forcebuy", ["XRP/USDT", 1.5], {}),
        ("forceenter", ["XRP/USDT", "short"], {}),
        ("forceenter", ["XRP/USDT", "short", 1.5], {}),
        ("forceenter", ["XRP/USDT", "short", 1.5], {"order_type": "market"}),
        ("forceenter", ["XRP/USDT", "short", 1.5], {"order_type": "market", "stake_amount": 100}),
        (
            "forceenter",
            ["XRP/USDT", "short", 1.5],
            {"order_type": "market", "stake_amount": 100, "leverage": 10.0},
        ),
        (
            "forceenter",
            ["XRP/USDT", "short", 1.5],
            {
                "order_type": "market",
                "stake_amount": 100,
                "leverage": 10.0,
                "enter_tag": "test_force_enter",
            },
        ),
        ("forceexit", [1], {}),
        ("forceexit", [1, "limit"], {}),
        ("forceexit", [1, "limit", 100], {}),
        ("strategies", [], {}),
        ("strategy", ["sampleStrategy"], {}),
        ("pairlists_available", [], {}),
        ("plot_config", [], {}),
        ("available_pairs", [], {}),
        ("available_pairs", ["5m"], {}),
        ("pair_candles", ["XRP/USDT", "5m"], {}),
        ("pair_candles", ["XRP/USDT", "5m", 500], {}),
        ("pair_candles", ["XRP/USDT", "5m", 500], {"columns": ["close_time,close"]}),
        ("pair_history", ["XRP/USDT", "5m", "SampleStrategy"], {}),
        ("pair_history", ["XRP/USDT", "5m"], {"strategy": "SampleStrategy"}),
        ("sysinfo", [], {}),
        ("health", [], {}),
    ],
)
def test_FtRestClient_call_explicit_methods(method, args, kwargs):
    client, mock = get_rest_client()
    executor = getattr(client, method)
    executor(*args, **kwargs)
    assert mock.call_count == 1


def test_ft_client(mocker, capsys, caplog):
    with pytest.raises(SystemExit):
        args = add_arguments(["-V"])

    args = add_arguments(["--show"])
    assert isinstance(args, dict)
    assert args["show"] is True
    with pytest.raises(SystemExit):
        main_exec(args)
    captured = capsys.readouterr()
    assert "Possible commands" in captured.out

    mock = mocker.patch("freqtrade_client.ft_client.FtRestClient._call")
    args = add_arguments(["--config", "tests/testdata/testconfigs/main_test_config.json", "ping"])
    main_exec(args)
    captured = capsys.readouterr()
    assert mock.call_count == 1

    with pytest.raises(SystemExit):
        args = add_arguments(["--config", "tests/testdata/testconfigs/nonexisting.json"])
        main_exec(args)

    assert log_has_re(r"Could not load config file .*nonexisting\.json\.", caplog)

    args = add_arguments(
        ["--config", "tests/testdata/testconfigs/main_test_config.json", "whatever"]
    )
    main_exec(args)
    assert log_has_re("Command whatever not defined", caplog)


@pytest.mark.parametrize(
    "params, expected_args, expected_kwargs",
    [
        ("forceenter BTC/USDT long", ["BTC/USDT", "long"], {}),
        ("forceenter BTC/USDT long limit", ["BTC/USDT", "long", "limit"], {}),
        (
            # Skip most parameters, only providing enter_tag
            "forceenter BTC/USDT long enter_tag=deadBeef",
            ["BTC/USDT", "long"],
            {"enter_tag": "deadBeef"},
        ),
        (
            "forceenter BTC/USDT long invalid_key=123",
            [],
            SystemExit,
            # {"invalid_key": "deadBeef"},
        ),
    ],
)
def test_ft_client_argparsing(mocker, params, expected_args, expected_kwargs, caplog):
    mocked_method = params.split(" ")[0]
    mocker.patch("freqtrade_client.ft_client.load_config", return_value={}, autospec=True)
    mm = mocker.patch(
        f"freqtrade_client.ft_client.FtRestClient.{mocked_method}", return_value={}, autospec=True
    )
    args = add_arguments(params.split(" "))
    if isinstance(expected_kwargs, dict):
        main_exec(args)
        mm.assert_called_once_with(ANY, *expected_args, **expected_kwargs)
    else:
        with pytest.raises(expected_kwargs):
            main_exec(args)

        assert log_has_re(f"Error executing command {mocked_method}: got an unexpected .*", caplog)
        mm.assert_not_called()
