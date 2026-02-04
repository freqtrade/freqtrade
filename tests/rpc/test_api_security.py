from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from freqtrade.enums import RunMode
from freqtrade.loggers import setup_logging
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.rpc import RPC
from tests.conftest import get_patched_freqtradebot


BASE_URI = "/api/v1"


@pytest.fixture
def botclient_security(default_conf, mocker):
    setup_logging(default_conf)
    default_conf["runmode"] = RunMode.DRY_RUN
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": "user",
                "password": "password",
                "jwt_secret_key": "super-secret",
                "CORS_origins": ["http://example.com"],
            }
        }
    )

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", MagicMock())
    apiserver = None
    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        with TestClient(apiserver.app, raise_server_exceptions=False) as client:
            yield ftbot, client
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()


def test_security_headers(botclient_security):
    _ftbot, client = botclient_security

    rc = client.get(f"{BASE_URI}/ping")
    assert rc.status_code == 200
    headers = rc.headers

    assert (
        headers["Content-Security-Policy"]
        == "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:;"
    )
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert headers["Permissions-Policy"] == (
        "geolocation=(), microphone=(), camera=(), payment=(), "
        "usb=(), vr=(), display-capture=(), serial=(), autoplay=(), fullscreen=()"
    )
    assert headers["Referrer-Policy"] == "same-origin"


def test_cors_restrictions(botclient_security):
    _ftbot, client = botclient_security

    # Preflight for GET (allowed)
    rc = client.options(
        f"{BASE_URI}/ping",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert rc.status_code == 200
    assert "access-control-allow-methods" in rc.headers
    assert "GET" in rc.headers["access-control-allow-methods"]

    # Preflight for TRACE (not allowed)
    rc = client.options(
        f"{BASE_URI}/ping",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "TRACE",
        },
    )
    # It might return 200 but allow methods shouldn't have TRACE
    if "access-control-allow-methods" in rc.headers:
        assert "TRACE" not in rc.headers["access-control-allow-methods"]


def test_generic_exception_handling(botclient_security, mocker):
    _ftbot, client = botclient_security

    # Patch RPC._rpc_show_config to raise exception
    mocker.patch(
        "freqtrade.rpc.rpc.RPC._rpc_show_config", side_effect=Exception("Secret Stack Trace")
    )

    from requests.auth import _basic_auth_str

    rc = client.get(
        f"{BASE_URI}/show_config", headers={"Authorization": _basic_auth_str("user", "password")}
    )
    assert rc.status_code == 500
    assert rc.json() == {"error": "Internal Server Error", "status": "error"}
    # The stack trace should NOT be in the response
    assert "Secret Stack Trace" not in rc.text


def test_pair_validation(botclient_security):
    _ftbot, client = botclient_security
    from requests.auth import _basic_auth_str

    headers = {"Authorization": _basic_auth_str("user", "password")}

    # Valid pair
    rc = client.get(f"{BASE_URI}/entries?pair=XRP/BTC", headers=headers)
    assert rc.status_code == 200

    # Invalid pair (injection attempt)
    rc = client.get(f"{BASE_URI}/entries?pair=XRP/BTC;DROP%20TABLE", headers=headers)
    assert rc.status_code == 422
    assert rc.json()["detail"][0]["msg"] == "String should match pattern '^[a-zA-Z0-9/_:]+$'"

    # Valid pair with numbers and :
    rc = client.get(f"{BASE_URI}/entries?pair=XRP/USDT:USDT", headers=headers)
    assert rc.status_code == 200
