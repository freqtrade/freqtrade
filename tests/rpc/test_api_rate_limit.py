
import pytest
from fastapi.testclient import TestClient
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.rpc import RPC
from freqtrade.enums import RunMode
from freqtrade.loggers import setup_logging
from unittest.mock import MagicMock

from requests.auth import _basic_auth_str

from tests.conftest import get_patched_freqtradebot


BASE_URI = "/api/v1"
_TEST_USER = "FreqTrader"
_TEST_PASS = "SuperSecurePassword1!"


@pytest.fixture
def botclient_ratelimit(default_conf, mocker):
    setup_logging(default_conf)
    default_conf["runmode"] = RunMode.DRY_RUN
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": _TEST_USER,
                "password": _TEST_PASS,
                "jwt_secret_key": "super-secret",
            }
        }
    )

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", MagicMock())
    apiserver = None

    # Reset cache for each test
    from freqtrade.rpc.api_server.api_auth import login_attempts_cache

    login_attempts_cache.clear()

    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        with TestClient(apiserver.app) as client:
            yield ftbot, client
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()


def test_login_rate_limit(botclient_ratelimit):
    _ftbot, client = botclient_ratelimit

    # Fail 5 times
    for _ in range(5):
        rc = client.post(
            f"{BASE_URI}/token/login", headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
        )
        assert rc.status_code == 401

    # 6th attempt should be rate limited
    rc = client.post(
        f"{BASE_URI}/token/login", headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
    )
    assert rc.status_code == 429
    assert "Too many login attempts" in rc.json()["detail"]

    # Even correct password should fail now
    rc = client.post(
        f"{BASE_URI}/token/login", headers={"Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS)}
    )
    assert rc.status_code == 429


def test_login_success_resets_limit(botclient_ratelimit):
    _ftbot, client = botclient_ratelimit

    # Fail 4 times
    for _ in range(4):
        client.post(
            f"{BASE_URI}/token/login",
            headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
        )

    # Succeed
    rc = client.post(
        f"{BASE_URI}/token/login",
        headers={"Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS)}
    )
    assert rc.status_code == 200

    # Fail 1 time (would be 5th if not reset)
    rc = client.post(
        f"{BASE_URI}/token/login",
        headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
    )
    assert rc.status_code == 401

    # Check if we can still try (should allow 4 more)
    for _ in range(4):
        client.post(
            f"{BASE_URI}/token/login",
            headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
        )

    # 6th attempt (after 5 failures)
    rc = client.post(
        f"{BASE_URI}/token/login",
        headers={"Authorization": _basic_auth_str(_TEST_USER, "WrongPass")}
    )
    assert rc.status_code == 429
