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
def botclient_sentinel(default_conf, mocker):
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


def test_input_max_length_entry_tag(botclient_sentinel):
    _ftbot, client = botclient_sentinel
    from requests.auth import _basic_auth_str

    headers = {"Authorization": _basic_auth_str("user", "password")}

    # 300 chars > 255 limit (to be implemented)
    long_tag = "a" * 300
    payload = {
        "pair": "XRP/BTC",
        "entry_tag": long_tag,
        "side": "long"
    }

    rc = client.post(f"{BASE_URI}/forceenter", json=payload, headers=headers)

    # CURRENTLY: It likely succeeds (200) or fails validation for other reasons but accepts length.
    # AFTER FIX: It should return 422 Unprocessable Entity due to max_length.

    # We assert that it FAILS with 422 if the fix is present.
    # If the fix is NOT present, this test MIGHT fail (if it returns 200).
    # Since we want to verify the fix, we check for 422 and specific error message.

    if rc.status_code == 200:
        pytest.fail("ForceEnterPayload accepted entry_tag > 255 chars")

    assert rc.status_code == 422
    assert "String should have at most 255 characters" in str(rc.json())


def test_xss_protection_header(botclient_sentinel):
    _ftbot, client = botclient_sentinel

    rc = client.get(f"{BASE_URI}/ping")
    assert rc.status_code == 200
    headers = rc.headers

    # CURRENTLY: Missing
    # AFTER FIX: Present
    if "X-XSS-Protection" not in headers:
         pytest.fail("X-XSS-Protection header missing")

    assert headers["X-XSS-Protection"] == "1; mode=block"


def test_trades_limit_cap(botclient_sentinel):
    _ftbot, client = botclient_sentinel
    from requests.auth import _basic_auth_str

    headers = {"Authorization": _basic_auth_str("user", "password")}

    # limit=2000 > 1000 limit (to be implemented)
    rc = client.get(f"{BASE_URI}/trades?limit=2000", headers=headers)

    # CURRENTLY: Returns 200 (if DB empty or has trades).
    # AFTER FIX: Should return 422.

    if rc.status_code == 200:
        pytest.fail("Trades endpoint accepted limit=2000")

    assert rc.status_code == 422
    assert "Input should be less than or equal to 1000" in str(rc.json())


def test_login_long_password(botclient_sentinel):
    _ftbot, client = botclient_sentinel
    from requests.auth import _basic_auth_str

    # 300 chars password
    long_pass = "a" * 300
    headers = {"Authorization": _basic_auth_str("user", long_pass)}

    rc = client.post(f"{BASE_URI}/token/login", headers=headers)

    # CURRENTLY: Returns 401 (Unauthorized) after checking password (slowly).
    # AFTER FIX: Should return 400 (Bad Request) or 422 before checking password.
    # However, HTTPBasicCredentials extraction happens before our handler code runs (in Depends).
    # Wait, `form_data: HTTPBasicCredentials = Depends(security)` extracts it.
    # Then `token_login` is called.
    # So we can check length inside `token_login`.
    # It will return 400 if we raise HTTPException(400).

    if rc.status_code == 401:
        pytest.fail("Login endpoint accepted long password for check (returned 401 instead of 400/422)")

    assert rc.status_code == 400
