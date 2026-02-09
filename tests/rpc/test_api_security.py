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
        == "default-src 'self'; base-uri 'self'; form-action 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; "
        "frame-ancestors 'none'"
    )
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert headers["Permissions-Policy"] == (
        "geolocation=(), microphone=(), camera=(), payment=(), "
        "usb=(), vr=(), display-capture=(), serial=(), autoplay=(), fullscreen=(), sync-xhr=()"
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


def test_validate_wildcard_pair():
    from freqtrade.rpc.api_server.api_schemas import validate_wildcard_pair

    # Valid
    assert validate_wildcard_pair(["BTC/USDT"]) == ["BTC/USDT"]
    assert validate_wildcard_pair(["BTC/.*"]) == ["BTC/.*"]
    assert validate_wildcard_pair([".*"]) == [".*"]
    assert validate_wildcard_pair(["BTC-PERP/USDT:USDT"]) == ["BTC-PERP/USDT:USDT"]

    # Invalid
    with pytest.raises(ValueError, match="Invalid pair name"):
        validate_wildcard_pair(["BTC/USDT;"])
    with pytest.raises(ValueError, match="Invalid pair name"):
        validate_wildcard_pair(["<script>"])
    with pytest.raises(ValueError, match="Invalid pair name"):
        validate_wildcard_pair(["(BTC)/USDT"])


def test_blacklist_payload_validation():
    from pydantic import ValidationError

    from freqtrade.rpc.api_server.api_schemas import BlacklistPayload

    # Valid
    payload = BlacklistPayload(blacklist=["BTC/.*"])
    assert payload.blacklist == ["BTC/.*"]

    # Invalid
    with pytest.raises(ValidationError):
        BlacklistPayload(blacklist=["<script>alert(1)</script>"])
    with pytest.raises(ValidationError):
        BlacklistPayload(blacklist=["BTC/USDT; DROP TABLE"])


def test_force_entry_payload_validation():
    from pydantic import ValidationError

    from freqtrade.rpc.api_server.api_schemas import ForceEnterPayload

    # Valid
    payload = ForceEnterPayload(pair="BTC/USDT", entry_tag="safe_tag")
    assert payload.entry_tag == "safe_tag"

    # Invalid entry_tag
    with pytest.raises(ValidationError):
        ForceEnterPayload(pair="BTC/USDT", entry_tag="<script>")
    with pytest.raises(ValidationError):
        ForceEnterPayload(pair="BTC/USDT", entry_tag="tag with spaces")


@pytest.mark.asyncio
async def test_rate_limiter():
    from fastapi import HTTPException

    from freqtrade.rpc.api_server.deps import RateLimiter

    limiter = RateLimiter(max_calls=2, time_seconds=1)
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.url.path = "/test"

    # Call 1 - OK
    await limiter(request)

    # Call 2 - OK
    await limiter(request)

    # Call 3 - Fail
    with pytest.raises(HTTPException) as excinfo:
        await limiter(request)
    assert excinfo.value.status_code == 429

    # Test different IP
    request2 = MagicMock()
    request2.client.host = "192.168.1.1"
    request2.url.path = "/test"

    # Call 1 new IP - OK
    await limiter(request2)


def test_force_exit_payload_negative_values():
    from pydantic import ValidationError

    from freqtrade.rpc.api_server.api_schemas import ForceExitPayload

    # This should raise ValidationError because we added gt=0 constraint
    with pytest.raises(ValidationError) as excinfo:
        ForceExitPayload(tradeid="1", price=-100, amount=-10)

    assert "Input should be greater than 0" in str(excinfo.value)


def test_download_data_payload_invalid_timerange():
    from pydantic import ValidationError

    from freqtrade.rpc.api_server.api_schemas import DownloadDataPayload

    # This should raise ValidationError because we added validation
    with pytest.raises(ValidationError) as excinfo:
        DownloadDataPayload(pairs=["BTC/USDT"], timerange="invalid-timerange")

    # TimeRange.parse_timerange raises ConfigurationError
    # Pydantic wraps it. The message should contain "Incorrect syntax for timerange"
    assert 'Incorrect syntax for timerange "invalid-timerange"' in str(excinfo.value)


def test_verify_auth_logic():
    from freqtrade.rpc.api_server.api_auth import verify_auth

    config = {"username": "user", "password": "password"}

    assert verify_auth(config, "user", "password") is True
    assert verify_auth(config, "user", "wrong") is False
    assert verify_auth(config, "wrong", "password") is False
    assert verify_auth(config, "wrong", "wrong") is False
    assert verify_auth({}, "user", "password") is False
