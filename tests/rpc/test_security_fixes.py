from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from freqtrade.configuration.config_secrets import sanitize_config
from freqtrade.rpc.api_server.api_schemas import (
    DownloadDataPayload,
    ForceEnterPayload,
    LocksPayload,
    PairCandlesRequest,
)
from freqtrade.rpc.api_server.deps import RateLimiter


# --- RateLimiter Tests ---
@pytest.mark.asyncio
async def test_rate_limiter_isolated_state():
    """
    Verify that RateLimiter uses isolated state across instances with same parameters.
    """
    limiter1 = RateLimiter(max_calls=2, time_seconds=60)
    limiter2 = RateLimiter(max_calls=2, time_seconds=60)

    # Mock Request
    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"
    request.url.path = "/test-endpoint"

    # Call 1: Success for limiter1
    await limiter1(request)

    # Call 2: Success for limiter1
    await limiter1(request)

    # Call 3: Should fail for limiter1 (limit reached)
    with pytest.raises(HTTPException) as excinfo:
        await limiter1(request)
    assert excinfo.value.status_code == 429

    # Limiter2 should NOT be blocked (state is isolated)
    await limiter2(request)  # 1
    await limiter2(request)  # 2

    # Call 3 for limiter2 should fail
    with pytest.raises(HTTPException) as excinfo:
        await limiter2(request)
    assert excinfo.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_distinct_params():
    """
    Verify that RateLimiter keeps state distinct for different parameters.
    """
    limiter1 = RateLimiter(max_calls=2, time_seconds=60)
    limiter2 = RateLimiter(max_calls=5, time_seconds=60)

    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"
    request.url.path = "/test-endpoint"

    # Exhaust limiter1
    await limiter1(request)
    await limiter1(request)
    with pytest.raises(HTTPException):
        await limiter1(request)

    # Limiter2 should still work
    await limiter2(request)  # 1
    await limiter2(request)  # 2
    await limiter2(request)  # 3
    # It works


# --- Secret Redaction Tests ---
def test_sanitize_config_secrets():
    """
    Verify that sanitize_config redacts new sensitive keys.
    """
    config = {
        "api_server": {
            "password": "secret_password",
            "jwt_secret_key": "secret_jwt",
            "ws_token": "secret_ws_token",
            "listen_ip_address": "127.0.0.1",
        },
        "webhook": {"url": "https://secret.url"},
    }

    sanitized = sanitize_config(config)

    assert sanitized["api_server"]["password"] == "REDACTED"
    assert sanitized["api_server"]["jwt_secret_key"] == "REDACTED"
    assert sanitized["api_server"]["ws_token"] == "REDACTED"
    assert sanitized["webhook"]["url"] == "REDACTED"
    assert sanitized["api_server"]["listen_ip_address"] == "127.0.0.1"  # Not redacted


# --- Input Validation Tests ---
def test_force_enter_payload_validation():
    """
    Verify ForceEnterPayload validation.
    """
    # Valid
    ForceEnterPayload(pair="BTC/USDT", price=100, stakeamount=10)

    # Invalid stakeamount (negative)
    with pytest.raises(ValidationError):
        ForceEnterPayload(pair="BTC/USDT", stakeamount=-10)

    # Invalid price (negative)
    with pytest.raises(ValidationError):
        ForceEnterPayload(pair="BTC/USDT", price=-100)

    # Invalid pair (regex)
    with pytest.raises(ValidationError):
        ForceEnterPayload(pair="BTC(USDT", stakeamount=10)


def test_download_data_payload_validation():
    """
    Verify DownloadDataPayload validation.
    """
    # Valid
    DownloadDataPayload(pairs=["BTC/USDT"], days=10)

    # Invalid days (negative)
    with pytest.raises(ValidationError):
        DownloadDataPayload(pairs=["BTC/USDT"], days=-10)

    # Invalid timeframes
    with pytest.raises(ValidationError):
        DownloadDataPayload(pairs=["BTC/USDT"], timeframes=["invalid"])

    # Invalid timeframe unit
    with pytest.raises(ValidationError):
        DownloadDataPayload(pairs=["BTC/USDT"], timeframes=["5x"])


def test_pair_candles_request_validation():
    """
    Verify PairCandlesRequest validation.
    """
    # Valid
    PairCandlesRequest(pair="BTC/USDT", timeframe="5m", limit=100)

    # Invalid limit (negative)
    with pytest.raises(ValidationError):
        PairCandlesRequest(pair="BTC/USDT", timeframe="5m", limit=-1)

    # Invalid limit (too high)
    with pytest.raises(ValidationError):
        PairCandlesRequest(pair="BTC/USDT", timeframe="5m", limit=2000)

    # Invalid timeframe
    with pytest.raises(ValidationError):
        PairCandlesRequest(pair="BTC/USDT", timeframe="invalid", limit=100)


def test_locks_payload_validation():
    """
    Verify LocksPayload validation.
    """
    now = datetime.now(UTC)

    # Valid
    LocksPayload(pair="BTC/USDT", until=now, side="long")

    # Invalid side (too long)
    with pytest.raises(ValidationError):
        LocksPayload(pair="BTC/USDT", until=now, side="long" * 10)
