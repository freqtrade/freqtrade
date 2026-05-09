from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BYBIT_LONG_SHORT_RATIO_ENDPOINT = "/v5/market/account-ratio"
BYBIT_PUBLIC_BASE_URL = "https://api.bybit.com"
SUPPORTED_LONG_SHORT_RATIO_PERIODS = {"5min", "15min", "30min", "1h", "4h", "1d"}


@dataclass(frozen=True)
class BybitLongShortRatioDownloadInputs:
    root_dir: Path
    symbol: str
    category: str
    period: str
    start_time: datetime
    end_time: datetime
    output_path: Path
    base_url: str = BYBIT_PUBLIC_BASE_URL
    limit: int = 500
    max_pages: int = 200
    timeout_seconds: float = 20.0


RequestJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def download_bybit_long_short_ratio(
    inputs: BybitLongShortRatioDownloadInputs,
    *,
    request_json: RequestJson | None = None,
) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    output_path = _resolve_inside(inputs.output_path, root)
    category = inputs.category.strip().lower()
    symbol = inputs.symbol.strip().upper()
    period = inputs.period.strip()
    limit = max(1, min(500, int(inputs.limit)))
    max_pages = max(1, int(inputs.max_pages))
    request_json = request_json or _request_json

    blockers = _input_blockers(
        category=category,
        symbol=symbol,
        period=period,
        start_time=inputs.start_time,
        end_time=inputs.end_time,
    )
    if blockers:
        return _artifact(
            inputs,
            output_path=output_path,
            status="blocked",
            rows=[],
            blocker_messages=blockers,
            page_count=0,
            truncated=False,
        )

    params: dict[str, Any] = {
        "category": category,
        "symbol": symbol,
        "period": period,
        "startTime": _datetime_to_milliseconds(inputs.start_time),
        "endTime": _datetime_to_milliseconds(inputs.end_time),
        "limit": limit,
    }
    rows: list[dict[str, Any]] = []
    cursor = ""
    page_count = 0
    truncated = False
    while True:
        request_params = dict(params)
        if cursor:
            request_params["cursor"] = cursor
        try:
            payload = request_json(
                _join_url(inputs.base_url, BYBIT_LONG_SHORT_RATIO_ENDPOINT),
                request_params,
                inputs.timeout_seconds,
            )
            ret_code = int(payload.get("retCode", -1))
        except Exception as exc:
            blockers.append(_request_failure_blocker(exc))
            break
        if ret_code != 0:
            ret_msg = str(payload.get("retMsg") or "unknown_error")
            blockers.append(f"bybit_ret_code:{ret_code}:{ret_msg}")
            break
        result = payload.get("result") or {}
        rows.extend(list(result.get("list") or []))
        page_count += 1
        cursor = str(result.get("nextPageCursor") or "")
        if not cursor:
            break
        if page_count >= max_pages:
            truncated = True
            blockers.append("max_pages_reached")
            break

    artifact = _artifact(
        inputs,
        output_path=output_path,
        status="completed" if not blockers else "blocked",
        rows=rows,
        blocker_messages=blockers,
        page_count=page_count,
        truncated=truncated,
    )
    if artifact["status"] == "completed":
        frame = _rows_to_frame(rows, symbol=symbol, category=category, period=period)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            frame.to_csv(output_path, index=False)
        else:
            frame.to_parquet(output_path, index=False)
        artifact["output_written"] = True
        artifact["row_count"] = int(len(frame))
        artifact["start"] = _timestamp_to_str(frame["date"].min()) if not frame.empty else None
        artifact["end"] = _timestamp_to_str(frame["date"].max()) if not frame.empty else None
    return artifact


def default_bybit_long_short_ratio_path(
    symbol: str,
    *,
    category: str = "linear",
    period: str = "1h",
    data_root: Path = Path("data/market_structure/bybit/futures"),
) -> Path:
    clean_symbol = symbol.strip().upper()
    if category.strip().lower() == "linear" and clean_symbol.endswith("USDT"):
        base = clean_symbol[:-4]
        file_name = f"{base}_USDT_USDT-{period}-long_short_ratio.parquet"
    else:
        file_name = f"{clean_symbol}-{period}-long_short_ratio.parquet"
    return data_root / file_name


def _request_json(url: str, params: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _rows_to_frame(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    category: str,
    period: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "long_account_ratio",
                "short_account_ratio",
                "long_short_ratio",
                "symbol",
                "category",
                "period",
            ]
        )
    frame = frame.rename(
        columns={"buyRatio": "long_account_ratio", "sellRatio": "short_account_ratio"}
    )
    frame["date"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    frame["long_account_ratio"] = pd.to_numeric(frame["long_account_ratio"], errors="coerce")
    frame["short_account_ratio"] = pd.to_numeric(frame["short_account_ratio"], errors="coerce")
    frame = frame.dropna(subset=["date", "long_account_ratio", "short_account_ratio"])
    frame = frame[
        (frame["long_account_ratio"] >= 0.0)
        & (frame["short_account_ratio"] > 0.0)
        & (frame["long_account_ratio"] <= 1.0)
        & (frame["short_account_ratio"] <= 1.0)
    ]
    frame["long_short_ratio"] = frame["long_account_ratio"] / frame["short_account_ratio"].where(
        frame["short_account_ratio"] > 0.0
    )
    frame["symbol"] = symbol
    frame["category"] = category
    frame["period"] = period
    return (
        frame[
            [
                "date",
                "long_account_ratio",
                "short_account_ratio",
                "long_short_ratio",
                "symbol",
                "category",
                "period",
            ]
        ]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _input_blockers(
    *,
    category: str,
    symbol: str,
    period: str,
    start_time: datetime,
    end_time: datetime,
) -> list[str]:
    blockers: list[str] = []
    if category not in {"linear", "inverse"}:
        blockers.append("category_must_be_linear_or_inverse")
    if not symbol:
        blockers.append("symbol_required")
    if period not in SUPPORTED_LONG_SHORT_RATIO_PERIODS:
        blockers.append("unsupported_period")
    if start_time >= end_time:
        blockers.append("start_time_must_be_before_end_time")
    return blockers


def _request_failure_blocker(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if message:
        return f"request_failed:{type(exc).__name__}:{message}"
    return f"request_failed:{type(exc).__name__}"


def _artifact(
    inputs: BybitLongShortRatioDownloadInputs,
    *,
    output_path: Path,
    status: str,
    rows: list[dict[str, Any]],
    blocker_messages: list[str],
    page_count: int,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "factory": "bybit_long_short_ratio_download",
        "status": status,
        "symbol": inputs.symbol.strip().upper(),
        "category": inputs.category.strip().lower(),
        "period": inputs.period.strip(),
        "start_time": inputs.start_time.astimezone(UTC).isoformat(),
        "end_time": inputs.end_time.astimezone(UTC).isoformat(),
        "endpoint": BYBIT_LONG_SHORT_RATIO_ENDPOINT,
        "base_url": inputs.base_url,
        "output_path": str(output_path),
        "output_written": False,
        "row_count": len(rows),
        "page_count": page_count,
        "truncated": truncated,
        "blockers": blocker_messages,
        "safety_scope": {
            "public_market_data_only": True,
            "api_key_required": False,
            "api_key_used": False,
            "order_endpoint_used": False,
            "exchange_order_process_started": False,
            "leverage_changed": False,
            "shorting_enabled": False,
        },
    }


def _datetime_to_milliseconds(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def _timestamp_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
