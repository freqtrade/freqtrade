"""Validation rules for platform-level OHLCV market data."""

from __future__ import annotations

import pandas as pd


class MarketDataValidator:
    """Validate raw OHLCV history before feature extraction or regime detection."""

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
    EXPECTED_TIMEFRAME_SECONDS = {
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    def _quality_report(self, data: pd.DataFrame, expected_timeframe: str | None = None) -> dict[str, object]:
        deltas = data["date"].diff().dropna()
        expected_seconds = self.EXPECTED_TIMEFRAME_SECONDS.get(expected_timeframe) if expected_timeframe else None
        missing = 0
        if len(deltas) > 1:
            median_delta = deltas.median()
            if expected_seconds is not None:
                expected_delta = pd.Timedelta(seconds=expected_seconds)
                if not pd.isna(median_delta) and median_delta > expected_delta * 1.5:
                    missing = int((median_delta / expected_delta) - 1)
        quality = {
            "duplicate_timestamps": bool(data["date"].duplicated().any()),
            "missing_candles": int(missing),
            "has_gaps": bool(missing > 0),
            "stale": False,
            "insufficient_history": len(data) < 12,
            "timeframe_expected": expected_timeframe,
            "timeframe_alignment_ok": True,
            "quality_state": "ok",
        }
        if expected_seconds is not None and len(deltas) > 1:
            observed = deltas.median()
            if not pd.isna(observed):
                expected_delta = pd.Timedelta(seconds=expected_seconds)
                quality["timeframe_alignment_ok"] = abs((observed - expected_delta).total_seconds()) <= max(1.0, expected_delta.total_seconds() * 0.1)
        if quality["missing_candles"] > 0 or not quality["timeframe_alignment_ok"]:
            quality["quality_state"] = "degraded"
        return quality

    def validate(self, dataframe: pd.DataFrame, *, expected_timeframe: str | None = None) -> pd.DataFrame:
        if dataframe is None:
            raise ValueError("invalid OHLCV market data: dataframe is required")

        data = dataframe.copy()
        if data.empty:
            raise ValueError("invalid OHLCV market data: empty dataframe")

        missing = [column for column in self.REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"invalid OHLCV market data: missing columns {missing}")

        data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
        if data["date"].isna().any():
            raise ValueError("invalid OHLCV market data: non-parsable datetime values")

        if data["date"].duplicated().any():
            raise ValueError("invalid OHLCV market data: duplicate OHLCV rows detected")

        data = data.sort_values("date").reset_index(drop=True)
        if not data["date"].is_monotonic_increasing:
            raise ValueError("invalid OHLCV market data: timeline is not strictly increasing")

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[numeric_columns].isna().any().any():
            raise ValueError("invalid OHLCV market data: non-numeric OHLCV values")

        if (data["volume"] <= 0).any():
            raise ValueError("invalid OHLCV market data: zero or negative volume candles are invalid")

        if ((data["high"] < data[["open", "close"]].max(axis=1)) | (data["low"] > data[["open", "close"]].min(axis=1))).any():
            raise ValueError("invalid OHLCV market data: OHLC boundaries are inconsistent")

        if not (data["close"].between(data["low"], data["high"]) & data["open"].between(data["low"], data["high"])).all():
            raise ValueError("invalid OHLCV market data: open/close must remain within candle bounds")

        if expected_timeframe:
            expected_seconds = self.EXPECTED_TIMEFRAME_SECONDS.get(expected_timeframe)
            if expected_seconds is not None and len(data) > 1:
                observed_delta = data["date"].diff().dropna().median()
                if pd.notna(observed_delta):
                    expected_delta = pd.Timedelta(seconds=expected_seconds)
                    if abs((observed_delta - expected_delta).total_seconds()) > max(1.0, expected_delta.total_seconds() * 0.15):
                        raise ValueError(f"invalid OHLCV market data: timeframe mismatch for {expected_timeframe}")

        if "closed" not in data.columns:
            data["closed"] = True
        data["closed"] = data["closed"].fillna(True).astype(bool)
        quality = self._quality_report(data, expected_timeframe=expected_timeframe)
        data.attrs["quality"] = quality
        return data
