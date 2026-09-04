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

    def _quality_report(
        self,
        data: pd.DataFrame,
        expected_timeframe: str | None = None,
        *,
        as_of: datetime | str | pd.Timestamp | None = None,
    ) -> dict[str, object]:
        deltas = data["date"].diff().dropna()
        expected_seconds = self.EXPECTED_TIMEFRAME_SECONDS.get(expected_timeframe) if expected_timeframe else None
        missing = 0
        missing_intervals: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
        if expected_seconds is not None and len(deltas) > 0:
            expected_delta = pd.Timedelta(seconds=expected_seconds)
            for previous, current in zip(data["date"].iloc[:-1], data["date"].iloc[1:]):
                delta = current - previous
                if delta > expected_delta:
                    cumulative_missing = int(delta / expected_delta) - 1
                    if cumulative_missing > 0:
                        missing += cumulative_missing
                        missing_intervals.append((previous, current, cumulative_missing))

        zero_volume = bool((data["volume"] <= 0).any())
        stale = False
        if expected_seconds is not None and as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            if as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize("UTC")
            else:
                as_of_ts = as_of_ts.tz_convert("UTC")
            latest = data["date"].iloc[-1]
            stale = (as_of_ts - latest) > pd.Timedelta(seconds=expected_seconds * 3)

        quality = {
            "duplicate_timestamps": bool(data["date"].duplicated().any()),
            "missing_candles": int(missing),
            "missing_intervals": missing_intervals,
            "has_gaps": bool(missing > 0),
            "zero_volume": zero_volume,
            "stale": stale,
            "insufficient_history": len(data) < 3,
            "timeframe_expected": expected_timeframe,
            "timeframe_alignment_ok": True,
            "quality_state": "ok",
        }
        if expected_seconds is not None and len(deltas) > 0:
            expected_delta = pd.Timedelta(seconds=expected_seconds)
            observed = deltas.median()
            if pd.notna(observed):
                quality["timeframe_alignment_ok"] = abs((observed - expected_delta).total_seconds()) <= max(1.0, expected_delta.total_seconds() * 0.15)
        if zero_volume:
            quality["quality_state"] = "zero_volume"
        if quality["missing_candles"] > 0 or not quality["timeframe_alignment_ok"]:
            quality["quality_state"] = "missing_data" if quality["missing_candles"] > 0 else "degraded"
        if stale:
            quality["quality_state"] = "stale"
        if len(data) < 3:
            quality["quality_state"] = "insufficient_history"
        return quality

    def validate(
        self,
        dataframe: pd.DataFrame,
        *,
        expected_timeframe: str | None = None,
        as_of: datetime | str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
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

        if not data["date"].is_monotonic_increasing:
            raise ValueError("invalid OHLCV market data: timeline is not strictly increasing (out-of-order input)")

        data = data.sort_values("date").reset_index(drop=True)

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[numeric_columns].isna().any().any():
            raise ValueError("invalid OHLCV market data: non-numeric OHLCV values")

        if (data[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("invalid OHLCV market data: negative or zero prices are invalid")

        if ((data["high"] < data[["open", "close"]].max(axis=1)) | (data["low"] > data[["open", "close"]].min(axis=1))).any():
            raise ValueError("invalid OHLCV market data: OHLC boundaries are inconsistent")

        if not (data["close"].between(data["low"], data["high"]) & data["open"].between(data["low"], data["high"])).all():
            raise ValueError("invalid OHLCV market data: open/close must remain within candle bounds")

        if expected_timeframe:
            expected_seconds = self.EXPECTED_TIMEFRAME_SECONDS.get(expected_timeframe)
            if expected_seconds is not None and len(data) > 1:
                expected_delta = pd.Timedelta(seconds=expected_seconds)
                deltas = data["date"].diff().dropna()
                if not deltas.empty:
                    observed_deltas = deltas.unique()
                    for observed_delta in observed_deltas:
                        if observed_delta > expected_delta * 1.5 and observed_delta <= expected_delta * 4:
                            quality = self._quality_report(data, expected_timeframe=expected_timeframe, as_of=as_of)
                            data.attrs["quality"] = quality
                            break

        if "closed" not in data.columns:
            data["closed"] = True
        data["closed"] = data["closed"].fillna(True).astype(bool)
        quality = self._quality_report(data, expected_timeframe=expected_timeframe, as_of=as_of)
        data.attrs["quality"] = quality
        return data
