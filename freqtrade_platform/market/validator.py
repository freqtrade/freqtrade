"""Validation rules for platform-level OHLCV market data."""

from __future__ import annotations

import pandas as pd


class MarketDataValidator:
    """Validate raw OHLCV history before feature extraction or regime detection."""

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    def validate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe is None:
            raise ValueError("invalid OHLCV market data: dataframe is required")

        data = dataframe.copy()
        if data.empty:
            raise ValueError("invalid OHLCV market data: empty dataframe")

        missing = [column for column in self.REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"invalid OHLCV market data: missing columns {missing}")

        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if data["date"].isna().any():
            raise ValueError("invalid OHLCV market data: non-parsable datetime values")

        data = data.sort_values("date").reset_index(drop=True)
        if not data["date"].is_monotonic_increasing:
            raise ValueError("invalid OHLCV market data: timeline is not strictly increasing")
        if data["date"].duplicated().any():
            raise ValueError("invalid OHLCV market data: duplicate OHLCV rows detected")

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[numeric_columns].isna().any().any():
            raise ValueError("invalid OHLCV market data: non-numeric OHLCV values")

        if (data["volume"] <= 0).any():
            raise ValueError("invalid OHLCV market data: zero volume candles are invalid")

        if ((data["high"] < data[["open", "close"]].max(axis=1)) | (data["low"] > data[["open", "close"]].min(axis=1))).any():
            raise ValueError("invalid OHLCV market data: OHLC boundaries are inconsistent")

        if not (data["close"].between(data["low"], data["high"]) & data["open"].between(data["low"], data["high"])).all():
            raise ValueError("invalid OHLCV market data: open/close must remain within candle bounds")

        return data
