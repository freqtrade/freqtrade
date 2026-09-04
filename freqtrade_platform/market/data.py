"""Canonical market-series adapter for Freqtrade OHLCV inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from freqtrade_platform.market.validator import MarketDataValidator


@dataclass(slots=True)
class CanonicalMarketSeries:
    """Normalized market-data view converted from Freqtrade OHLCV output."""

    symbol: str
    timeframe: str
    data: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)
    as_of: datetime | str | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol is required")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe is required")
        if self.data is None or self.data.empty:
            raise ValueError("market data cannot be empty")
        self.data = self.data.copy()
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(self.data.columns))
        if missing:
            raise ValueError(f"invalid OHLCV market data: missing columns {missing}")
        if "closed" not in self.data.columns:
            self.data["closed"] = True
        self.data["date"] = pd.to_datetime(self.data["date"], utc=True)
        self.data = self.data.sort_values("date").reset_index(drop=True)
        self.data["closed"] = self.data["closed"].fillna(True).astype(bool)
        closed = self.data[self.data["closed"]].copy()
        if not closed.empty:
            last_closed = closed.iloc[-1]["date"]
            if isinstance(last_closed, pd.Timestamp):
                self.as_of = last_closed.to_pydatetime().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            self.as_of = self.data.iloc[-1]["date"].to_pydatetime().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        self.quality = self.metadata.get("quality", self.quality) or {}

    @property
    def closed_candles(self) -> pd.DataFrame:
        return self.data[self.data["closed"]].copy().reset_index(drop=True)

    def aligned_to(self, as_of: datetime | str | None) -> pd.DataFrame:
        if as_of is None:
            return self.closed_candles
        anchor = pd.Timestamp(as_of).tz_convert("UTC") if hasattr(pd.Timestamp(as_of), "tzinfo") and pd.Timestamp(as_of).tzinfo is not None else pd.Timestamp(as_of, tz="UTC")
        return self.closed_candles[self.closed_candles["date"] <= anchor].copy().reset_index(drop=True)


class DataProviderMarketAdapter:
    """Bridge from Freqtrade DataProvider output to canonical platform market series."""

    def __init__(self, validator: MarketDataValidator | None = None) -> None:
        self._validator = validator or MarketDataValidator()

    def from_dataframe(
        self,
        symbol: str,
        timeframe: str,
        dataframe: pd.DataFrame,
        *,
        metadata: dict[str, object] | None = None,
        as_of: datetime | str | None = None,
    ) -> CanonicalMarketSeries:
        cleaned = self._validator.validate(dataframe, expected_timeframe=timeframe)
        quality = getattr(cleaned, "attrs", {}).get("quality", {})
        series = CanonicalMarketSeries(
            symbol=str(symbol).strip(),
            timeframe=str(timeframe).strip(),
            data=cleaned,
            metadata={**(metadata or {}), "quality": quality},
        )
        if as_of is not None:
            series.as_of = as_of
        return series

    def from_data_provider(self, data_provider: object, pair: str, timeframe: str, candle_type: str = "") -> CanonicalMarketSeries:
        """Fetch a normalized series directly from a Freqtrade DataProvider."""
        getter = getattr(data_provider, "get_pair_dataframe", None)
        if getter is None:
            raise TypeError("data_provider does not expose get_pair_dataframe")
        dataframe = getter(pair=pair, timeframe=timeframe, candle_type=candle_type)
        if dataframe is None:
            raise ValueError(f"no market data returned for {pair} on {timeframe}")
        return self.from_dataframe(pair, timeframe, dataframe, metadata={"source": "freqtrade_data_provider"})
