"""Canonical market-series adapter for Freqtrade OHLCV inputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from freqtrade_platform.market.validator import MarketDataValidator


@dataclass(slots=True)
class CanonicalMarketSeries:
    """Normalized market-data view converted from Freqtrade OHLCV output."""

    symbol: str
    timeframe: str
    data: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

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
    ) -> CanonicalMarketSeries:
        cleaned = self._validator.validate(dataframe)
        return CanonicalMarketSeries(
            symbol=str(symbol).strip(),
            timeframe=str(timeframe).strip(),
            data=cleaned,
            metadata=metadata or {},
        )

    def from_data_provider(self, data_provider: object, pair: str, timeframe: str, candle_type: str = "") -> CanonicalMarketSeries:
        """Fetch a normalized series directly from a Freqtrade DataProvider."""
        getter = getattr(data_provider, "get_pair_dataframe", None)
        if getter is None:
            raise TypeError("data_provider does not expose get_pair_dataframe")
        dataframe = getter(pair=pair, timeframe=timeframe, candle_type=candle_type)
        if dataframe is None:
            raise ValueError(f"no market data returned for {pair} on {timeframe}")
        return self.from_dataframe(pair, timeframe, dataframe, metadata={"source": "freqtrade_data_provider"})
