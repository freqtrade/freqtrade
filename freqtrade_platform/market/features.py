"""Deterministic feature extraction for regime detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from freqtrade_platform.market.data import CanonicalMarketSeries


class MarketFeatureExtractor:
    """Convert a canonical market series into stable, deterministic feature metrics."""

    def extract(self, series: CanonicalMarketSeries) -> dict[str, float]:
        if not isinstance(series, CanonicalMarketSeries):
            raise TypeError("series must be a CanonicalMarketSeries")

        data = series.data.copy()
        close = pd.to_numeric(data["close"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        returns = close.pct_change().fillna(0.0)

        if close.empty:
            return {"trend_strength": 0.0, "volatility": 0.0, "range": 0.0, "breakout_bias": 0.0, "momentum": 0.0}

        start_close = float(close.iloc[0]) if len(close) > 0 else 0.0
        end_close = float(close.iloc[-1]) if len(close) > 0 else 0.0
        trend_strength = float((end_close - start_close) / start_close) if start_close else 0.0
        volatility = float(returns.std(ddof=0) * 100.0)
        range_width = float((data["high"].max() - data["low"].min()) / max(abs(end_close), 1e-9))

        rolling_mid = close.rolling(window=max(2, min(5, len(close))), min_periods=1).median()
        breakout_bias = float((end_close - float(rolling_mid.iloc[-1])) / max(abs(float(rolling_mid.iloc[-1])), 1e-9))
        momentum = float(returns.iloc[-1] * 100.0)
        volume_slope = float(volume.pct_change().fillna(0.0).mean() * 100.0)

        return {
            "trend_strength": trend_strength,
            "volatility": volatility,
            "range": range_width,
            "breakout_bias": breakout_bias,
            "momentum": momentum,
            "volume_slope": volume_slope,
        }

    def summarize(self, series: CanonicalMarketSeries) -> dict[str, Any]:
        features = self.extract(series)
        return {
            "symbol": series.symbol,
            "timeframe": series.timeframe,
            "features": features,
        }
