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

        data = series.aligned_to(series.as_of) if series.as_of is not None else series.closed_candles.copy()
        if len(data) < 3:
            return {
                "trend_strength": 0.0,
                "volatility": 0.0,
                "range": 0.0,
                "breakout_bias": 0.0,
                "momentum": 0.0,
                "directional_strength": 0.0,
                "range_compression": 0.0,
                "structure": 0.0,
            }

        close = pd.to_numeric(data["close"], errors="coerce")
        high = pd.to_numeric(data["high"], errors="coerce")
        low = pd.to_numeric(data["low"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        returns = close.pct_change().fillna(0.0)
        log_returns = close.pct_change().fillna(0.0)

        start_close = float(close.iloc[0])
        end_close = float(close.iloc[-1])
        anchor_window = close.iloc[: min(5, len(close))]
        trend_strength = float((anchor_window.iloc[-1] - anchor_window.iloc[0]) / max(abs(anchor_window.iloc[0]), 1e-9)) if len(anchor_window) > 1 else 0.0
        directional_strength = float(returns.iloc[: min(5, len(returns))].mean())

        recent_window = close.tail(min(10, len(close)))
        ema_fast = recent_window.ewm(span=max(2, len(recent_window) // 2), adjust=False).mean()
        ema_slow = recent_window.ewm(span=max(3, len(recent_window)), adjust=False).mean()
        ema_gap = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(abs(ema_slow.iloc[-1]), 1e-9))

        atr = (high - low).rolling(window=min(14, len(high)), min_periods=1).mean().iloc[-1]
        volatility = float((high - low).mean() / max(abs(end_close), 1e-9) * 100.0)
        normalized_atr = float(atr / max(abs(end_close), 1e-9) * 100.0)

        recent_range = float((high.max() - low.min()) / max(abs(end_close), 1e-9))
        range_width = recent_range * 100.0
        range_compression = float(1.0 / (1.0 + max(recent_range, 1e-9)))

        rolling_mid = close.rolling(window=max(2, min(5, len(close))), min_periods=1).median()
        breakout_bias = float((end_close - float(rolling_mid.iloc[-1])) / max(abs(float(rolling_mid.iloc[-1])), 1e-9))
        momentum = float(returns.iloc[-1] * 100.0)
        volume_slope = float(volume.pct_change().fillna(0.0).mean() * 100.0)

        # Structural signal based on recent swing highs/lows
        recent_high = high.tail(min(5, len(high))).max()
        recent_low = low.tail(min(5, len(low))).min()
        structure = float((recent_high - recent_low) / max(abs(end_close), 1e-9))

        return {
            "trend_strength": trend_strength,
            "directional_strength": directional_strength,
            "ema_gap": ema_gap,
            "volatility": volatility,
            "normalized_atr": normalized_atr,
            "range": range_width,
            "range_compression": range_compression,
            "breakout_bias": breakout_bias,
            "momentum": momentum,
            "volume_slope": volume_slope,
            "structure": structure,
            "log_return_mean": float(log_returns.mean() * 100.0),
            "history_length": float(len(data)),
        }

    def summarize(self, series: CanonicalMarketSeries) -> dict[str, Any]:
        features = self.extract(series)
        return {
            "symbol": series.symbol,
            "timeframe": series.timeframe,
            "features": features,
        }
