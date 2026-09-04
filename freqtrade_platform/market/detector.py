"""Deterministic market regime detector built on canonical market observations."""

from __future__ import annotations

from datetime import datetime, timezone

from freqtrade_platform.market.features import MarketFeatureExtractor
from freqtrade_platform.regimes.models import MarketObservation, MarketRegimeResult, MarketRegimeType


class MarketRegimeDetector:
    """Simple deterministic detector for trend, range, and breakout market regimes."""

    def __init__(self, feature_extractor: MarketFeatureExtractor | None = None) -> None:
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()

    def detect(self, observations: list[MarketObservation] | tuple[MarketObservation, ...]) -> MarketRegimeResult:
        if not observations:
            raise ValueError("at least one observation is required")

        latest = observations[-1]
        features = latest.metadata.get("features", {}) if isinstance(latest.metadata, dict) else {}
        if not features:
            features = self._feature_extractor.extract(latest.metadata.get("series")) if "series" in latest.metadata else {}

        trend = float(features.get("trend_strength", 0.0))
        volatility = float(features.get("volatility", 0.0))
        breakout = float(features.get("breakout_bias", 0.0))
        momentum = float(features.get("momentum", 0.0))

        if trend > 0.03 and breakout > 0.0:
            regime = MarketRegimeType.STRONG_UPTREND
            dominant_signal = "trend"
            confidence = min(0.99, 0.6 + abs(trend) * 10.0 + max(0.0, momentum) / 100.0)
        elif trend < -0.03 and breakout < 0.0:
            regime = MarketRegimeType.STRONG_DOWNTREND
            dominant_signal = "trend"
            confidence = min(0.99, 0.6 + abs(trend) * 10.0 + max(0.0, -momentum) / 100.0)
        elif abs(trend) > 0.008:
            regime = MarketRegimeType.WEAK_UPTREND if trend > 0 else MarketRegimeType.WEAK_DOWNTREND
            dominant_signal = "momentum"
            confidence = min(0.9, 0.52 + min(0.3, abs(trend) * 10.0))
        elif volatility > 5.0:
            regime = MarketRegimeType.VOLATILE_RANGE
            dominant_signal = "range"
            confidence = min(0.94, 0.55 + min(0.25, volatility / 40.0))
        else:
            regime = MarketRegimeType.QUIET_RANGE
            dominant_signal = "range"
            confidence = min(0.88, 0.5 + min(0.2, max(0.0, abs(trend)) * 10.0))

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return MarketRegimeResult(
            regime=regime,
            confidence=max(0.0, min(1.0, confidence)),
            timeframe=latest.timeframe,
            timestamp=timestamp,
            evidence={
                "trend_strength": trend,
                "volatility": volatility,
                "breakout_bias": breakout,
                "dominant_signal": dominant_signal,
            },
            observations=list(observations),
            metadata={"source": "platform_market_regime_detector"},
        )
