"""Deterministic market regime detector built on canonical market observations."""

from __future__ import annotations

from datetime import datetime, timezone

from freqtrade_platform.market.features import MarketFeatureExtractor
from freqtrade_platform.regimes.models import MarketObservation, MarketRegimeResult, MarketRegimeType


class MarketRegimeDetector:
    """Deterministic detector that reasons over multi-timeframe closed-candle observations."""

    def __init__(self, feature_extractor: MarketFeatureExtractor | None = None) -> None:
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()

    def _timeframe_strength(self, observations: list[MarketObservation]) -> dict[str, float]:
        values: dict[str, float] = {}
        for observation in observations:
            if not isinstance(observation.metadata, dict):
                continue
            features = observation.metadata.get("features", {})
            if not isinstance(features, dict):
                continue
            values[observation.timeframe] = float(features.get("trend_strength", 0.0))
        return values

    def _coalesce_features(self, observations: list[MarketObservation]) -> dict[str, float]:
        features: dict[str, float] = {}
        for observation in observations:
            if not isinstance(observation.metadata, dict):
                continue
            obs_features = observation.metadata.get("features", {})
            if not isinstance(obs_features, dict):
                continue
            for key, value in obs_features.items():
                features[key] = float(value)
        return features

    def detect(self, observations: list[MarketObservation] | tuple[MarketObservation, ...]) -> MarketRegimeResult:
        if not observations:
            raise ValueError("at least one observation is required")

        ordered = list(observations)
        latest = ordered[-1]
        combined = self._coalesce_features(ordered)
        timeframe_strength = self._timeframe_strength(ordered)

        if not combined:
            raise ValueError("observations must include feature data for regime detection")

        history_length = float(combined.get("history_length", 0.0))
        if history_length < 3.0:
            regime = MarketRegimeType.NO_TRADE
            dominant_signal = "quality"
            confidence = 0.8
            timestamp = str(latest.metadata.get("as_of")) if isinstance(latest.metadata, dict) and "as_of" in latest.metadata else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return MarketRegimeResult(
                regime=regime,
                confidence=confidence,
                timeframe=latest.timeframe,
                timestamp=timestamp,
                evidence={"dominant_signal": dominant_signal, "history_length": history_length, "quality": "insufficient_history"},
                observations=list(ordered),
                metadata={"source": "platform_market_regime_detector", "quality": "insufficient_history"},
            )

        trend = float(combined.get("trend_strength", 0.0))
        directional_strength = float(combined.get("directional_strength", 0.0))
        volatility = float(combined.get("volatility", 0.0))
        atr = float(combined.get("normalized_atr", 0.0))
        breakout = float(combined.get("breakout_bias", 0.0))
        momentum = float(combined.get("momentum", 0.0))
        range_width = float(combined.get("range", 0.0))
        structure = float(combined.get("structure", 0.0))

        agreement = sum(1 for value in timeframe_strength.values() if value > 0.0) + sum(1 for value in timeframe_strength.values() if value < 0.0)
        alignment_score = max(0.0, min(1.0, agreement / max(len(timeframe_strength), 1)))
        quality_score = 0.7 + min(0.3, max(0.0, 1.0 - (range_width / 100.0)))

        if abs(trend) > 0.08 and breakout > 0.02 and directional_strength > 0.0:
            regime = MarketRegimeType.STRONG_UPTREND if trend > 0 else MarketRegimeType.STRONG_DOWNTREND
            dominant_signal = "trend"
            confidence = max(0.5, min(0.99, 0.55 + abs(trend) * 4.5 + alignment_score * 0.25 + quality_score * 0.15))
        elif abs(trend) >= 0.003 and abs(trend) <= 0.08:
            regime = MarketRegimeType.WEAK_UPTREND if trend > 0 else MarketRegimeType.WEAK_DOWNTREND
            dominant_signal = "momentum"
            confidence = max(0.45, min(0.9, 0.5 + abs(trend) * 4.0 + alignment_score * 0.15))
        elif atr > 4.0 and abs(breakout) < 0.02 and volatility > 5.0:
            regime = MarketRegimeType.VOLATILE_RANGE
            dominant_signal = "range"
            confidence = max(0.45, min(0.95, 0.5 + min(0.3, volatility / 25.0) + alignment_score * 0.2))
        elif abs(breakout) > 0.04 and abs(momentum) > 1.5 and structure > 0.0:
            regime = MarketRegimeType.BREAKOUT
            dominant_signal = "breakout"
            confidence = max(0.5, min(0.98, 0.52 + abs(breakout) * 2.5 + abs(momentum) / 50.0))
        elif alignment_score < 0.5 and abs(trend) < 0.05 and volatility > 2.0:
            regime = MarketRegimeType.TRANSITION
            dominant_signal = "structure"
            confidence = max(0.4, min(0.85, 0.45 + (1.0 - alignment_score) * 0.35 + min(0.2, volatility / 40.0)))
        elif atr > 8.0 or volatility > 12.0:
            regime = MarketRegimeType.EXTREME
            dominant_signal = "volatility"
            confidence = max(0.5, min(0.98, 0.52 + min(0.3, volatility / 25.0)))
        elif abs(trend) <= 0.015 and volatility < 3.0 and range_width < 15.0:
            regime = MarketRegimeType.QUIET_RANGE
            dominant_signal = "range"
            confidence = max(0.4, min(0.9, 0.48 + (1.0 - min(1.0, abs(trend) * 25.0)) * 0.25))
        else:
            regime = MarketRegimeType.NO_TRADE
            dominant_signal = "quality"
            confidence = max(0.2, min(0.7, 0.38 + (1.0 - alignment_score) * 0.25))

        latest_timestamp = latest.metadata.get("as_of") if isinstance(latest.metadata, dict) else None
        timestamp = str(latest_timestamp) if latest_timestamp else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return MarketRegimeResult(
            regime=regime,
            confidence=max(0.0, min(1.0, confidence)),
            timeframe=latest.timeframe,
            timestamp=timestamp,
            evidence={
                "trend_strength": trend,
                "directional_strength": directional_strength,
                "volatility": volatility,
                "atr": atr,
                "breakout_bias": breakout,
                "momentum": momentum,
                "range": range_width,
                "structure": structure,
                "dominant_signal": dominant_signal,
                "timeframe_alignment": alignment_score,
                "quality_score": quality_score,
            },
            observations=list(ordered),
            metadata={"source": "platform_market_regime_detector", "timeframe_alignment": alignment_score},
        )
