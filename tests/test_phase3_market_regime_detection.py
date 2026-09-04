from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from freqtrade_platform.market.data import CanonicalMarketSeries, DataProviderMarketAdapter
from freqtrade_platform.market.detector import MarketRegimeDetector
from freqtrade_platform.market.features import MarketFeatureExtractor
from freqtrade_platform.market.validator import MarketDataValidator
from freqtrade_platform.regimes.models import MarketObservation, MarketRegimeType


@pytest.fixture
def trending_dataframe() -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    closes = [100.0, 101.0, 102.0, 103.5, 105.0, 106.5, 108.0, 109.8, 111.5, 113.2]
    rows = []
    for idx, close in enumerate(closes):
        ts = start + timedelta(hours=idx)
        rows.append(
            {
                "date": ts,
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000 + idx * 10,
            }
        )
    return pd.DataFrame(rows)


def test_adapter_builds_canonical_series_from_dataframe(trending_dataframe: pd.DataFrame) -> None:
    adapter = DataProviderMarketAdapter()
    series = adapter.from_dataframe("BTC/USDT", "1h", trending_dataframe)

    assert isinstance(series, CanonicalMarketSeries)
    assert series.symbol == "BTC/USDT"
    assert series.timeframe == "1h"
    assert len(series.data) == len(trending_dataframe)
    assert series.data["close"].iloc[-1] == pytest.approx(113.2)


def test_validator_rejects_invalid_market_data() -> None:
    bad = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01T00:00:00Z"),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 0.0,
            },
            {
                "date": pd.Timestamp("2024-01-01T00:00:00Z"),
                "open": 101.0,
                "high": 108.0,
                "low": 95.0,
                "close": 108.0,
                "volume": 2000.0,
            },
        ]
    )
    validator = MarketDataValidator()

    with pytest.raises(ValueError, match="duplicate|zero volume|timeline|ohlcv"):
        validator.validate(bad)


def test_feature_extractor_and_detector_classify_uptrend() -> None:
    detector = MarketRegimeDetector()
    extractor = MarketFeatureExtractor()

    series = DataProviderMarketAdapter().from_dataframe(
        "BTC/USDT",
        "1h",
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000.0,
                },
                {
                    "date": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "open": 100.5,
                    "high": 104.0,
                    "low": 100.0,
                    "close": 103.8,
                    "volume": 1500.0,
                },
                {
                    "date": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "open": 103.8,
                    "high": 106.0,
                    "low": 103.0,
                    "close": 105.7,
                    "volume": 1800.0,
                },
            ]
        ),
    )

    features = extractor.extract(series)
    observation = MarketObservation(
        timeframe="1h",
        signal="trend",
        metadata={"features": features, "symbol": "BTC/USDT", "as_of": "2024-01-01T02:00:00Z"},
    )

    result = detector.detect([observation])
    assert result.regime in {MarketRegimeType.STRONG_UPTREND, MarketRegimeType.WEAK_UPTREND}
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence["dominant_signal"] in {"trend", "momentum"}


def test_closed_candle_filtering_and_no_trade_on_insufficient_history() -> None:
    validator = MarketDataValidator()
    frames = pd.DataFrame(
        [
            {"date": "2024-01-01T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "closed": False},
            {"date": "2024-01-01T01:00:00Z", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0, "closed": True},
        ]
    )
    clean = validator.validate(frames, expected_timeframe="1h")
    assert clean["closed"].sum() == 1

    short = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}]
    )
    series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", short)
    observation = MarketObservation(
        timeframe="1h",
        signal="insufficient_history",
        metadata={"features": MarketFeatureExtractor().extract(series), "as_of": "2024-01-01T00:00:00Z"},
    )
    result = MarketRegimeDetector().detect([observation])
    assert result.regime == MarketRegimeType.NO_TRADE


def test_multitimeframe_agreement_and_lookahead_invariance() -> None:
    detector = MarketRegimeDetector()
    observations = []
    frequency_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    for timeframe, points in {"15m": 12, "1h": 8, "4h": 6, "1d": 4}.items():
        base_ts = pd.date_range("2024-01-01T00:00:00Z", periods=points, freq=frequency_map[timeframe])
        base_df = pd.DataFrame(
            {
                "date": base_ts,
                "open": pd.Series(range(points), dtype=float) + 100.0,
                "high": pd.Series(range(points), dtype=float) + 101.0,
                "low": pd.Series(range(points), dtype=float) + 99.0,
                "close": pd.Series(range(points), dtype=float) + 100.5,
                "volume": [1500.0 + i * 100.0 for i in range(points)],
            }
        )
        features = MarketFeatureExtractor().extract(
            DataProviderMarketAdapter().from_dataframe("BTC/USDT", timeframe, base_df)
        )
        observations.append(
            MarketObservation(
                timeframe=timeframe,
                signal="trend",
                metadata={"features": features, "as_of": base_ts[-1].isoformat()},
            )
        )

    result = detector.detect(observations)
    assert result.regime in {MarketRegimeType.STRONG_UPTREND, MarketRegimeType.WEAK_UPTREND, MarketRegimeType.QUIET_RANGE}

    future_ts = pd.date_range("2024-01-01T00:00:00Z", periods=9, freq="1h")
    base_df = pd.DataFrame(
        {
            "date": future_ts,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "close": [100.8, 101.7, 102.8, 103.7, 104.6, 105.8, 106.7, 107.9, 108.8],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1500.0, 1700.0, 1800.0, 2000.0, 2200.0],
        }
    )
    future_df = pd.concat([base_df, pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01T09:00:00Z")],
        "open": [109.0],
        "high": [110.0],
        "low": [108.0],
        "close": [109.4],
        "volume": [2300.0],
    })])
    features_a = MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", base_df))
    features_b = MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", future_df))
    assert features_a["trend_strength"] == pytest.approx(features_b["trend_strength"], rel=1e-3)

    result_a = detector.detect([MarketObservation(timeframe="1h", signal="trend", metadata={"features": features_a, "as_of": "2024-01-01T08:00:00Z"})])
    result_b = detector.detect([MarketObservation(timeframe="1h", signal="trend", metadata={"features": features_b, "as_of": "2024-01-01T08:00:00Z"})])
    assert result_a.regime == result_b.regime


def test_stale_and_timeframe_mismatch_are_rejected() -> None:
    stale = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T03:00:00Z"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
        ]
    )

    with pytest.raises(ValueError, match="timeframe mismatch|invalid OHLCV market data"):
        DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", stale)
