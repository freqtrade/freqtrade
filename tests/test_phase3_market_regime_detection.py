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
        metadata={"features": features, "symbol": "BTC/USDT"},
    )

    result = detector.detect([observation])
    assert result.regime in {MarketRegimeType.STRONG_UPTREND, MarketRegimeType.WEAK_UPTREND}
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence["dominant_signal"] in {"trend", "momentum"}
