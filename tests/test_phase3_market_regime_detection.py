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


def _detector_features(frame: pd.DataFrame) -> tuple[dict[str, float], MarketRegimeType]:
    series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", frame)
    features = MarketFeatureExtractor().extract(series)
    result = MarketRegimeDetector().detect(
        [MarketObservation(timeframe="1h", signal="trend", metadata={"features": features, "as_of": series.as_of})]
    )
    return features, result.regime


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


def test_validator_detects_out_of_order_input_before_sorting() -> None:
    bad = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T00:30:00Z"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
            {"date": pd.Timestamp("2024-01-01T00:15:00Z"), "open": 102.0, "high": 103.0, "low": 100.5, "close": 102.8, "volume": 1200.0},
        ]
    )

    with pytest.raises(ValueError, match="strictly increasing|out-of-order|timeline"):
        MarketDataValidator().validate(bad, expected_timeframe="15m")


def test_validator_reports_missing_intervals_and_zero_volume_policy() -> None:
    gaps = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T00:15:00Z"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
            {"date": pd.Timestamp("2024-01-01T00:30:00Z"), "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.8, "volume": 0.0},
            {"date": pd.Timestamp("2024-01-01T01:00:00Z"), "open": 103.0, "high": 104.0, "low": 102.0, "close": 103.5, "volume": 1200.0},
        ]
    )

    cleaned = MarketDataValidator().validate(gaps, expected_timeframe="15m")
    quality = cleaned.attrs["quality"]
    assert quality["missing_candles"] >= 1
    assert quality["has_gaps"] is True
    assert quality["zero_volume"] is True
    assert quality["quality_state"] in {"degraded", "zero_volume", "missing_data"}


def test_stale_data_is_detected_with_as_of() -> None:
    stale = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T01:00:00Z"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
            {"date": pd.Timestamp("2024-01-01T02:00:00Z"), "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1200.0},
        ]
    )
    quality = MarketDataValidator().validate(stale, expected_timeframe="1h", as_of=pd.Timestamp("2024-01-01T20:00:00Z")).attrs["quality"]
    assert quality["stale"] is True

    fresh = MarketDataValidator().validate(stale, expected_timeframe="1h", as_of=pd.Timestamp("2024-01-01T03:00:00Z")).attrs["quality"]
    assert fresh["stale"] is False


def test_bullish_and_bearish_regime_detection_is_directionally_symmetric() -> None:
    detector = MarketRegimeDetector()

    bullish = pd.DataFrame(
        [
            {"date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"), "open": 100.0 + hour, "high": 102.0 + hour, "low": 99.0 + hour, "close": 101.5 + hour, "volume": 1000.0 + hour * 100.0}
            for hour in range(12)
        ]
    )
    bearish = bullish.copy()
    bearish["open"] = 200.0 - bearish["open"]
    bearish["close"] = 200.0 - bearish["close"]
    bearish["high"] = bearish[["open", "close"]].max(axis=1) + 1.0
    bearish["low"] = bearish[["open", "close"]].min(axis=1) - 1.0

    bullish_series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", bullish)
    bearish_series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", bearish)

    bullish_result = detector.detect([
        MarketObservation(timeframe="1h", signal="trend", metadata={"features": MarketFeatureExtractor().extract(bullish_series), "as_of": bullish_series.as_of})
    ])
    bearish_result = detector.detect([
        MarketObservation(timeframe="1h", signal="trend", metadata={"features": MarketFeatureExtractor().extract(bearish_series), "as_of": bearish_series.as_of})
    ])

    assert bullish_result.regime in {MarketRegimeType.STRONG_UPTREND, MarketRegimeType.WEAK_UPTREND}
    assert bearish_result.regime in {MarketRegimeType.STRONG_DOWNTREND, MarketRegimeType.WEAK_DOWNTREND}
    assert bullish_result.regime != bearish_result.regime


def test_signal_does_not_change_regime_outcome() -> None:
    detector = MarketRegimeDetector()
    transition = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.5, "volume": 1500.0},
            {"date": pd.Timestamp("2024-01-01T01:00:00Z"), "open": 101.5, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1700.0},
            {"date": pd.Timestamp("2024-01-01T02:00:00Z"), "open": 102.0, "high": 104.0, "low": 101.0, "close": 103.5, "volume": 1600.0},
            {"date": pd.Timestamp("2024-01-01T03:00:00Z"), "open": 103.5, "high": 105.0, "low": 96.5, "close": 97.5, "volume": 2200.0},
            {"date": pd.Timestamp("2024-01-01T04:00:00Z"), "open": 97.5, "high": 99.0, "low": 94.0, "close": 95.0, "volume": 2300.0},
        ]
    )
    series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", transition)
    features = MarketFeatureExtractor().extract(series)
    signal_a = MarketObservation(timeframe="1h", signal="trend", metadata={"features": features, "as_of": series.as_of})
    signal_b = MarketObservation(timeframe="1h", signal="transition", metadata={"features": features, "as_of": series.as_of})

    result_a = detector.detect([signal_a])
    result_b = detector.detect([signal_b])

    assert result_a.regime == result_b.regime
    assert result_a.confidence == pytest.approx(result_b.confidence)
    assert result_a.evidence == result_b.evidence


def test_quiet_volatile_and_extreme_are_distinct_end_to_end() -> None:
    quiet_rows = [
        {"date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"), "open": 100.0 + 0.08 * hour, "high": 100.5 + 0.08 * hour, "low": 99.5 + 0.08 * hour, "close": 100.1 + 0.08 * hour, "volume": 1100.0 + hour * 40.0}
        for hour in range(12)
    ]
    volatile_rows = [
        {
            "date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"),
            "open": 100.0 + (5.0 if hour % 2 == 0 else -4.5),
            "close": 100.0 + (2.0 if hour % 2 == 0 else -2.0),
            "high": max(100.0 + (5.0 if hour % 2 == 0 else -4.5), 100.0 + (2.0 if hour % 2 == 0 else -2.0)) + 2.5,
            "low": min(100.0 + (5.0 if hour % 2 == 0 else -4.5), 100.0 + (2.0 if hour % 2 == 0 else -2.0)) - 2.5,
            "volume": 1800.0 + hour * 200.0,
        }
        for hour in range(12)
    ]
    extreme_rows = [
        {
            "date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"),
            "open": 100.0 + (20.0 if hour % 2 == 0 else -18.0),
            "close": 100.0 + (12.0 if hour % 2 == 0 else -10.0),
            "high": max(100.0 + (20.0 if hour % 2 == 0 else -18.0), 100.0 + (12.0 if hour % 2 == 0 else -10.0)) + 8.0,
            "low": min(100.0 + (20.0 if hour % 2 == 0 else -18.0), 100.0 + (12.0 if hour % 2 == 0 else -10.0)) - 8.0,
            "volume": 5000.0 + hour * 300.0,
        }
        for hour in range(12)
    ]

    quiet_features, quiet_regime = _detector_features(pd.DataFrame(quiet_rows))
    volatile_features, volatile_regime = _detector_features(pd.DataFrame(volatile_rows))
    extreme_features, extreme_regime = _detector_features(pd.DataFrame(extreme_rows))

    assert quiet_regime == MarketRegimeType.QUIET_RANGE
    assert quiet_features["volatility"] < 3.0
    assert quiet_features["range"] < 15.0
    assert quiet_features["trend_strength"] <= 0.015

    assert volatile_regime == MarketRegimeType.VOLATILE_RANGE
    assert volatile_features["volatility"] > 5.0
    assert volatile_features["volatility"] < 12.0
    assert volatile_features["normalized_atr"] > 4.0
    assert volatile_features["range"] > 14.0

    assert extreme_regime == MarketRegimeType.EXTREME
    assert extreme_features["volatility"] > 12.0
    assert extreme_features["normalized_atr"] > 8.0
    assert extreme_features["range"] >= 60.0


def test_breakout_and_transition_are_reachable_end_to_end() -> None:
    detector = MarketRegimeDetector()

    consolidation = pd.DataFrame(
        [
            {"date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1200.0}
            for hour in range(8)
        ]
    )
    breakout = pd.DataFrame(
        [
            {"date": pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1200.0}
            for hour in range(8)
        ]
    )
    breakout = pd.concat([
        breakout,
        pd.DataFrame([
            {"date": pd.Timestamp("2024-01-01T08:00:00Z"), "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.5, "volume": 2200.0},
        ])
    ]).sort_values("date").reset_index(drop=True)

    transition = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.5, "volume": 1500.0},
            {"date": pd.Timestamp("2024-01-01T01:00:00Z"), "open": 101.5, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1700.0},
            {"date": pd.Timestamp("2024-01-01T02:00:00Z"), "open": 102.0, "high": 104.0, "low": 101.0, "close": 103.5, "volume": 1600.0},
            {"date": pd.Timestamp("2024-01-01T03:00:00Z"), "open": 103.5, "high": 105.0, "low": 96.5, "close": 97.5, "volume": 2200.0},
            {"date": pd.Timestamp("2024-01-01T04:00:00Z"), "open": 97.5, "high": 99.0, "low": 94.0, "close": 95.0, "volume": 2300.0},
        ]
    )

    breakout_series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", breakout)
    transition_series = DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", transition)

    breakout_result = detector.detect([
        MarketObservation(timeframe="1h", signal="breakout", metadata={"features": MarketFeatureExtractor().extract(breakout_series), "as_of": breakout_series.as_of})
    ])
    transition_result = detector.detect([
        MarketObservation(timeframe="1h", signal="transition", metadata={"features": MarketFeatureExtractor().extract(transition_series), "as_of": transition_series.as_of})
    ])

    assert breakout_result.regime == MarketRegimeType.BREAKOUT
    assert transition_result.regime == MarketRegimeType.TRANSITION


def test_multitimeframe_independence_and_common_as_of() -> None:
    detector = MarketRegimeDetector()
    as_of = pd.Timestamp("2024-01-01T12:00:00Z")

    fifteen = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T00:15:00Z"), "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.2, "volume": 1100.0},
            {"date": pd.Timestamp("2024-01-01T00:30:00Z"), "open": 101.2, "high": 102.0, "low": 101.0, "close": 101.9, "volume": 1200.0},
        ]
    )
    hourly = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T10:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T11:00:00Z"), "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.6, "volume": 1300.0},
            {"date": pd.Timestamp("2024-01-01T12:00:00Z"), "open": 101.6, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1500.0},
        ]
    )
    daily = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 90.0, "high": 95.0, "low": 88.0, "close": 94.0, "volume": 5000.0},
            {"date": pd.Timestamp("2024-01-02T00:00:00Z"), "open": 94.0, "high": 99.0, "low": 92.0, "close": 98.0, "volume": 5200.0},
        ]
    )

    obs = [
        MarketObservation(timeframe="15m", signal="trend", metadata={"features": MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "15m", fifteen, as_of=as_of)), "as_of": as_of.isoformat()}),
        MarketObservation(timeframe="1h", signal="trend", metadata={"features": MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", hourly, as_of=as_of)), "as_of": as_of.isoformat()}),
        MarketObservation(timeframe="1d", signal="trend", metadata={"features": MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1d", daily, as_of=as_of)), "as_of": as_of.isoformat()}),
    ]
    result = detector.detect(obs)
    assert result.metadata["timeframe_alignment"] >= 0.0
    assert result.evidence["dominant_signal"] in {"trend", "momentum", "range", "quality"}


def test_result_timestamp_is_market_bound_and_not_wall_clock() -> None:
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"date": pd.Timestamp("2024-01-01T01:00:00Z"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
            {"date": pd.Timestamp("2024-01-01T02:00:00Z"), "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.7, "volume": 1200.0},
        ]
    )
    obs = MarketObservation(timeframe="1h", signal="trend", metadata={"features": MarketFeatureExtractor().extract(DataProviderMarketAdapter().from_dataframe("BTC/USDT", "1h", df)), "as_of": "2024-01-01T02:00:00Z"})
    result = MarketRegimeDetector().detect([obs])
    assert result.timestamp == "2024-01-01T02:00:00Z"


def test_validator_rejects_missing_required_columns() -> None:
    bad = pd.DataFrame([{"date": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}])
    with pytest.raises(ValueError, match="missing columns|invalid OHLCV market data"):
        MarketDataValidator().validate(bad)
