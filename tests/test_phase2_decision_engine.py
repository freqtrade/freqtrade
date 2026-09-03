from __future__ import annotations

import pytest

from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.regimes.interface import MarketRegimeDetector
from freqtrade_platform.regimes.models import MarketRegimeResult, MarketRegimeType, MarketObservation
from freqtrade_platform.strategies.models import StrategyDefinition
from freqtrade_platform.trading.universe import TradingUniverse
from freqtrade_platform.compatibility.evaluator import StrategyCompatibilityEvaluator
from freqtrade_platform.selection.selector import StrategySelector


@pytest.fixture
def profile() -> TradingProfile:
    return TradingProfile(
        profile_id="phase2-profile",
        name="Phase2 Profile",
        exchange="binance",
        market_type="spot",
        symbol_scope=["BTC/USDT", "ETH/USDT"],
        assigned_strategies=["trend-core", "range-core", "breakout-core"],
    )


@pytest.fixture
def universe() -> TradingUniverse:
    return TradingUniverse(
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        exclude_symbols=["ADA/USDT"],
        max_symbols=3,
        enabled=True,
    )


def test_detector_contract_and_profile_universe_relationship() -> None:
    profile = TradingProfile(
        profile_id="universe-profile",
        name="Universe Profile",
        exchange="binance",
        market_type="spot",
        universe_id="uv-1",
        symbol_scope=["BTC/USDT"],
    )

    assert profile.universe_id == "uv-1"
    assert profile.symbol_scope == ["BTC/USDT"]

    class FakeDetector:
        def detect(self, observations: list[MarketObservation]) -> MarketRegimeResult:
            return MarketRegimeResult(
                regime=MarketRegimeType.STRONG_UPTREND,
                confidence=0.8,
                timeframe="1h",
                timestamp="2026-01-01T00:00:00Z",
                evidence={"trend_strength": "strong"},
                observations=observations,
            )

    detector: MarketRegimeDetector = FakeDetector()
    result = detector.detect([MarketObservation(timeframe="1D", signal="trend")])
    assert result.regime == MarketRegimeType.STRONG_UPTREND
    assert isinstance(result, MarketRegimeResult)


def test_profile_scope_is_a_narrowing_constraint_on_universe() -> None:
    profile = TradingProfile(
        profile_id="scoped-profile",
        name="Scoped Profile",
        exchange="binance",
        market_type="spot",
        universe_id="uv-1",
        symbol_scope=["btc/usdt", " sol/usdt ", "DOGE/USDT"],
    )

    universe = TradingUniverse(
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        exclude_symbols=["ETH/USDT"],
    )

    assert profile.resolve_symbols(universe.eligible_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT"])) == ["BTC/USDT", "SOL/USDT"]
    assert profile.resolve_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT"], universe_enabled=True) == ["BTC/USDT", "SOL/USDT"]
    assert profile.resolve_symbols(["BTC/USDT", "ETH/USDT"], universe_enabled=False) == []


def test_universe_filters_and_normalizes_symbols(universe: TradingUniverse) -> None:
    universe.add_symbol("btc/usdt")
    universe.exclude_symbol("eth/usdt")

    assert "BTC/USDT" in universe.include_symbols
    assert "ETH/USDT" in universe.exclude_symbols
    assert universe.contains("BTC/USDT") is True
    assert universe.contains("ADA/USDT") is False
    assert universe.eligible_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]) == ["BTC/USDT", "SOL/USDT"]
    assert universe.eligible_symbols([]) == []

    empty_universe = TradingUniverse(exchange="binance", market_type="spot", include_symbols=[])
    assert empty_universe.contains("BTC/USDT") is True
    assert empty_universe.eligible_symbols(["BTC/USDT", "ETH/USDT"]) == ["BTC/USDT", "ETH/USDT"]


def test_universe_rejects_disabled_or_invalid_state() -> None:
    universe = TradingUniverse(
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", ""],
        enabled=False,
    )

    assert universe.enabled is False
    assert universe.contains("BTC/USDT") is False
    assert universe.eligible_symbols(["BTC/USDT", "ETH/USDT"]) == []

    with pytest.raises(ValueError, match="exchange"):
        TradingUniverse(exchange="", market_type="spot", include_symbols=["BTC/USDT"])


def test_market_regime_result_requires_valid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MarketRegimeResult(
            regime=MarketRegimeType.STRONG_UPTREND,
            confidence=1.5,
            timeframe="1h",
            timestamp="2026-01-01T00:00:00Z",
            evidence={"trend_strength": "strong"},
        )


def test_compatibility_and_selection_are_deterministic() -> None:
    trend = StrategyDefinition(
        strategy_id="trend-core",
        name="Trend Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.STRONG_UPTREND, MarketRegimeType.WEAK_UPTREND],
    )
    range_strategy = StrategyDefinition(
        strategy_id="range-core",
        name="Range Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.QUIET_RANGE, MarketRegimeType.TRANSITION],
    )
    breakout = StrategyDefinition(
        strategy_id="breakout-core",
        name="Breakout Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.BREAKOUT, MarketRegimeType.VOLATILE_RANGE],
    )

    regime_result = MarketRegimeResult(
        regime=MarketRegimeType.STRONG_UPTREND,
        confidence=0.82,
        timeframe="1h",
        timestamp="2026-01-01T00:00:00Z",
        evidence={"trend_strength": "strong", "structure": "bullish"},
        observations=[
            MarketObservation(timeframe="1D", signal="trend", metadata={"bias": "bullish"}),
            MarketObservation(timeframe="1H", signal="trend", metadata={"bias": "bullish"}),
        ],
    )

    evaluator = StrategyCompatibilityEvaluator()
    compatible_trend = evaluator.evaluate(trend, regime_result, market_type="spot")
    assert compatible_trend.compatible is True

    incompatible = evaluator.evaluate(range_strategy, regime_result, market_type="spot")
    assert incompatible.compatible is False
    assert "regime" in incompatible.reason.lower()

    selector = StrategySelector()
    selection = selector.select(
        profile=TradingProfile(
            profile_id="selection-profile",
            name="Selection Profile",
            exchange="binance",
            market_type="spot",
            symbol_scope=["BTC/USDT"],
            assigned_strategies=["range-core", "trend-core", "breakout-core"],
        ),
        regime_result=regime_result,
        strategies=[trend, range_strategy, breakout],
    )

    assert selection.selected_strategy_id == "trend-core"
    assert selection.decision == "TRADE"
    assert selection.reason


def test_selector_returns_no_trade_when_nothing_matches() -> None:
    no_trade = StrategyDefinition(
        strategy_id="no-trade-core",
        name="No Trade Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.NO_TRADE],
    )

    regime_result = MarketRegimeResult(
        regime=MarketRegimeType.VOLATILE_RANGE,
        confidence=0.9,
        timeframe="4h",
        timestamp="2026-01-01T00:00:00Z",
        evidence={"volatility": "elevated"},
    )

    selector = StrategySelector()
    selection = selector.select(
        profile=TradingProfile(
            profile_id="empty-selection",
            name="Empty Selection",
            exchange="binance",
            market_type="spot",
            symbol_scope=["BTC/USDT"],
            assigned_strategies=["no-trade-core"],
        ),
        regime_result=regime_result,
        strategies=[no_trade],
    )

    assert selection.selected_strategy_id is None
    assert selection.decision == "NO_TRADE"
