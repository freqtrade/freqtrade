from __future__ import annotations

import pytest

from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.profiles.repository import TradingProfileRepository
from freqtrade_platform.regimes.interface import MarketRegimeDetector
from freqtrade_platform.regimes.models import MarketRegimeResult, MarketRegimeType, MarketObservation
from freqtrade_platform.storage.database import PlatformDatabase
from freqtrade_platform.strategies.models import StrategyDefinition
from freqtrade_platform.trading.repository import TradingUniverseRepository
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
        universe_id="uv-fixture",
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
        universe_id="uv-profile-scope",
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

    empty_universe = TradingUniverse(universe_id="uv-empty", exchange="binance", market_type="spot", include_symbols=[])
    assert empty_universe.contains("BTC/USDT") is True
    assert empty_universe.eligible_symbols(["BTC/USDT", "ETH/USDT"]) == ["BTC/USDT", "ETH/USDT"]


def test_universe_rejects_disabled_or_invalid_state() -> None:
    universe = TradingUniverse(
        universe_id="uv-disabled",
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", ""],
        enabled=False,
    )

    assert universe.enabled is False
    assert universe.contains("BTC/USDT") is False
    assert universe.eligible_symbols(["BTC/USDT", "ETH/USDT"]) == []

    with pytest.raises(ValueError, match="exchange"):
        TradingUniverse(universe_id="uv-invalid", exchange="", market_type="spot", include_symbols=["BTC/USDT"])


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


def test_trading_profile_repository_roundtrip_preserves_phase2_state() -> None:
    db = PlatformDatabase("sqlite:///:memory:")
    db.create_all()

    repo = TradingProfileRepository(db)
    profile = TradingProfile(
        profile_id="profile-roundtrip",
        name="Roundtrip Profile",
        exchange="binance",
        market_type="spot",
        universe_id="uv-1",
        symbol_scope=["BTC/USDT", "ETH/USDT"],
        primary_timeframe="1h",
        informative_timeframes=["4h", "1d"],
        assigned_strategies=["trend-core", "range-core"],
        regime_policy="trend-following",
        risk_configuration={"max_drawdown": 0.15},
        execution_configuration={"mode": "paper"},
        capital_allocation=55.0,
    )

    repo.add(profile)
    loaded = repo.get("profile-roundtrip")
    assert loaded is not None
    assert loaded.profile_id == profile.profile_id
    assert loaded.primary_timeframe == "1h"
    assert loaded.informative_timeframes == ["4h", "1d"]
    assert loaded.assigned_strategies == ["trend-core", "range-core"]
    assert loaded.regime_policy == "trend-following"
    assert loaded.risk_configuration == {"max_drawdown": 0.15}
    assert loaded.execution_configuration == {"mode": "paper"}
    assert loaded.capital_allocation == 55.0


def test_universe_identity_and_repository_roundtrip() -> None:
    db = PlatformDatabase("sqlite:///:memory:")
    db.create_all()

    repo = TradingUniverseRepository(db)
    universe = TradingUniverse(
        universe_id="uv-1",
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", "ETH/USDT"],
        exclude_symbols=["ETH/USDT"],
        max_symbols=1,
        enabled=True,
        metadata={"source": "synthetic"},
    )

    repo.add(universe)
    loaded = repo.get("uv-1")
    assert loaded is not None
    assert loaded.universe_id == "uv-1"
    assert loaded.exchange == "binance"
    assert loaded.include_symbols == ["BTC/USDT"]
    assert loaded.metadata == {"source": "synthetic"}
    assert len(repo.list()) == 1

    second = TradingUniverse(universe_id="uv-2", exchange="kraken", market_type="spot", include_symbols=["SOL/USDT"])
    repo.add(second)
    assert {item.universe_id for item in repo.list()} == {"uv-1", "uv-2"}

    with pytest.raises(ValueError, match="universe_id"):
        TradingUniverse(universe_id="", exchange="binance", market_type="spot")


def test_sqlite_database_migrates_legacy_platform_tables() -> None:
    db_path = "sqlite:///./.tmp_legacy_platform.db"
    legacy = PlatformDatabase(db_path)
    with legacy.engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE platform_profiles (id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL, exchange TEXT NOT NULL, market_type TEXT NOT NULL, universe_id TEXT, symbol_scope TEXT, capital_allocation REAL)")
    legacy.create_all()

    with legacy.engine.begin() as connection:
        columns = {column[1] for column in connection.exec_driver_sql("PRAGMA table_info(platform_profiles)").fetchall()}
        assert "primary_timeframe" in columns
        assert "assigned_strategies" in columns
        assert "risk_configuration" in columns
        assert "execution_configuration" in columns


def test_profile_must_reference_existing_universe_and_scope_can_only_narrow() -> None:
    universe = TradingUniverse(
        universe_id="uv-profile",
        exchange="binance",
        market_type="spot",
        include_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        exclude_symbols=["ETH/USDT"],
        enabled=True,
    )

    profile = TradingProfile(
        profile_id="profile-refs-universe",
        name="Universe Profile",
        exchange="binance",
        market_type="spot",
        universe_id="uv-profile",
        symbol_scope=["BTC/USDT", "DOGE/USDT"],
    )

    assert profile.resolve_symbols(universe.eligible_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT"])) == ["BTC/USDT"]
    assert profile.resolve_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT"], universe_enabled=False) == []

    repo = TradingUniverseRepository()
    assert repo.get("missing-universe") is None


def test_selector_only_uses_assigned_strategies_and_rejects_unknown_or_empty() -> None:
    trend = StrategyDefinition(
        strategy_id="trend-core",
        name="Trend Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.STRONG_UPTREND],
    )
    range_strategy = StrategyDefinition(
        strategy_id="range-core",
        name="Range Core",
        market_type="spot",
        compatible_regimes=[MarketRegimeType.QUIET_RANGE],
    )

    regime_result = MarketRegimeResult(
        regime=MarketRegimeType.STRONG_UPTREND,
        confidence=0.9,
        timeframe="1h",
        timestamp="2026-01-01T00:00:00Z",
        evidence={"trend": "strong"},
    )

    selector = StrategySelector()
    profile = TradingProfile(
        profile_id="assigned-only",
        name="Assigned Only",
        exchange="binance",
        market_type="spot",
        assigned_strategies=["trend-core", "range-core", "missing-core"],
    )

    selection = selector.select(profile=profile, regime_result=regime_result, strategies=[trend, range_strategy])
    assert selection.selected_strategy_id == "trend-core"
    assert selection.decision == "TRADE"
    assert "missing-core" in " ".join(selection.rejected_candidates)

    empty_profile = TradingProfile(
        profile_id="empty-assigned",
        name="Empty Assigned",
        exchange="binance",
        market_type="spot",
        assigned_strategies=[],
    )
    empty_selection = selector.select(profile=empty_profile, regime_result=regime_result, strategies=[trend])
    assert empty_selection.decision == "NO_TRADE"
    assert empty_selection.selected_strategy_id is None

    disabled = StrategyDefinition(
        strategy_id="disabled-core",
        name="Disabled Core",
        market_type="spot",
        enabled=False,
        compatible_regimes=[MarketRegimeType.STRONG_UPTREND],
    )
    disabled_profile = TradingProfile(
        profile_id="disabled-assigned",
        name="Disabled Assigned",
        exchange="binance",
        market_type="spot",
        assigned_strategies=["disabled-core"],
    )
    disabled_selection = selector.select(profile=disabled_profile, regime_result=regime_result, strategies=[disabled])
    assert disabled_selection.decision == "NO_TRADE"
    assert disabled_selection.selected_strategy_id is None
