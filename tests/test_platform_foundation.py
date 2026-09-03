import ast
from pathlib import Path

import pytest

import platform
from pathlib import Path

from freqtrade_platform.account.models import AccountSnapshot
from freqtrade_platform.capital.models import CapitalAllocation
from freqtrade_platform.core.exceptions import PlatformValidationError
from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.strategies.models import StrategyMetadata
from freqtrade_platform.strategies.registry import StrategyRegistry


def test_platform_package_imports_successfully():
    assert platform is not None
    assert platform.__file__
    assert "freqtrade_platform" not in str(platform.__file__).lower()


def test_core_domain_models_instantiate_correctly():
    profile = TradingProfile(
        profile_id="profile-1",
        name="BTC Core",
        exchange="binance",
        market_type="spot",
        symbol_scope=["BTC/USDT"],
    )
    assert profile.profile_id == "profile-1"
    assert profile.market_type == "spot"

    strategy = StrategyMetadata(
        strategy_id="sma-fast",
        name="SMA Fast",
        market_type="spot",
        compatible_regimes=["STRONG_UPTREND"],
    )
    assert strategy.strategy_id == "sma-fast"
    assert strategy.enabled is True


def test_trading_profile_validation_works():
    with pytest.raises(PlatformValidationError, match="profile_id"):
        TradingProfile(profile_id="", name="Bad Profile", exchange="binance", market_type="spot")

    with pytest.raises(PlatformValidationError, match="exchange"):
        TradingProfile(profile_id="profile-2", name="Bad Exchange", exchange="", market_type="spot")

    with pytest.raises(PlatformValidationError, match="market_type"):
        TradingProfile(profile_id="profile-3", name="Bad Market", exchange="binance", market_type="")


def test_strategy_metadata_validation_works():
    with pytest.raises(PlatformValidationError, match="strategy_id"):
        StrategyMetadata(strategy_id="", name="Bad Strategy", market_type="spot")

    with pytest.raises(PlatformValidationError, match="name"):
        StrategyMetadata(strategy_id="id-1", name="", market_type="spot")

    with pytest.raises(PlatformValidationError, match="market_type"):
        StrategyMetadata(strategy_id="id-2", name="Bad Market", market_type="")


def test_strategy_registry_prevents_duplicate_ids_and_supports_enable_disable():
    registry = StrategyRegistry()
    first = StrategyMetadata(strategy_id="dup-safe", name="One", market_type="spot")
    second = StrategyMetadata(strategy_id="dup-safe", name="Two", market_type="spot")

    registry.register(first)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(second)

    registry.disable("dup-safe")
    assert registry.get("dup-safe").enabled is False

    registry.enable("dup-safe")
    assert registry.get("dup-safe").enabled is True

    assert registry.list()


def test_capital_allocation_validates_percentages():
    allocation = CapitalAllocation(profile_id="btc", allocation_percent=40.0)
    assert allocation.allocation_percent == 40.0

    with pytest.raises(PlatformValidationError, match="allocation_percent"):
        CapitalAllocation(profile_id="bad", allocation_percent=-1)

    with pytest.raises(PlatformValidationError, match="allocation_percent"):
        CapitalAllocation(profile_id="bad", allocation_percent=101)


def test_account_snapshot_keeps_real_and_simulated_values_distinct():
    snapshot = AccountSnapshot(
        timestamp="2026-01-01T00:00:00Z",
        exchange="binance",
        market_type="futures",
        available_balance=100.0,
        total_balance=120.0,
        equity=115.0,
        positions={"BTC/USDT": 0.25},
        raw_source_metadata={"source": "exchange"},
        simulated_balance=75.0,
        simulated_equity=80.0,
    )

    assert snapshot.available_balance == 100.0
    assert snapshot.simulated_balance == 75.0
    assert snapshot.equity != snapshot.simulated_equity


def test_platform_modules_do_not_use_disallowed_infrastructure():
    platform_root = Path(__file__).resolve().parents[1] / "freqtrade_platform"
    disallowed = {"telegram", "supabase", "render", "flask", "ccxt"}

    for py_file in platform_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        lower = text.lower()
        for token in disallowed:
            assert token not in lower, f"{py_file} contains disallowed token: {token}"

        parsed = ast.parse(text, filename=str(py_file))
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Exchange":
                    raise AssertionError(f"{py_file} instantiates Exchange directly")


def test_platform_modules_do_not_import_second_exchange_client():
    platform_root = Path(__file__).resolve().parents[1] / "freqtrade_platform"
    for py_file in platform_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "ccxt" not in text.lower()
        assert "binance(" not in text.lower()


def test_frequent_existing_core_imports_remain_usable():
    from freqtrade.freqtradebot import FreqtradeBot
    from freqtrade.resolvers import StrategyResolver

    assert FreqtradeBot is not None
    assert StrategyResolver is not None
