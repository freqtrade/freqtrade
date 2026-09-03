import ast
import importlib
from pathlib import Path

import pytest

import platform
from pathlib import Path

from freqtrade_platform.account.models import RealAccountSnapshot, SimulationAccount, SimulationBootstrap
from freqtrade_platform.account.service import AccountService
from freqtrade_platform.capital.models import CapitalAllocation
from freqtrade_platform.core.exceptions import PlatformValidationError
from freqtrade_platform.core.lifecycle import PlatformLifecycle
from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.profiles.repository import TradingProfileRepository
from freqtrade_platform.storage.database import PlatformDatabase
from freqtrade_platform.storage.models import PlatformProfileRecord
from freqtrade_platform.storage.repositories import PlatformProfileRepository
from freqtrade_platform.strategies.manager import StrategyManager
from freqtrade_platform.strategies.models import StrategyDefinition
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

    strategy = StrategyDefinition(
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
        StrategyDefinition(strategy_id="", name="Bad Strategy", market_type="spot")

    with pytest.raises(PlatformValidationError, match="name"):
        StrategyDefinition(strategy_id="id-1", name="", market_type="spot")

    with pytest.raises(PlatformValidationError, match="market_type"):
        StrategyDefinition(strategy_id="id-2", name="Bad Market", market_type="")


def test_strategy_registry_prevents_duplicate_ids_and_supports_enable_disable():
    registry = StrategyRegistry()
    first = StrategyDefinition(strategy_id="dup-safe", name="One", market_type="spot")
    second = StrategyDefinition(strategy_id="dup-safe", name="Two", market_type="spot")

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
    service = AccountService()

    real_snapshot = service.create_real_snapshot(
        timestamp="2026-01-01T00:00:00Z",
        exchange="binance",
        market_type="futures",
        available_balance=100.0,
        total_balance=120.0,
        equity=115.0,
        positions={"BTC/USDT": 0.25},
        source_metadata={"source": "exchange"},
    )
    assert isinstance(real_snapshot, RealAccountSnapshot)
    assert not hasattr(real_snapshot, "simulated_balance")

    bootstrap = service.create_simulation_bootstrap(
        timestamp="2026-01-01T00:00:00Z",
        exchange="binance",
        market_type="futures",
        starting_balance=1000.0,
        metadata={"source": "bootstrap"},
    )
    assert isinstance(bootstrap, SimulationBootstrap)

    simulated = service.create_simulation_account(
        timestamp="2026-01-01T00:00:00Z",
        exchange="binance",
        market_type="futures",
        starting_balance=1000.0,
        available_balance=75.0,
        total_balance=80.0,
        equity=78.0,
        positions={"BTC/USDT": 0.25},
        metadata={"source": "simulation"},
        bootstrap=bootstrap,
    )

    assert isinstance(simulated, SimulationAccount)
    assert simulated.available_balance == 75.0
    assert simulated.equity != real_snapshot.equity
    assert not hasattr(real_snapshot, "simulated_equity")


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


def test_strategy_model_is_single_canonical_domain_model():
    strategy = StrategyDefinition(
        strategy_id="single-source",
        name="Single Source",
        market_type="spot",
        compatible_regimes=["STRONG_UPTREND"],
    )

    assert strategy.config == {}
    assert strategy.enabled is True


def test_strategy_registry_update_uses_public_contract():
    registry = StrategyRegistry()
    registry.register(
        StrategyDefinition(strategy_id="manager-1", name="Alpha", market_type="spot")
    )

    updated = registry.update("manager-1", name="Beta", enabled=False)
    assert updated.name == "Beta"
    assert updated.enabled is False

    manager = StrategyManager(registry)
    manager.update("manager-1", description="Updated description")
    assert manager.get("manager-1").description == "Updated description"


def test_trading_profile_repository_persists_to_sqlite():
    db = PlatformDatabase("sqlite:///:memory:")
    db.create_all()

    repo = TradingProfileRepository(db)
    profile = TradingProfile(
        profile_id="profile-db-1",
        name="DB Profile",
        exchange="binance",
        market_type="spot",
        symbol_scope=["BTC/USDT"],
        capital_allocation=55.0,
    )

    repo.add(profile)
    assert repo.get("profile-db-1") is not None
    assert len(repo.list()) == 1


def test_platform_lifecycle_rejects_invalid_transitions():
    lifecycle = PlatformLifecycle()
    lifecycle.mark_ready()
    lifecycle.start()

    with pytest.raises(ValueError, match="Transition"):
        lifecycle.start()

    lifecycle.pause()
    lifecycle.resume()
    lifecycle.stop()
    assert lifecycle.state.value == "stopped"


def test_storage_repositories_crud_operations():
    db = PlatformDatabase("sqlite:///:memory:")
    db.create_all()

    with db.session() as session:
        repository = PlatformProfileRepository(session)
        record = PlatformProfileRecord(
            profile_id="db-profile",
            name="SQLite Profile",
            exchange="binance",
            market_type="spot",
            capital_allocation=33.3,
        )
        repository.add(record)
        assert repository.get("db-profile").profile_id == "db-profile"
        assert len(repository.list()) == 1


def test_trading_profile_repository_uses_sqlite_as_authoritative_storage():
    db = PlatformDatabase("sqlite:///:memory:")
    db.create_all()

    repo = TradingProfileRepository(db)
    profile = TradingProfile(
        profile_id="profile-sqlite-authoritative",
        name="Authoritative",
        exchange="binance",
        market_type="spot",
        symbol_scope=["BTC/USDT"],
        capital_allocation=42.0,
    )

    repo.add(profile)
    assert repo._profiles is None
    assert repo.get("profile-sqlite-authoritative") is not None
    assert repo.list()


def test_strategy_registry_update_is_atomic_on_validation_failure():
    registry = StrategyRegistry()
    registry.register(StrategyDefinition(strategy_id="atomic-1", name="Alpha", market_type="spot"))

    with pytest.raises(PlatformValidationError, match="name"):
        registry.update("atomic-1", name="", market_type="")

    assert registry.get("atomic-1").name == "Alpha"
    assert registry.get("atomic-1").market_type == "spot"


def test_platform_lifecycle_stop_moves_directly_to_stopped_state():
    lifecycle = PlatformLifecycle()
    lifecycle.mark_ready()
    lifecycle.start()
    lifecycle.stop()

    assert lifecycle.state == lifecycle.state.__class__.STOPPED


def test_account_models_do_not_expose_legacy_alias():
    account_module = importlib.import_module("freqtrade_platform.account.models")
    assert not hasattr(account_module, "AccountSnapshot")
