"""Tests for Phase 4 — Strategy Runtime & Dynamic Strategy Management."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from freqtrade_platform.core.exceptions import PlatformValidationError
from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.runtime.adapter import StrategyRuntimeAdapter
from freqtrade_platform.runtime.manager import StrategyRuntimeManager
from freqtrade_platform.runtime.models import (
    MarketType,
    RuntimeMode,
    RuntimeState,
    StrategyRuntimeInstance,
    calculate_source_hash,
)
from freqtrade_platform.runtime.process import RuntimeProcessManager
from freqtrade_platform.runtime.validator import RuntimeStrategyValidator, StaticStrategyValidator
from freqtrade_platform.runtime.workspace import RuntimeWorkspaceManager
from freqtrade_platform.storage.models import (
    PlatformProfileRecord,
    PlatformRuntimeRecord,
    PlatformStrategySourceRecord,
    PlatformUniverseRecord,
)
from freqtrade_platform.storage.repositories import (
    PlatformProfileRepository,
    PlatformRuntimeRepository,
    PlatformStrategyRepository,
    PlatformStrategySourceRepository,
    PlatformUniverseRepository,
)
from freqtrade_platform.strategies.manager import StrategyManager


SAMPLE_VALID_STRATEGY = """
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class SampleStrategy(IStrategy):
    minimal_roi = {"0": 0.1}
    stoploss = -0.05
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
"""

SAMPLE_INVALID_SYNTAX_STRATEGY = """
class InvalidStrategy(IStrategy:
    minimal_roi = {"0": 0.1}
"""

SAMPLE_MISSING_CLASS_STRATEGY = """
# Just a python script with no strategy class
def foo():
    return 42
"""

SAMPLE_SECOND_STRATEGY = """
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class ReplacementStrategy(IStrategy):
    minimal_roi = {"0": 0.05}
    stoploss = -0.10
    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
"""


@pytest.fixture
def temp_workspace_dir(tmp_path):
    return tmp_path / "runtimes"


@pytest.fixture
def setup_manager(temp_workspace_dir):
    prof_repo = PlatformProfileRepository()
    univ_repo = PlatformUniverseRepository()
    source_repo = PlatformStrategySourceRepository()
    rt_repo = PlatformRuntimeRepository()
    strat_repo = PlatformStrategyRepository()

    prof_repo.add(
        PlatformProfileRecord(
            profile_id="prof_btc",
            name="BTC Scalp Profile",
            exchange="binance",
            market_type="SPOT",
            universe_id="univ_btc",
        )
    )

    univ_repo.add(
        PlatformUniverseRecord(
            universe_id="univ_btc",
            exchange="binance",
            market_type="SPOT",
            include_symbols="BTC/USDT,ETH/USDT,SOL/USDT",
            exclude_symbols="SOL/USDT",
            enabled=True,
        )
    )

    ws_mgr = RuntimeWorkspaceManager(base_workspace_dir=temp_workspace_dir)
    proc_mgr = RuntimeProcessManager()
    adapter = StrategyRuntimeAdapter()

    mgr = StrategyRuntimeManager(
        profile_repository=prof_repo,
        universe_repository=univ_repo,
        strategy_source_repository=source_repo,
        runtime_repository=rt_repo,
        strategy_repository=strat_repo,
        workspace_manager=ws_mgr,
        process_manager=proc_mgr,
        adapter=adapter,
    )
    return mgr


# --- 1. Persistence & Reload ---
def test_strategy_source_persistence(setup_manager):
    mgr = setup_manager
    strat_def = mgr.paste_and_register_strategy(
        strategy_id="strat_1",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )
    assert strat_def.strategy_id == "strat_1"

    source_rec = mgr.strategy_source_repository.get("strat_1")
    assert source_rec is not None
    assert source_rec.name == "SampleStrategy"
    assert source_rec.source_code == SAMPLE_VALID_STRATEGY
    assert len(source_rec.source_hash) == 64


def test_strategy_persistence_reload_across_manager_recreation(temp_workspace_dir):
    source_repo = PlatformStrategySourceRepository()
    strat_repo = PlatformStrategyRepository()

    # Manager 1 registers a strategy
    mgr1 = StrategyRuntimeManager(
        strategy_source_repository=source_repo,
        strategy_repository=strat_repo,
        workspace_manager=RuntimeWorkspaceManager(base_workspace_dir=temp_workspace_dir),
    )
    mgr1.paste_and_register_strategy(
        strategy_id="strat_reload",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    # Manager 2 recreated with same repositories
    new_strat_mgr = StrategyManager(
        source_repository=source_repo,
        strategy_repository=strat_repo,
    )
    mgr2 = StrategyRuntimeManager(
        strategy_manager=new_strat_mgr,
        strategy_source_repository=source_repo,
        strategy_repository=strat_repo,
        workspace_manager=RuntimeWorkspaceManager(base_workspace_dir=temp_workspace_dir),
    )

    reloaded_strat = mgr2.strategy_manager.get("strat_reload")
    assert reloaded_strat is not None
    assert reloaded_strat.name == "SampleStrategy"


# --- 2. Deterministic Hash ---
def test_deterministic_source_hash():
    hash1 = calculate_source_hash(SAMPLE_VALID_STRATEGY)
    hash2 = calculate_source_hash(SAMPLE_VALID_STRATEGY)
    hash_diff = calculate_source_hash(SAMPLE_SECOND_STRATEGY)

    assert hash1 == hash2
    assert hash1 != hash_diff


# --- 3. Static AST Validation ---
def test_static_ast_validation():
    validator = StaticStrategyValidator()

    # Valid strategy
    res_valid = validator.validate_source(SAMPLE_VALID_STRATEGY)
    assert res_valid.is_valid is True
    assert res_valid.strategy_name == "SampleStrategy"

    # Syntax error
    res_syntax = validator.validate_source(SAMPLE_INVALID_SYNTAX_STRATEGY)
    assert res_syntax.is_valid is False
    assert "Invalid Python syntax" in res_syntax.error_message

    # Missing strategy class
    res_missing = validator.validate_source(SAMPLE_MISSING_CLASS_STRATEGY)
    assert res_missing.is_valid is False
    assert "No strategy class" in res_missing.error_message


# --- 4. Real Freqtrade StrategyResolver Loading ---
def test_real_strategy_resolver_loading(temp_workspace_dir):
    ws_mgr = RuntimeWorkspaceManager(base_workspace_dir=temp_workspace_dir)
    ws_path = ws_mgr.prepare_workspace(
        runtime_id="rt_test_load",
        strategy_name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
        mode=RuntimeMode.DRY_RUN,
        market_type=MarketType.SPOT,
    )

    validator = RuntimeStrategyValidator()
    strat_file = ws_path / "strategies" / "SampleStrategy.py"

    res = validator.validate_materialized_strategy(
        strategy_path=strat_file,
        strategy_name="SampleStrategy",
        market_type=MarketType.SPOT,
    )
    assert res.is_valid is True
    assert res.strategy_name == "SampleStrategy"


# --- 5. Workspace Isolation ---
def test_workspace_isolation(temp_workspace_dir):
    ws_mgr = RuntimeWorkspaceManager(base_workspace_dir=temp_workspace_dir)

    ws1 = ws_mgr.prepare_workspace(
        runtime_id="rt_1",
        strategy_name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    ws2 = ws_mgr.prepare_workspace(
        runtime_id="rt_2",
        strategy_name="ReplacementStrategy",
        source_code=SAMPLE_SECOND_STRATEGY,
    )

    assert ws1 != ws2
    assert (ws1 / "strategies" / "SampleStrategy.py").exists()
    assert not (ws1 / "strategies" / "ReplacementStrategy.py").exists()
    assert (ws2 / "strategies" / "ReplacementStrategy.py").exists()


# --- 6. Universe Symbol Resolution ---
def test_profile_universe_symbol_resolution(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_univ",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    rt = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_univ",
    )

    ws_config_file = Path(rt.workspace_path) / "config" / "config.json"
    assert ws_config_file.exists()

    config_data = json.loads(ws_config_file.read_text(encoding="utf-8"))
    whitelist = config_data["exchange"]["pair_whitelist"]

    # Profile -> Universe -> include BTC/USDT, ETH/USDT; exclude SOL/USDT
    assert "BTC/USDT" in whitelist
    assert "ETH/USDT" in whitelist
    assert "SOL/USDT" not in whitelist


# --- 7. Complete Mode & Market Type CLI/Config Boundaries ---
def test_complete_mode_and_market_type_boundaries():
    adapter = StrategyRuntimeAdapter(python_executable="python")
    ws_path = Path("/tmp/dummy_workspace")

    # DRY_RUN SPOT
    inst_dry_spot = StrategyRuntimeInstance(
        runtime_id="rt_dry_spot",
        profile_id="p1",
        strategy_id="s1",
        strategy_source_hash="h1",
        mode=RuntimeMode.DRY_RUN,
        market_type=MarketType.SPOT,
    )
    cmd_dry_spot = adapter.build_command(inst_dry_spot, ws_path, "SampleStrategy")
    assert "trade" in cmd_dry_spot
    assert "--dry-run" in cmd_dry_spot
    assert "--trading-mode" in cmd_dry_spot and cmd_dry_spot[cmd_dry_spot.index("--trading-mode") + 1] == "spot"

    # LIVE FUTURES
    inst_live_fut = StrategyRuntimeInstance(
        runtime_id="rt_live_fut",
        profile_id="p1",
        strategy_id="s1",
        strategy_source_hash="h1",
        mode=RuntimeMode.LIVE,
        market_type=MarketType.FUTURES,
    )
    cmd_live_fut = adapter.build_command(inst_live_fut, ws_path, "SampleStrategy")
    assert "trade" in cmd_live_fut
    assert "--dry-run" not in cmd_live_fut
    assert "--trading-mode" in cmd_live_fut and cmd_live_fut[cmd_live_fut.index("--trading-mode") + 1] == "futures"

    # BACKTEST SPOT
    inst_backtest = StrategyRuntimeInstance(
        runtime_id="rt_backtest",
        profile_id="p1",
        strategy_id="s1",
        strategy_source_hash="h1",
        mode=RuntimeMode.BACKTEST,
        market_type=MarketType.SPOT,
    )
    cmd_backtest = adapter.build_command(inst_backtest, ws_path, "SampleStrategy")
    assert "backtesting" in cmd_backtest


# --- 8. Runtime Lifecycle Transitions ---
def test_runtime_state_transitions():
    inst = StrategyRuntimeInstance(
        runtime_id="rt_state_1",
        profile_id="prof_1",
        strategy_id="strat_1",
        strategy_source_hash="dummy_hash",
        state=RuntimeState.CREATED,
    )

    inst.transition_to(RuntimeState.VALIDATING)
    assert inst.state == RuntimeState.VALIDATING

    inst.transition_to(RuntimeState.READY)
    assert inst.state == RuntimeState.READY

    inst.transition_to(RuntimeState.STARTING)
    assert inst.state == RuntimeState.STARTING

    inst.transition_to(RuntimeState.RUNNING)
    assert inst.state == RuntimeState.RUNNING

    inst.transition_to(RuntimeState.STOPPING)
    assert inst.state == RuntimeState.STOPPING

    inst.transition_to(RuntimeState.STOPPED)
    assert inst.state == RuntimeState.STOPPED

    with pytest.raises(PlatformValidationError):
        inst.transition_to(RuntimeState.RUNNING)


# --- 9. Real Subprocess Lifecycle & Startup Confirmation ---
def test_real_process_lifecycle(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_proc",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    rt = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_proc",
        mode=RuntimeMode.DRY_RUN,
        market_type=MarketType.SPOT,
    )
    assert rt.state == RuntimeState.READY

    dummy_cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    started_rt = mgr.start_runtime(rt.runtime_id, cmd_override=dummy_cmd)
    assert started_rt.state == RuntimeState.RUNNING
    assert started_rt.process_id is not None
    assert mgr.process_manager.is_running(rt.runtime_id) is True

    stopped_rt = mgr.stop_runtime(rt.runtime_id, timeout=2.0)
    assert stopped_rt.state == RuntimeState.STOPPED
    assert mgr.process_manager.is_running(rt.runtime_id) is False


def test_real_process_startup_failure_detection(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_fail_proc",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    rt = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_fail_proc",
    )

    # Process that exits immediately with failure code
    failing_cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]

    with pytest.raises(RuntimeError, match="exited immediately upon startup"):
        mgr.start_runtime(rt.runtime_id, cmd_override=failing_cmd)

    assert rt.state == RuntimeState.FAILED


# --- 10. Duplicate Active Runtime Prevention ---
def test_duplicate_active_runtime_rejection(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_dup",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    rt1 = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_dup",
    )
    assert rt1.state == RuntimeState.READY

    with pytest.raises(PlatformValidationError, match="already has an active runtime instance"):
        mgr.create_runtime(
            profile_id="prof_btc",
            strategy_id="strat_dup",
        )


# --- 11. Crash Detection ---
def test_crash_detection(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_crash",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    rt = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_crash",
    )

    crash_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    # Bypass initial startup window check by using a delayed crash script
    delayed_crash_cmd = [sys.executable, "-c", "import time, sys; time.sleep(0.4); sys.exit(1)"]
    mgr.start_runtime(rt.runtime_id, cmd_override=delayed_crash_cmd)

    time.sleep(0.6)
    crashed = mgr.monitor_and_detect_crashes()

    assert rt.runtime_id in crashed
    assert rt.state == RuntimeState.FAILED
    assert "Process exited unexpectedly" in rt.last_error


# --- 12. Disabled Strategy Blocking ---
def test_disabled_strategy_cannot_start(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_dis",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    mgr.strategy_manager.deactivate("strat_dis")

    with pytest.raises(PlatformValidationError, match="disabled"):
        mgr.create_runtime(
            profile_id="prof_btc",
            strategy_id="strat_dis",
        )


# --- 13. Profile Mapping & Unknown Identifiers ---
def test_unknown_profile_or_strategy_rejected(setup_manager):
    mgr = setup_manager
    mgr.paste_and_register_strategy(
        strategy_id="strat_valid",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )

    with pytest.raises(PlatformValidationError, match="Unknown profile"):
        mgr.create_runtime(profile_id="prof_unknown", strategy_id="strat_valid")

    with pytest.raises(PlatformValidationError, match="Unknown strategy"):
        mgr.create_runtime(profile_id="prof_btc", strategy_id="strat_unknown")


# --- 14. Safe Dynamic Strategy Switching & Rollback ---
def test_safe_dynamic_strategy_switching(setup_manager):
    mgr = setup_manager

    mgr.paste_and_register_strategy(
        strategy_id="strat_alpha",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )
    mgr.paste_and_register_strategy(
        strategy_id="strat_beta",
        name="ReplacementStrategy",
        source_code=SAMPLE_SECOND_STRATEGY,
    )

    rt_alpha = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_alpha",
    )

    dummy_cmd_alpha = [sys.executable, "-c", "import time; time.sleep(10)"]
    mgr.start_runtime(rt_alpha.runtime_id, cmd_override=dummy_cmd_alpha)
    assert rt_alpha.state == RuntimeState.RUNNING

    dummy_cmd_beta = [sys.executable, "-c", "import time; time.sleep(10)"]

    rt_beta = mgr.switch_strategy(
        profile_id="prof_btc",
        replacement_strategy_id="strat_beta",
        start_replacement=True,
        replacement_cmd_override=dummy_cmd_beta,
    )

    assert rt_beta.strategy_id == "strat_beta"
    assert rt_beta.state == RuntimeState.RUNNING
    assert rt_alpha.state == RuntimeState.STOPPED

    mgr.stop_runtime(rt_beta.runtime_id)


def test_failed_replacement_process_startup_preserves_current_runtime(setup_manager):
    mgr = setup_manager

    mgr.paste_and_register_strategy(
        strategy_id="strat_alpha",
        name="SampleStrategy",
        source_code=SAMPLE_VALID_STRATEGY,
    )
    mgr.paste_and_register_strategy(
        strategy_id="strat_beta_fail",
        name="ReplacementStrategy",
        source_code=SAMPLE_SECOND_STRATEGY,
    )

    rt_alpha = mgr.create_runtime(
        profile_id="prof_btc",
        strategy_id="strat_alpha",
    )

    dummy_cmd_alpha = [sys.executable, "-c", "import time; time.sleep(10)"]
    mgr.start_runtime(rt_alpha.runtime_id, cmd_override=dummy_cmd_alpha)
    assert rt_alpha.state == RuntimeState.RUNNING

    failing_cmd_beta = [sys.executable, "-c", "import sys; sys.exit(3)"]

    # Replacement strategy Beta exists and is valid, but fails during startup
    with pytest.raises(PlatformValidationError, match="Strategy switch failed"):
        mgr.switch_strategy(
            profile_id="prof_btc",
            replacement_strategy_id="strat_beta_fail",
            start_replacement=True,
            replacement_cmd_override=failing_cmd_beta,
        )

    # Confirm rt_alpha remains RUNNING and alive
    assert rt_alpha.state == RuntimeState.RUNNING
    assert mgr.process_manager.is_running(rt_alpha.runtime_id) is True

    mgr.stop_runtime(rt_alpha.runtime_id)
