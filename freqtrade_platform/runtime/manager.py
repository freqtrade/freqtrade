"""Central Strategy Runtime Manager governing runtime lifecycle, dynamic process switching, and workspace isolation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError
from freqtrade_platform.profiles.manager import TradingProfileManager
from freqtrade_platform.runtime.adapter import StrategyRuntimeAdapter
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
    PlatformRuntimeRecord,
    PlatformStrategyRecord,
    PlatformStrategySourceRecord,
)
from freqtrade_platform.storage.repositories import (
    PlatformProfileRepository,
    PlatformRuntimeRepository,
    PlatformStrategyRepository,
    PlatformStrategySourceRepository,
    PlatformUniverseRepository,
)
from freqtrade_platform.strategies.manager import StrategyManager
from freqtrade_platform.strategies.models import StrategyDefinition


class StrategyRuntimeManager:
    """Central lifecycle manager for strategy runtimes and dynamic process switching."""

    def __init__(
        self,
        strategy_manager: StrategyManager | None = None,
        profile_manager: TradingProfileManager | None = None,
        profile_repository: PlatformProfileRepository | None = None,
        universe_repository: PlatformUniverseRepository | None = None,
        strategy_source_repository: PlatformStrategySourceRepository | None = None,
        runtime_repository: PlatformRuntimeRepository | None = None,
        strategy_repository: PlatformStrategyRepository | None = None,
        workspace_manager: RuntimeWorkspaceManager | None = None,
        process_manager: RuntimeProcessManager | None = None,
        adapter: StrategyRuntimeAdapter | None = None,
    ) -> None:
        self.strategy_manager = strategy_manager or StrategyManager()
        self.profile_manager = profile_manager or TradingProfileManager()
        self.profile_repository = profile_repository or PlatformProfileRepository()
        self.universe_repository = universe_repository or PlatformUniverseRepository()
        self.strategy_source_repository = (
            strategy_source_repository or PlatformStrategySourceRepository()
        )
        self.runtime_repository = runtime_repository or PlatformRuntimeRepository()
        self.strategy_repository = strategy_repository or PlatformStrategyRepository()
        self.workspace_manager = workspace_manager or RuntimeWorkspaceManager()
        self.process_manager = process_manager or RuntimeProcessManager()
        self.adapter = adapter or StrategyRuntimeAdapter()

        self.static_validator = StaticStrategyValidator()
        self.runtime_validator = RuntimeStrategyValidator()

        self._runtimes: dict[str, StrategyRuntimeInstance] = {}

    def paste_and_register_strategy(
        self,
        strategy_id: str,
        name: str,
        source_code: str,
        market_type: str = "SPOT",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyDefinition:
        """Paste strategy code, statically validate with AST, store source and hash, register in StrategyRegistry."""
        static_res = self.static_validator.validate_source(source_code)
        if not static_res.is_valid:
            raise PlatformValidationError(f"Static AST validation failed: {static_res.error_message}")

        source_hash = calculate_source_hash(source_code)

        now_str = datetime.now(timezone.utc).isoformat()
        source_record = PlatformStrategySourceRecord(
            strategy_id=strategy_id,
            name=name,
            source_code=source_code,
            source_hash=source_hash,
            lifecycle_state="REGISTERED",
            metadata_json=str(metadata) if metadata else None,
            created_at=now_str,
            updated_at=now_str,
        )
        self.strategy_source_repository.add(source_record)

        strat_def = StrategyDefinition(
            strategy_id=strategy_id,
            name=name,
            market_type=market_type,
            description=description,
            enabled=True,
            config=metadata or {},
        )
        self.strategy_manager.add(strat_def)

        strat_record = PlatformStrategyRecord(
            strategy_id=strategy_id,
            name=name,
            market_type=market_type,
            enabled=True,
        )
        self.strategy_repository.add(strat_record)

        return strat_def

    def create_runtime(
        self,
        profile_id: str,
        strategy_id: str,
        mode: RuntimeMode = RuntimeMode.DRY_RUN,
        market_type: MarketType = MarketType.SPOT,
        custom_config: dict[str, Any] | None = None,
    ) -> StrategyRuntimeInstance:
        """Create and materialize a strategy runtime for a trading profile."""
        profile_rec = self.profile_repository.get(profile_id)
        if not profile_rec and hasattr(self.profile_manager, "get_profile"):
            prof_domain = self.profile_manager.get_profile(profile_id)
            if not prof_domain:
                raise PlatformValidationError(f"Unknown profile: {profile_id}")
        elif not profile_rec:
            raise PlatformValidationError(f"Unknown profile: {profile_id}")

        strat_def = self.strategy_manager.get(strategy_id)
        if not strat_def:
            raise PlatformValidationError(f"Unknown strategy: {strategy_id}")

        if not strat_def.enabled:
            raise PlatformValidationError(f"Strategy {strategy_id} is disabled and cannot be run")

        source_rec = self.strategy_source_repository.get(strategy_id)
        if not source_rec:
            raise PlatformValidationError(f"No source code found for strategy {strategy_id}")

        active_rec = self.runtime_repository.get_active_for_profile(profile_id)
        if active_rec and active_rec.state in {"READY", "STARTING", "RUNNING"}:
            raise PlatformValidationError(
                f"Profile {profile_id} already has an active runtime instance ({active_rec.runtime_id})"
            )

        runtime_id = f"rt_{profile_id}_{uuid.uuid4().hex[:8]}"
        now_str = datetime.now(timezone.utc).isoformat()

        instance = StrategyRuntimeInstance(
            runtime_id=runtime_id,
            profile_id=profile_id,
            strategy_id=strategy_id,
            strategy_source_hash=source_rec.source_hash,
            mode=mode,
            market_type=market_type,
            state=RuntimeState.CREATED,
            created_at=now_str,
        )

        instance.transition_to(RuntimeState.VALIDATING)

        symbols = self._resolve_universe_symbols(profile_rec)

        ws_path = self.workspace_manager.prepare_workspace(
            runtime_id=runtime_id,
            strategy_name=strat_def.name,
            source_code=source_rec.source_code,
            mode=mode,
            market_type=market_type,
            symbols=symbols,
            custom_config=custom_config,
        )
        instance.workspace_path = str(ws_path)

        strategy_file = ws_path / "strategies" / f"{strat_def.name}.py"
        rt_val_res = self.runtime_validator.validate_materialized_strategy(
            strategy_path=strategy_file,
            strategy_name=strat_def.name,
            market_type=market_type,
        )

        if not rt_val_res.is_valid:
            instance.transition_to(
                RuntimeState.FAILED,
                error_message=f"Runtime validation failed: {rt_val_res.error_message}",
            )
            self._save_runtime_record(instance)
            raise PlatformValidationError(f"Runtime strategy loading failed: {rt_val_res.error_message}")

        instance.transition_to(RuntimeState.READY)
        self._runtimes[runtime_id] = instance
        self._save_runtime_record(instance)

        return instance

    def start_runtime(
        self,
        runtime_id: str,
        cmd_override: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> StrategyRuntimeInstance:
        """Start isolated process for runtime instance."""
        instance = self.get_runtime(runtime_id)
        if not instance:
            raise PlatformValidationError(f"Unknown runtime: {runtime_id}")

        if instance.state not in {RuntimeState.READY, RuntimeState.STOPPED}:
            raise PlatformValidationError(f"Cannot start runtime in state {instance.state.value}")

        strat_def = self.strategy_manager.get(instance.strategy_id)
        if not strat_def or not strat_def.enabled:
            raise PlatformValidationError(f"Strategy {instance.strategy_id} is disabled and cannot be started")

        instance.transition_to(RuntimeState.STARTING)

        ws_path = Path(instance.workspace_path)
        cmd = cmd_override or self.adapter.build_command(
            instance=instance,
            workspace_path=ws_path,
            strategy_name=strat_def.name,
            extra_args=extra_args,
        )

        log_path = ws_path / "logs" / "process.log"

        try:
            handle = self.process_manager.start_process(
                runtime_id=runtime_id,
                cmd=cmd,
                cwd=ws_path,
                stdout_path=log_path,
            )
            instance.process_id = handle.pid
            instance.started_at = datetime.now(timezone.utc).isoformat()

            # Perform startup confirmation window check
            if not handle.confirm_startup(check_window_secs=0.3):
                exit_code = handle.poll()
                error_msg = f"Process exited immediately upon startup with exit code {exit_code}"
                instance.transition_to(RuntimeState.FAILED, error_message=error_msg)
                self._save_runtime_record(instance)
                raise RuntimeError(error_msg)

            instance.transition_to(RuntimeState.RUNNING)
        except Exception as e:
            if instance.state != RuntimeState.FAILED:
                instance.transition_to(RuntimeState.FAILED, error_message=str(e))
            self._save_runtime_record(instance)
            raise RuntimeError(f"Failed to start runtime process: {e}") from e

        self._save_runtime_record(instance)
        return instance

    def stop_runtime(self, runtime_id: str, timeout: float = 5.0) -> StrategyRuntimeInstance:
        """Stop running process for runtime instance."""
        instance = self.get_runtime(runtime_id)
        if not instance:
            raise PlatformValidationError(f"Unknown runtime: {runtime_id}")

        if instance.state not in {RuntimeState.RUNNING, RuntimeState.STARTING}:
            if instance.state == RuntimeState.STOPPED:
                return instance
            raise PlatformValidationError(f"Cannot stop runtime in state {instance.state.value}")

        instance.transition_to(RuntimeState.STOPPING)
        self.process_manager.stop_process(runtime_id, timeout=timeout)

        instance.stopped_at = datetime.now(timezone.utc).isoformat()
        instance.transition_to(RuntimeState.STOPPED)
        self._save_runtime_record(instance)
        return instance

    def switch_strategy(
        self,
        profile_id: str,
        replacement_strategy_id: str,
        mode: RuntimeMode = RuntimeMode.DRY_RUN,
        market_type: MarketType = MarketType.SPOT,
        custom_config: dict[str, Any] | None = None,
        start_replacement: bool = True,
        replacement_cmd_override: list[str] | None = None,
    ) -> StrategyRuntimeInstance:
        """Process-based dynamic strategy switching with safe preservation policy."""
        current_active = self.get_active_runtime_for_profile(profile_id)

        try:
            replacement_runtime = self._create_switch_replacement_runtime(
                profile_id=profile_id,
                strategy_id=replacement_strategy_id,
                mode=mode,
                market_type=market_type,
                custom_config=custom_config,
            )

            if start_replacement:
                self.start_runtime(
                    replacement_runtime.runtime_id,
                    cmd_override=replacement_cmd_override,
                )

        except Exception as e:
            if current_active and current_active.state == RuntimeState.RUNNING:
                pass
            raise PlatformValidationError(
                f"Strategy switch failed. Current runtime preserved. Error: {str(e)}"
            ) from e

        if current_active and current_active.runtime_id != replacement_runtime.runtime_id:
            if current_active.state in {RuntimeState.RUNNING, RuntimeState.STARTING}:
                self.stop_runtime(current_active.runtime_id)

        return replacement_runtime

    def _create_switch_replacement_runtime(
        self,
        profile_id: str,
        strategy_id: str,
        mode: RuntimeMode,
        market_type: MarketType,
        custom_config: dict[str, Any] | None,
    ) -> StrategyRuntimeInstance:
        profile_rec = self.profile_repository.get(profile_id)
        if not profile_rec and hasattr(self.profile_manager, "get_profile"):
            prof_domain = self.profile_manager.get_profile(profile_id)
            if not prof_domain:
                raise PlatformValidationError(f"Unknown profile: {profile_id}")
        elif not profile_rec:
            raise PlatformValidationError(f"Unknown profile: {profile_id}")

        strat_def = self.strategy_manager.get(strategy_id)
        if not strat_def:
            raise PlatformValidationError(f"Unknown strategy: {strategy_id}")

        if not strat_def.enabled:
            raise PlatformValidationError(f"Strategy {strategy_id} is disabled and cannot be run")

        source_rec = self.strategy_source_repository.get(strategy_id)
        if not source_rec:
            raise PlatformValidationError(f"No source code found for strategy {strategy_id}")

        runtime_id = f"rt_{profile_id}_{uuid.uuid4().hex[:8]}"
        now_str = datetime.now(timezone.utc).isoformat()

        instance = StrategyRuntimeInstance(
            runtime_id=runtime_id,
            profile_id=profile_id,
            strategy_id=strategy_id,
            strategy_source_hash=source_rec.source_hash,
            mode=mode,
            market_type=market_type,
            state=RuntimeState.CREATED,
            created_at=now_str,
        )

        instance.transition_to(RuntimeState.VALIDATING)

        symbols = self._resolve_universe_symbols(profile_rec)

        ws_path = self.workspace_manager.prepare_workspace(
            runtime_id=runtime_id,
            strategy_name=strat_def.name,
            source_code=source_rec.source_code,
            mode=mode,
            market_type=market_type,
            symbols=symbols,
            custom_config=custom_config,
        )
        instance.workspace_path = str(ws_path)

        strategy_file = ws_path / "strategies" / f"{strat_def.name}.py"
        rt_val_res = self.runtime_validator.validate_materialized_strategy(
            strategy_path=strategy_file,
            strategy_name=strat_def.name,
            market_type=market_type,
        )

        if not rt_val_res.is_valid:
            instance.transition_to(
                RuntimeState.FAILED,
                error_message=f"Runtime validation failed: {rt_val_res.error_message}",
            )
            self._save_runtime_record(instance)
            raise PlatformValidationError(f"Runtime strategy loading failed: {rt_val_res.error_message}")

        instance.transition_to(RuntimeState.READY)
        self._runtimes[runtime_id] = instance
        self._save_runtime_record(instance)

        return instance

    def monitor_and_detect_crashes(self) -> list[str]:
        """Poll running runtime processes. Detect crashed/unexpectedly terminated processes."""
        crashed = []
        for runtime_id, instance in list(self._runtimes.items()):
            if instance.state == RuntimeState.RUNNING:
                if not self.process_manager.is_running(runtime_id):
                    exit_code = self.process_manager.poll(runtime_id)
                    instance.transition_to(
                        RuntimeState.FAILED,
                        error_message=f"Process exited unexpectedly with exit code {exit_code}",
                    )
                    instance.stopped_at = datetime.now(timezone.utc).isoformat()
                    self._save_runtime_record(instance)
                    crashed.append(runtime_id)
        return crashed

    def get_runtime(self, runtime_id: str) -> StrategyRuntimeInstance | None:
        if runtime_id in self._runtimes:
            return self._runtimes[runtime_id]
        rec = self.runtime_repository.get(runtime_id)
        if rec:
            inst = self._record_to_instance(rec)
            self._runtimes[runtime_id] = inst
            return inst
        return None

    def get_active_runtime_for_profile(self, profile_id: str) -> StrategyRuntimeInstance | None:
        for inst in self._runtimes.values():
            if inst.profile_id == profile_id and inst.state in {
                RuntimeState.READY,
                RuntimeState.STARTING,
                RuntimeState.RUNNING,
            }:
                return inst

        rec = self.runtime_repository.get_active_for_profile(profile_id)
        if rec:
            inst = self._record_to_instance(rec)
            self._runtimes[rec.runtime_id] = inst
            return inst
        return None

    def _resolve_universe_symbols(self, profile_rec: Any) -> list[str]:
        if not profile_rec or not profile_rec.universe_id:
            return ["BTC/USDT"]

        univ_rec = self.universe_repository.get(profile_rec.universe_id)
        if not univ_rec:
            raise PlatformValidationError(f"Unknown universe: {profile_rec.universe_id}")

        if hasattr(univ_rec, "enabled") and not univ_rec.enabled:
            raise PlatformValidationError(f"Universe {profile_rec.universe_id} is disabled")

        if univ_rec.include_symbols:
            inc = [s.strip().upper() for s in univ_rec.include_symbols.split(",") if s.strip()]
        else:
            inc = []

        if univ_rec.exclude_symbols:
            exc = {s.strip().upper() for s in univ_rec.exclude_symbols.split(",") if s.strip()}
        else:
            exc = set()

        eligible = [s for s in inc if s not in exc]
        if hasattr(univ_rec, "max_symbols") and univ_rec.max_symbols is not None:
            eligible = eligible[: univ_rec.max_symbols]

        return eligible if eligible else ["BTC/USDT"]

    def _save_runtime_record(self, instance: StrategyRuntimeInstance) -> None:
        record = PlatformRuntimeRecord(
            runtime_id=instance.runtime_id,
            profile_id=instance.profile_id,
            strategy_id=instance.strategy_id,
            strategy_source_hash=instance.strategy_source_hash,
            mode=instance.mode.value if isinstance(instance.mode, RuntimeMode) else str(instance.mode),
            market_type=instance.market_type.value if isinstance(instance.market_type, MarketType) else str(instance.market_type),
            state=instance.state.value if isinstance(instance.state, RuntimeState) else str(instance.state),
            workspace_path=instance.workspace_path,
            process_id=instance.process_id,
            created_at=instance.created_at,
            started_at=instance.started_at,
            stopped_at=instance.stopped_at,
            last_error=instance.last_error,
        )
        self.runtime_repository.add(record)

    def _record_to_instance(self, rec: PlatformRuntimeRecord) -> StrategyRuntimeInstance:
        return StrategyRuntimeInstance(
            runtime_id=rec.runtime_id,
            profile_id=rec.profile_id,
            strategy_id=rec.strategy_id,
            strategy_source_hash=rec.strategy_source_hash,
            mode=RuntimeMode(rec.mode),
            market_type=MarketType(rec.market_type),
            state=RuntimeState(rec.state),
            workspace_path=rec.workspace_path,
            process_id=rec.process_id,
            created_at=rec.created_at,
            started_at=rec.started_at,
            stopped_at=rec.stopped_at,
            last_error=rec.last_error,
        )
