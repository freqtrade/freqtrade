"""Strategy Runtime Adapter bridging Platform Runtime Instance with Freqtrade CLI process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from freqtrade_platform.runtime.models import MarketType, RuntimeMode, StrategyRuntimeInstance


class StrategyRuntimeAdapter:
    """Adapts platform runtime specifications into Freqtrade executable command invocations."""

    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def build_command(
        self,
        instance: StrategyRuntimeInstance,
        workspace_path: Path | str,
        strategy_name: str,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        ws_path = Path(workspace_path).resolve()
        config_path = ws_path / "config" / "config.json"
        strategies_dir = ws_path / "strategies"

        if instance.mode == RuntimeMode.BACKTEST:
            cmd = [
                self.python_executable,
                "-m",
                "freqtrade",
                "backtesting",
                "--config",
                str(config_path),
                "--strategy",
                strategy_name,
                "--strategy-path",
                str(strategies_dir),
            ]
        else:
            cmd = [
                self.python_executable,
                "-m",
                "freqtrade",
                "trade",
                "--config",
                str(config_path),
                "--strategy",
                strategy_name,
                "--strategy-path",
                str(strategies_dir),
            ]

            if instance.mode == RuntimeMode.DRY_RUN:
                cmd.append("--dry-run")

        if instance.market_type == MarketType.FUTURES:
            cmd.extend(["--trading-mode", "futures"])
        elif instance.market_type == MarketType.SPOT:
            cmd.extend(["--trading-mode", "spot"])

        if extra_args:
            cmd.extend(extra_args)

        return cmd
