"""Workspace manager for isolated Strategy Runtime workspaces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from freqtrade_platform.runtime.models import MarketType, RuntimeMode


class RuntimeWorkspaceManager:
    """Manages creation, materialization, and cleanup of isolated runtime workspaces.

    Each runtime instance receives its own folder structure:
        <base_dir>/<runtime_id>/
            config/
                config.json
            strategies/
                <strategy_name>.py
            logs/
            state/
    """

    def __init__(self, base_workspace_dir: str | Path = "user_data/runtimes") -> None:
        self.base_workspace_dir = Path(base_workspace_dir).resolve()

    def get_workspace_path(self, runtime_id: str) -> Path:
        return self.base_workspace_dir / runtime_id

    def prepare_workspace(
        self,
        runtime_id: str,
        strategy_name: str,
        source_code: str,
        mode: RuntimeMode = RuntimeMode.DRY_RUN,
        market_type: MarketType = MarketType.SPOT,
        symbols: list[str] | None = None,
        custom_config: dict[str, Any] | None = None,
    ) -> Path:
        pair_whitelist = list(symbols) if symbols else ["BTC/USDT"]
        workspace_dir = self.get_workspace_path(runtime_id)
        config_dir = workspace_dir / "config"
        strategies_dir = workspace_dir / "strategies"
        logs_dir = workspace_dir / "logs"
        state_dir = workspace_dir / "state"

        for directory in (config_dir, strategies_dir, logs_dir, state_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # Materialize strategy source file
        strategy_file = strategies_dir / f"{strategy_name}.py"
        strategy_file.write_text(source_code, encoding="utf-8")

        # Materialize Freqtrade configuration JSON file
        config_data: dict[str, Any] = {
            "max_open_trades": 3,
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "tradable_balance_ratio": 0.99,
            "fiat_display_currency": "USD",
            "dry_run": mode != RuntimeMode.LIVE,
            "dry_run_wallet": 1000,
            "cancel_open_orders_on_exit": False,
            "trading_mode": "futures" if market_type == MarketType.FUTURES else "spot",
            "margin_mode": "isolated" if market_type == MarketType.FUTURES else None,
            "unfilledtimeout": {
                "entry": 10,
                "exit": 10,
                "unit": "minutes",
            },
            "entry_pricing": {
                "price_side": "same",
                "use_order_book": True,
                "order_book_top": 1,
                "price_last_balance": 0.0,
            },
            "exit_pricing": {
                "price_side": "same",
                "use_order_book": True,
                "order_book_top": 1,
            },
            "exchange": {
                "name": "binance",
                "key": "",
                "secret": "",
                "pair_whitelist": pair_whitelist,
                "pair_blacklist": [],
            },
            "pairlists": [{"method": "StaticPairList"}],
            "bot_name": f"freqtrade_{runtime_id}",
            "initial_state": "running",
            "force_entry_enable": False,
            "internals": {"process_throttle_secs": 5},
            "strategy": strategy_name,
            "strategy_path": str(strategies_dir),
            "user_data_dir": str(workspace_dir),
            "db_url": f"sqlite:///{state_dir}/tradesv3.sqlite",
            "logfile": str(logs_dir / "freqtrade.log"),
        }

        if custom_config:
            config_data.update(custom_config)

        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        return workspace_dir

    def cleanup_workspace(self, runtime_id: str) -> None:
        workspace_dir = self.get_workspace_path(runtime_id)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
