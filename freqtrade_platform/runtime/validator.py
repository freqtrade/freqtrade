"""Validation components for Strategy Runtime: static AST and real StrategyResolver."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from freqtrade.enums import TradingMode
from freqtrade.resolvers import StrategyResolver
from freqtrade_platform.core.exceptions import PlatformValidationError
from freqtrade_platform.runtime.models import MarketType


class ValidationResult:
    """Result of static or runtime validation."""

    def __init__(self, is_valid: bool, strategy_name: str | None = None, error_message: str | None = None) -> None:
        self.is_valid = is_valid
        self.strategy_name = strategy_name
        self.error_message = error_message


class StaticStrategyValidator:
    """Validates strategy source code statically using Python AST without executing code."""

    def validate_source(self, source_code: str) -> ValidationResult:
        if not source_code or not source_code.strip():
            return ValidationResult(is_valid=False, error_message="Empty strategy source code")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid Python syntax: line {e.lineno}, {e.msg}",
            )

        # Search for strategy classes (classes that inherit from IStrategy or have Strategy in name/bases)
        found_classes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)

                if "IStrategy" in base_names or any("Strategy" in b for b in base_names) or node.name.endswith("Strategy"):
                    found_classes.append(node.name)

        if not found_classes:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found_classes.append(node.name)

        if not found_classes:
            return ValidationResult(
                is_valid=False,
                error_message="No strategy class definition found in source code",
            )

        strategy_class_name = found_classes[0]
        return ValidationResult(is_valid=True, strategy_name=strategy_class_name)


class RuntimeStrategyValidator:
    """Validates materialized strategy file using real Freqtrade StrategyResolver."""

    def validate_materialized_strategy(
        self,
        strategy_path: str | Path,
        strategy_name: str,
        market_type: MarketType = MarketType.SPOT,
        config: dict[str, Any] | None = None,
    ) -> ValidationResult:
        path = Path(strategy_path).resolve()
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                error_message=f"Strategy directory/file does not exist: {path}",
            )

        extra_dir = str(path if path.is_dir() else path.parent)

        base_config: dict[str, Any] = {
            "strategy": strategy_name,
            "strategy_path": extra_dir,
            "user_data_dir": Path(path.parent.parent if path.is_dir() else path.parent),
            "trading_mode": TradingMode.FUTURES if market_type == MarketType.FUTURES else TradingMode.SPOT,
            "margin_mode": "isolated" if market_type == MarketType.FUTURES else None,
            "stake_currency": "USDT",
            "dry_run": True,
            "exchange": {"name": "binance", "key": "", "secret": ""},
        }
        if config:
            base_config.update(config)

        try:
            loaded_strategy = StrategyResolver.load_strategy(base_config)
            if loaded_strategy is None:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"StrategyResolver returned None for {strategy_name}",
                )
            return ValidationResult(is_valid=True, strategy_name=strategy_name)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                strategy_name=strategy_name,
                error_message=f"StrategyResolver validation failed: {str(e)}",
            )
