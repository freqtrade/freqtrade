"""Deterministic trading universe definition."""

from __future__ import annotations

from dataclasses import dataclass, field

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class TradingUniverse:
    """Platform-owned definition of eligible tradable symbols for a market profile."""

    universe_id: str
    exchange: str
    market_type: str
    include_symbols: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    max_symbols: int | None = None
    enabled: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()
        self.include_symbols = self._normalize_symbols(self.include_symbols)
        self.exclude_symbols = self._normalize_symbols(self.exclude_symbols)
        self.include_symbols = [symbol for symbol in self.include_symbols if symbol not in self.exclude_symbols]
        if self.max_symbols is not None and self.max_symbols < 0:
            raise PlatformValidationError("max_symbols cannot be negative")

    def validate(self) -> None:
        if not self.universe_id or not self.universe_id.strip():
            raise ValueError("universe_id is required")
        if not self.exchange or not self.exchange.strip():
            raise ValueError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise ValueError("market_type is required")

    @staticmethod
    def _normalize_symbols(symbols: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol is None:
                continue
            cleaned = str(symbol).strip()
            if not cleaned:
                continue
            canonical = cleaned.upper()
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized

    def contains(self, symbol: str) -> bool:
        if not self.enabled:
            return False
        canonical = str(symbol).strip().upper()
        if not canonical:
            return False
        if canonical in self.exclude_symbols:
            return False
        if not self.include_symbols:
            return True
        return canonical in self.include_symbols

    def eligible_symbols(self, all_available_symbols: list[str]) -> list[str]:
        if not self.enabled:
            return []
        pool = self._normalize_symbols(all_available_symbols)
        if not self.include_symbols:
            eligible = [symbol for symbol in pool if symbol not in self.exclude_symbols]
        else:
            eligible = [symbol for symbol in pool if self.contains(symbol)]
        if self.max_symbols is not None:
            eligible = eligible[: self.max_symbols]
        return eligible

    def add_symbol(self, symbol: str) -> None:
        normalized = self._normalize_symbols([symbol])
        for item in normalized:
            if item not in self.include_symbols:
                self.include_symbols.append(item)
            self.exclude_symbols = [excluded for excluded in self.exclude_symbols if excluded != item]

    def remove_symbol(self, symbol: str) -> None:
        canonical = str(symbol).strip().upper()
        self.include_symbols = [entry for entry in self.include_symbols if entry != canonical]
        self.exclude_symbols = [entry for entry in self.exclude_symbols if entry != canonical]

    def exclude_symbol(self, symbol: str) -> None:
        canonical = str(symbol).strip().upper()
        if canonical and canonical not in self.exclude_symbols:
            self.exclude_symbols.append(canonical)
        self.include_symbols = [entry for entry in self.include_symbols if entry != canonical]

    def include_symbol(self, symbol: str) -> None:
        self.add_symbol(symbol)

    def __contains__(self, symbol: str) -> bool:
        return self.contains(symbol)
