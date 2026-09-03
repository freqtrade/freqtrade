"""Service boundary for future account state inspection and composition."""

from __future__ import annotations

from typing import Any

from freqtrade_platform.account.models import AccountSnapshot


class AccountService:
    """Provides a read-only boundary for account state discovery.

    No second exchange client or live account synchronization is implemented in this phase.
    """

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    def snapshot(self, *, simulated: bool = False, **kwargs: Any) -> AccountSnapshot:
        """Return a snapshot object using either real or simulated values."""
        if simulated:
            return AccountSnapshot(
                timestamp=kwargs.get("timestamp", "1970-01-01T00:00:00Z"),
                exchange=kwargs.get("exchange", "binance"),
                market_type=kwargs.get("market_type", "spot"),
                available_balance=float(kwargs.get("simulated_balance", 0.0)),
                total_balance=float(kwargs.get("simulated_total_balance", 0.0)),
                equity=float(kwargs.get("simulated_equity", 0.0)),
                positions=kwargs.get("positions", {}),
                raw_source_metadata=kwargs.get("raw_source_metadata", {}),
                simulated_balance=float(kwargs.get("simulated_balance", 0.0)),
                simulated_equity=float(kwargs.get("simulated_equity", 0.0)),
            )

        return AccountSnapshot(
            timestamp=kwargs.get("timestamp", "1970-01-01T00:00:00Z"),
            exchange=kwargs.get("exchange", "binance"),
            market_type=kwargs.get("market_type", "spot"),
            available_balance=float(kwargs.get("available_balance", 0.0)),
            total_balance=float(kwargs.get("total_balance", 0.0)),
            equity=float(kwargs.get("equity", 0.0)),
            positions=kwargs.get("positions", {}),
            raw_source_metadata=kwargs.get("raw_source_metadata", {}),
            simulated_balance=kwargs.get("simulated_balance"),
            simulated_equity=kwargs.get("simulated_equity"),
        )
