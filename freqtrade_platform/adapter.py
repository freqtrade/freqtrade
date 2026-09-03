"""Integration boundary between the platform layer and Freqtrade runtime components.

This adapter is intentionally narrow: it exposes a stable contract for later platform services
without duplicating Freqtrade's trading engine or creating a second exchange abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FreqtradeAdapter:
    """Read-only adapter boundary for future platform integration with Freqtrade.

    The actual bot, strategy resolver, wallets, exchange, and backtesting objects remain owned by
    Freqtrade. This adapter simply holds references and defines the integration seam.
    """

    bot: Any | None = None
    data_provider: Any | None = None
    wallets: Any | None = None
    strategy_resolver: Any | None = None
    backtesting: Any | None = None
    rpc_api: Any | None = None
    exchange: Any | None = None
    trade: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def inspect_trading_state(self) -> dict[str, Any]:
        """Return a lightweight view of the current Freqtrade state for platform consumers."""
        return {
            "bot": self.bot is not None,
            "data_provider": self.data_provider is not None,
            "wallets": self.wallets is not None,
            "strategy_resolver": self.strategy_resolver is not None,
            "backtesting": self.backtesting is not None,
            "exchange": self.exchange is not None,
        }
