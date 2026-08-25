# research/trader_mining/engine.py
"""Trade reconstruction: groups a (trader, symbol)'s already-normalized fills into
logical ReconstructedTrade records, following Hyperliquid's own position transitions
(startPosition + signed quantity) rather than an imposed lot-accounting convention like
FIFO. See docs/superpowers/specs/2026-08-25-trader-mining-release-2-design.md for the
full algorithm writeup, including why reversal closed_pnl goes 100% to the closing leg
(corrected via external review, not the original proposal draft's proportional-split
guess) and why position tracking uses Decimal, not float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from research.models import NormalizedFill, ReconstructedTrade


@dataclass
class _TradeState:
    is_truncated_start: bool
    fallback_price: Decimal
    fallback_timestamp: datetime
    direction: str
    entry_notional: Decimal = field(default_factory=lambda: Decimal(0))
    entry_qty: Decimal = field(default_factory=lambda: Decimal(0))
    entry_timestamp: datetime | None = None
    exit_notional: Decimal = field(default_factory=lambda: Decimal(0))
    exit_qty: Decimal = field(default_factory=lambda: Decimal(0))
    exit_timestamp: datetime | None = None
    gross_pnl: Decimal = field(default_factory=lambda: Decimal(0))
    fees: Decimal = field(default_factory=lambda: Decimal(0))
    n_fills: int = 0
    was_liquidated: bool = False

    def add_entry(self, fill: NormalizedFill, qty: Decimal, fee: Decimal) -> None:
        if self.entry_timestamp is None:
            self.entry_timestamp = fill.timestamp
        self.entry_notional += Decimal(str(fill.price)) * qty
        self.entry_qty += qty
        self.fees += fee
        self.n_fills += 1
        if "Liquidat" in fill.direction:
            self.was_liquidated = True

    def add_exit(
        self, fill: NormalizedFill, qty: Decimal, closed_pnl: Decimal, fee: Decimal
    ) -> None:
        self.exit_timestamp = fill.timestamp
        self.exit_notional += Decimal(str(fill.price)) * qty
        self.exit_qty += qty
        self.gross_pnl += closed_pnl
        self.fees += fee
        self.n_fills += 1
        if "Liquidat" in fill.direction:
            self.was_liquidated = True

    def finalize(self, trader: str, symbol: str) -> ReconstructedTrade:
        entry_price = (
            float(self.entry_notional / self.entry_qty)
            if self.entry_qty
            else float(self.fallback_price)
        )
        entry_timestamp = self.entry_timestamp or self.fallback_timestamp
        exit_price = (
            float(self.exit_notional / self.exit_qty)
            if self.exit_qty
            else float(self.fallback_price)
        )
        exit_timestamp = self.exit_timestamp or self.fallback_timestamp
        quantity = float(self.exit_qty) if self.exit_qty else float(self.entry_qty)
        return ReconstructedTrade(
            trader=trader,
            symbol=symbol,
            direction=self.direction,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=float(self.gross_pnl),
            fees=float(self.fees),
            net_pnl=float(self.gross_pnl - self.fees),
            holding_time_seconds=(exit_timestamp - entry_timestamp).total_seconds(),
            n_fills=self.n_fills,
            is_truncated_start=self.is_truncated_start,
            was_liquidated=self.was_liquidated,
        )


def reconstruct_trades(
    trader: str, symbol: str, fills: list[NormalizedFill]
) -> list[ReconstructedTrade]:
    """fills must already be sorted by (timestamp, tid) -- not re-sorted here."""
    if not fills:
        return []

    for fill in fills:
        if fill.quantity <= 0:
            raise ValueError(f"fill tid={fill.tid} has non-positive quantity {fill.quantity}")

    trades: list[ReconstructedTrade] = []
    running_position = Decimal(str(fills[0].position))
    trade: _TradeState | None = None

    if running_position != 0:
        trade = _TradeState(
            is_truncated_start=True,
            fallback_price=Decimal(str(fills[0].price)),
            fallback_timestamp=fills[0].timestamp,
            direction="long" if running_position > 0 else "short",
        )

    for fill in fills:
        qty = Decimal(str(fill.quantity))
        signed_qty = qty if fill.side == "buy" else -qty
        end_position = running_position + signed_qty

        if trade is None:
            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )

        is_reversal = (
            running_position != 0
            and end_position != 0
            and (end_position > 0) != (running_position > 0)
        )

        if is_reversal:
            close_qty = abs(running_position)
            open_qty = qty - close_qty
            fee = Decimal(str(fill.fee))
            close_fee = fee * (close_qty / qty)
            trade.add_exit(fill, close_qty, Decimal(str(fill.closed_pnl)), close_fee)
            trades.append(trade.finalize(trader, symbol))

            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )
            trade.add_entry(fill, open_qty, fee - close_fee)
            running_position = end_position
            continue

        if abs(end_position) > abs(running_position):
            trade.add_entry(fill, qty, Decimal(str(fill.fee)))
        else:
            trade.add_exit(fill, qty, Decimal(str(fill.closed_pnl)), Decimal(str(fill.fee)))

        running_position = end_position

        if running_position == 0:
            trades.append(trade.finalize(trader, symbol))
            trade = None

    return trades
