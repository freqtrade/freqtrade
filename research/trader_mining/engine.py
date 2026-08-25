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

from sqlalchemy import func
from sqlalchemy.orm import Session

from research.models import NormalizedFill, ReconstructedTrade
from research.trader_mining.symbols import base_asset_of


_KNOWN_QUOTE_CURRENCIES = frozenset({"USDC", "USDT", "USDT0"})


def _is_base_asset_fee(fill: NormalizedFill) -> bool:
    """True if this fill's fee was charged in-kind, in the traded symbol's own base
    asset, rather than in the pair's quote/settlement currency. Hyperliquid spot fees
    are deducted from whichever asset the fill actually delivers to the trader -- in
    practice the base asset on a buy, the quote asset on a sell (confirmed against real
    wallet data: a HYPE/USDC buy fill's own reported post-fill position nets out its
    HYPE-denominated fee exactly).

    Determined by comparing fee_currency to the base asset parsed from `symbol` (the
    part before "/") when parseable -- this is what catches HYPE/USDT paying its fee in
    USDT0 (the pair's own quote currency, not USDC and not the base asset) as NOT an
    in-kind fee; a naive "fee_currency != USDC" check would have gotten that wrong,
    since Hyperliquid has non-USDC-quoted spot pairs. ponytail: a handful of very new
    spot markets surface as an unparsable raw internal index (e.g. "@705", no "/") --
    for those, fall back to "not a known quote currency"; confirmed against real data
    (fee_currency='SKHYX' on symbol '@705').
    """
    base = base_asset_of(fill.symbol)
    if base is not None:
        return fill.fee_currency == base
    return fill.fee_currency not in _KNOWN_QUOTE_CURRENCIES


def _fee_in_quote_currency(fill: NormalizedFill, fee: Decimal) -> Decimal:
    """Convert an in-kind (base-asset-denominated) fee to the quote currency using the
    fill's own execution price (already expressed as quote-per-base for this fill) --
    otherwise summing fee amounts across mixed currencies (e.g. HYPE and USDC) into one
    `fees`/`net_pnl` figure would be nonsensical unit-mixing. A quote-currency fee is
    already in the right units and passes through unchanged."""
    if _is_base_asset_fee(fill):
        return fee * Decimal(str(fill.price))
    return fee


def _end_position(running_position: Decimal, fill: NormalizedFill) -> Decimal:
    """Position after applying `fill` to `running_position` -- an in-kind
    (base-asset-denominated) fee reduces the actual position change; see
    _is_base_asset_fee. A quote-currency fee never touches position."""
    qty = Decimal(str(fill.quantity))
    position_delta = qty if fill.side == "buy" else -qty
    if _is_base_asset_fee(fill):
        position_delta -= Decimal(str(fill.fee))
    return running_position + position_delta


def _check_position_continuity(
    fill: NormalizedFill, end_position: Decimal, next_fill: NormalizedFill
) -> None:
    """A gap in ingested history (the 10,000-fill provider ceiling, an interrupted
    trader-import run, or an external transfer never captured as a trade fill at all)
    must not silently produce wrong trade boundaries -- if `fill`'s computed end
    position doesn't match `next_fill`'s own reported starting position, that's a real
    discontinuity, not something to paper over."""
    next_position = Decimal(str(next_fill.position))
    # epsilon, not exact equality: `position` values recorded on real fills (and in
    # hand-built test fixtures that accumulate float positions incrementally) can differ
    # from our own Decimal-summed end_position by float-repr noise even when there's no
    # real gap.
    if abs(next_position - end_position) > Decimal("1e-8"):
        raise ValueError(
            f"position gap: fill tid={fill.tid} ends at position {end_position} but "
            f"next fill tid={next_fill.tid} starts at position {next_position} -- "
            "likely missing fills between them (ingestion gap or provider truncation)"
        )


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
    """fills must already be sorted by true execution order -- not re-sorted here. The
    caller (reconstruct_and_persist_trades) sorts by (timestamp, abs(position), tid),
    NOT (timestamp, tid) -- tid is not a monotonic sequence number, confirmed against a
    real active wallet's fill history (see that function's own docstring)."""
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

    for i, fill in enumerate(fills):
        qty = Decimal(str(fill.quantity))
        end_position = _end_position(running_position, fill)

        if i + 1 < len(fills):
            _check_position_continuity(fill, end_position, fills[i + 1])

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
            fee = _fee_in_quote_currency(fill, Decimal(str(fill.fee)))
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

        fee = _fee_in_quote_currency(fill, Decimal(str(fill.fee)))
        if abs(end_position) > abs(running_position):
            trade.add_entry(fill, qty, fee)
        else:
            trade.add_exit(fill, qty, Decimal(str(fill.closed_pnl)), fee)

        running_position = end_position

        if running_position == 0:
            trades.append(trade.finalize(trader, symbol))
            trade = None

    return trades


@dataclass
class ReconstructResult:
    n_trades: int
    symbols: list[str]


def reconstruct_and_persist_trades(
    session: Session, trader: str, symbol: str | None = None
) -> ReconstructResult:
    """Recompute (trader, symbol)'s trades from scratch -- deletes existing
    ReconstructedTrade rows for the scope and re-derives from every currently-stored
    NormalizedFill, rather than incrementally patching. Simpler and more correct: a
    later trader-import run backfilling older history could retroactively change
    earlier trade boundaries, which incremental reconciliation can't handle cleanly.

    Fills are ordered by (timestamp, abs(position)), NOT (timestamp, tid) -- found to be
    a real bug, not a theoretical one, while validating against a real active wallet's
    fill history: Hyperliquid's tid is not a monotonic sequence number, and a batch of
    same-millisecond fills sorted by tid came out in EXACTLY REVERSED chronological
    order (verified against the position arithmetic: fill N's startPosition + its own
    signed quantity landed within rounding of fill N+1's startPosition, in the abs
    (position)-ascending order, not the tid-ascending order). abs(position) recovers
    true execution order correctly for same-direction accumulation ties (the only case
    observed), including short accumulation (magnitude still grows away from zero).

    ponytail: known residual risk, not fixed here -- a REVERSAL fill landing in the same
    tied-timestamp group as other fills could sort incorrectly, since abs(position) isn't
    monotonic across a sign change. Narrower and rarer than the tid bug this replaces
    (needs same-millisecond fills AND a reversal within that exact group), not observed
    in the real wallet data validated so far. Upgrade path if this bites: resolve ties by
    the actual position-delta chain (which fill's end_position matches another's
    start_position) rather than a single sort key.
    """
    symbols_query = session.query(NormalizedFill.symbol).filter(NormalizedFill.trader == trader)
    if symbol is not None:
        symbols_query = symbols_query.filter(NormalizedFill.symbol == symbol)
    symbols = sorted({s for (s,) in symbols_query.distinct().all()})

    total_trades = 0
    try:
        for sym in symbols:
            fills = (
                session.query(NormalizedFill)
                .filter(NormalizedFill.trader == trader, NormalizedFill.symbol == sym)
                .order_by(
                    NormalizedFill.timestamp,
                    func.abs(NormalizedFill.position),
                    NormalizedFill.tid,
                )
                .all()
            )
            session.query(ReconstructedTrade).filter(
                ReconstructedTrade.trader == trader, ReconstructedTrade.symbol == sym
            ).delete()

            new_trades = reconstruct_trades(trader, sym, fills)
            for t in new_trades:
                session.add(t)
            total_trades += len(new_trades)
    except Exception:
        session.rollback()
        raise

    session.commit()
    return ReconstructResult(n_trades=total_trades, symbols=symbols)
