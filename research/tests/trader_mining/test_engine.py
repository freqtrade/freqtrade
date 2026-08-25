# research/tests/trader_mining/test_engine.py
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, NormalizedFill, ReconstructedTrade
from research.trader_mining.engine import reconstruct_and_persist_trades, reconstruct_trades


TRADER = "0xAAA"
SYMBOL = "BTC/USDC:USDC"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _fill(
    tid,
    side,
    price,
    qty,
    position,
    ts=T0,
    closed_pnl=0.0,
    fee=0.1,
    direction="Open Long",
    symbol=SYMBOL,
    fee_currency="USDC",
):
    return NormalizedFill(
        trader=TRADER,
        tid=tid,
        timestamp=ts,
        symbol=symbol,
        side=side,
        price=price,
        quantity=qty,
        notional=price * qty,
        position=position,
        closed_pnl=closed_pnl,
        direction=direction,
        crossed=True,
        fee=fee,
        fee_currency=fee_currency,
        order_id=str(tid),
    )


def test_one_entry_one_exit():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            110.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=50.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(110.0)
    assert t.quantity == pytest.approx(5.0)
    assert t.gross_pnl == pytest.approx(50.0)
    assert t.fees == pytest.approx(0.2)
    assert t.net_pnl == pytest.approx(49.8)
    assert t.n_fills == 2
    assert t.is_truncated_start is False
    assert t.was_liquidated is False
    assert t.holding_time_seconds == pytest.approx(3600.0)


def test_scale_in_weighted_average_entry_price():
    fills = [
        _fill(1, "buy", 100.0, 3.0, position=0.0, ts=T0),
        _fill(2, "buy", 106.0, 2.0, position=3.0, ts=T0 + timedelta(minutes=1)),
        _fill(
            3,
            "sell",
            120.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=88.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    # (100*3 + 106*2) / 5 = 102.4
    assert trades[0].entry_price == pytest.approx(102.4)
    assert trades[0].quantity == pytest.approx(5.0)


def test_scale_out_weighted_average_exit_price():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            108.0,
            2.0,
            position=5.0,
            ts=T0 + timedelta(minutes=1),
            closed_pnl=16.0,
            direction="Close Long",
        ),
        _fill(
            3,
            "sell",
            112.0,
            3.0,
            position=3.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=36.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    # (108*2 + 112*3) / 5 = 110.4
    assert trades[0].exit_price == pytest.approx(110.4)
    assert trades[0].gross_pnl == pytest.approx(52.0)


def test_reversal_with_followup_close_produces_two_trades():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        # reversal: +5 -> sell 8 -> -3. Closing leg qty=5 (full closed_pnl=40, fee split
        # 5/8), opening leg qty=3 (closed_pnl=0, fee split 3/8).
        _fill(
            2,
            "sell",
            120.0,
            8.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=40.0,
            fee=0.8,
            direction="Close Long",
        ),
        # closes the new -3 short
        _fill(
            3,
            "buy",
            115.0,
            3.0,
            position=-3.0,
            ts=T0 + timedelta(hours=2),
            closed_pnl=15.0,
            direction="Close Short",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 2
    closed_long, closed_short = trades[0], trades[1]
    assert closed_long.direction == "long"
    assert closed_long.quantity == pytest.approx(5.0)
    assert closed_long.gross_pnl == pytest.approx(40.0)
    assert closed_long.fees == pytest.approx(0.1 + 0.8 * (5 / 8))
    assert closed_long.n_fills == 2

    assert closed_short.direction == "short"
    assert closed_short.quantity == pytest.approx(3.0)
    assert closed_short.entry_price == pytest.approx(120.0)
    assert closed_short.gross_pnl == pytest.approx(15.0)  # 0 from reversal + 15 from close
    assert closed_short.fees == pytest.approx(0.8 * (3 / 8) + 0.1)
    assert closed_short.n_fills == 2


def test_reversal_with_no_followup_produces_only_the_closing_trade():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            120.0,
            8.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=40.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    # the newly-opened -3 short is still open -- not emitted
    assert len(trades) == 1
    assert trades[0].direction == "long"
    assert trades[0].quantity == pytest.approx(5.0)


def test_truncated_start_with_no_observed_entry_falls_back_to_first_fill():
    fills = [
        _fill(
            1,
            "sell",
            110.0,
            5.0,
            position=5.0,  # non-zero on the very first fill -- predates history
            ts=T0,
            closed_pnl=25.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.is_truncated_start is True
    assert t.direction == "long"
    # no entry-like fill was observed -- entry falls back to this fill's own price/time
    assert t.entry_price == pytest.approx(110.0)
    assert t.entry_timestamp == T0
    assert t.exit_price == pytest.approx(110.0)
    assert t.gross_pnl == pytest.approx(25.0)


def test_truncated_start_with_one_observed_entry_fill():
    fills = [
        _fill(1, "buy", 100.0, 2.0, position=3.0, ts=T0),  # 3 -> 5, scale-in observed
        _fill(
            2,
            "sell",
            120.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=60.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.is_truncated_start is True
    # entry_price reflects ONLY the observed 2.0-qty entry fill, not the unobserved 3.0
    assert t.entry_price == pytest.approx(100.0)
    assert t.quantity == pytest.approx(5.0)


def test_liquidation_fill_flags_the_trade():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            80.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=-100.0,
            direction="Liquidated Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert trades[0].was_liquidated is True


def test_nonpositive_quantity_raises():
    fills = [_fill(1, "buy", 100.0, 0.0, position=0.0, ts=T0)]

    with pytest.raises(ValueError, match="tid=1"):
        reconstruct_trades(TRADER, SYMBOL, fills)


def test_unclosed_trailing_position_is_not_emitted():
    fills = [_fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0)]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert trades == []


def test_empty_fills_returns_empty_list():
    assert reconstruct_trades(TRADER, SYMBOL, []) == []


def test_long_fill_sequence_scaling_repeatedly_closes_exactly():
    """Regression guard for float-drift risk -- Decimal arithmetic must close this
    exactly even after 40 scale-in/scale-out fills that would accumulate float error if
    summed naively."""
    fills = []
    position = 0.0
    tid = 1
    ts = T0
    for i in range(20):
        fills.append(_fill(tid, "buy", 100.0 + i, 0.1, position=position, ts=ts))
        position += 0.1
        tid += 1
        ts += timedelta(minutes=1)
    for i in range(19):
        fills.append(_fill(tid, "sell", 100.0 + i, 0.1, position=position, ts=ts, closed_pnl=0.01))
        position -= 0.1
        tid += 1
        ts += timedelta(minutes=1)
    # final fill fully closes the remaining 0.1
    fills.append(_fill(tid, "sell", 105.0, 0.1, position=position, ts=ts, closed_pnl=0.01))

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    assert trades[0].quantity == pytest.approx(2.0)


def test_failure_on_one_symbol_rolls_back_and_leaves_other_symbols_untouched():
    """If reconstruction blows up partway through a multi-symbol trader (e.g. a
    position-gap ValueError on the second symbol), the session must not be left with a
    dangling delete()/add() for the symbol that already succeeded -- and a caller who
    catches the exception and keeps using the session shouldn't inherit a poisoned
    transaction."""
    session = _memory_session()
    _add_normalized_fill(
        session, tid=1, symbol="BTC/USDC:USDC", side="buy", position=0.0, timestamp=T0
    )
    _add_normalized_fill(
        session,
        tid=2,
        symbol="BTC/USDC:USDC",
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    # ETH fills have a position gap: fill 3 ends at 5.0, fill 4 claims it started at 9.0
    _add_normalized_fill(
        session, tid=3, symbol="ETH/USDC:USDC", side="buy", position=0.0, timestamp=T0
    )
    _add_normalized_fill(
        session,
        tid=4,
        symbol="ETH/USDC:USDC",
        side="sell",
        position=9.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=10.0,
        direction="Close Long",
    )
    session.commit()

    with pytest.raises(ValueError, match="position gap"):
        reconstruct_and_persist_trades(session, TRADER)

    # nothing was committed -- not even the BTC symbol that reconstructed successfully
    # before ETH blew up -- and the session is still usable afterwards.
    assert session.query(ReconstructedTrade).count() == 0
    assert session.query(NormalizedFill).count() == 4


def test_base_asset_fee_in_kind_nets_out_of_position_not_flagged_as_gap():
    """Regression test for a real bug found live-validating trader-analyze against real
    leaderboard wallets: on Hyperliquid spot markets, a fill's fee can be charged in-kind
    in the base asset being received (e.g. buying HYPE/USDC with the fee taken in HYPE),
    not in the quote currency. Hyperliquid's own reported position nets this out --
    confirmed against real data: buying qty=66.66 HYPE with fee=0.026664 HYPE left the
    position up by exactly 66.66 - 0.026664 = 66.633336, not the full 66.66. Before this
    fix, the position-gap guard (added for the earlier code review) raised a false
    positive on every such fill."""
    fills = [
        _fill(
            1,
            "buy",
            100.0,
            10.0,
            position=0.0,
            ts=T0,
            fee=0.05,
            fee_currency="HYPE",
            symbol="HYPE/USDC",
        ),
        _fill(
            2,
            "sell",
            110.0,
            9.95,  # can only sell what was actually received net of the in-kind fee
            position=9.95,
            ts=T0 + timedelta(hours=1),
            fee=0.1,
            fee_currency="USDC",
            closed_pnl=99.5,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    trades = reconstruct_trades(TRADER, "HYPE/USDC", fills)

    assert len(trades) == 1
    assert trades[0].quantity == pytest.approx(9.95)


def test_base_asset_fee_converted_to_quote_currency_before_subtracting_from_pnl():
    """The in-kind HYPE fee must be converted to USDC (via the fill's own execution
    price) before being subtracted from PnL -- summing raw fee units regardless of
    currency would mix HYPE-denominated and USDC-denominated numbers into one figure."""
    fills = [
        _fill(
            1,
            "buy",
            100.0,
            10.0,
            position=0.0,
            ts=T0,
            fee=0.05,
            fee_currency="HYPE",
            symbol="HYPE/USDC",
        ),
        _fill(
            2,
            "sell",
            110.0,
            9.95,
            position=9.95,
            ts=T0 + timedelta(hours=1),
            fee=0.0,
            fee_currency="USDC",
            closed_pnl=99.5,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    trades = reconstruct_trades(TRADER, "HYPE/USDC", fills)

    # 0.05 HYPE * 100.0 USDC/HYPE (this fill's own price) = 5.0 USDC -- not raw 0.05.
    assert trades[0].fees == pytest.approx(5.0)
    assert trades[0].net_pnl == pytest.approx(99.5 - 5.0)


def test_quote_currency_fee_on_a_non_usdc_quoted_pair_does_not_affect_position():
    """A fee denominated in the pair's OWN quote currency (not USDC, not the base asset
    -- e.g. HYPE/USDT paying fees in USDT0) must NOT be treated as an in-kind base-asset
    fee. Real data check that killed a naive 'fee_currency != USDC' heuristic: Hyperliquid
    has non-USDC-quoted spot pairs, so 'not USDC' does not imply 'is the base asset'."""
    fills = [
        _fill(
            1,
            "buy",
            100.0,
            10.0,
            position=0.0,
            ts=T0,
            fee=0.5,
            fee_currency="USDT0",  # HYPE/USDT's quote currency, not HYPE
            symbol="HYPE/USDT",
        ),
        _fill(
            2,
            "sell",
            110.0,
            10.0,  # full qty received -- the fee did NOT come out of the base asset
            position=10.0,
            ts=T0 + timedelta(hours=1),
            fee=0.1,
            fee_currency="USDT0",
            closed_pnl=100.0,
            direction="Close Long",
            symbol="HYPE/USDT",
        ),
    ]

    trades = reconstruct_trades(TRADER, "HYPE/USDT", fills)

    assert len(trades) == 1
    assert trades[0].quantity == pytest.approx(10.0)
    # both fees are USDT0 (the quote currency) -- summed as-is, no price conversion
    assert trades[0].fees == pytest.approx(0.6)


def test_reconciled_gap_does_not_raise_and_continues_reconstruction():
    """The exact scenario that motivated this feature (shape matches the real wallet):
    a truncated-start position of 70000.0 is topped up by 8008.0 to 78008.0, then a
    spotTransfer moves 62264.0 HYPE OUT of the wallet, leaving 15744.0 -- which the
    next fill fully closes. A real spot position can never go negative, so the
    transfer must come out of an existing large holding, not exceed it (an earlier,
    unrealistic draft of this test had the transfer exceed the whole position)."""
    fills = [
        _fill(1, "buy", 100.0, 8008.0, position=70000.0, ts=T0, symbol="HYPE/USDC"),
        _fill(
            2,
            "sell",
            110.0,
            15744.0,
            position=15744.0,  # 78008.0 - 62264.0, the reconciled real gap shape
            ts=T0 + timedelta(hours=1),
            closed_pnl=1000.0,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    def fake_reconcile(asset, window_start, window_end):
        assert asset == "HYPE"
        return Decimal("-62264.0")

    reconciled_gaps: list[str] = []
    trades = reconstruct_trades(
        TRADER, "HYPE/USDC", fills, reconcile=fake_reconcile, reconciled_gaps=reconciled_gaps
    )

    assert len(trades) == 1
    assert len(reconciled_gaps) == 1
    assert "HYPE/USDC" in reconciled_gaps[0]


def test_unreconciled_gap_still_raises_even_with_reconcile_supplied():
    """reconcile is consulted but doesn't explain the gap -- must still hard-fail, never
    silently proceed."""
    fills = [
        _fill(1, "buy", 100.0, 10.0, position=0.0, ts=T0, symbol="HYPE/USDC"),
        _fill(
            2,
            "sell",
            110.0,
            10.0,
            position=999.0,  # nothing explains this
            ts=T0 + timedelta(hours=1),
            closed_pnl=100.0,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    def fake_reconcile(asset, window_start, window_end):
        return Decimal(0)  # ledger has nothing for this window

    with pytest.raises(ValueError, match="position gap"):
        reconstruct_trades(TRADER, "HYPE/USDC", fills, reconcile=fake_reconcile)


def test_position_gap_between_fills_raises():
    """A gap in ingested history (e.g. the 10,000-fill provider ceiling, or an
    interrupted trader-import run) must not silently produce wrong trade boundaries --
    if one fill's ending position doesn't match the next fill's own reported starting
    position, that's a real discontinuity, not something to paper over."""
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        # after fill 1: position should be 5.0, but this fill claims it started at 9.0
        # -- a gap of missing fills in between.
        _fill(
            2,
            "sell",
            110.0,
            5.0,
            position=9.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=50.0,
            direction="Close Long",
        ),
    ]

    with pytest.raises(ValueError, match="position gap"):
        reconstruct_trades(TRADER, SYMBOL, fills)


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _add_normalized_fill(session, **kwargs):
    defaults = {
        "trader": TRADER,
        "symbol": SYMBOL,
        "side": "buy",
        "price": 100.0,
        "quantity": 5.0,
        "notional": 500.0,
        "position": 0.0,
        "closed_pnl": 0.0,
        "direction": "Open Long",
        "crossed": True,
        "fee": 0.1,
        "fee_currency": "USDC",
        "order_id": "1",
    }
    defaults.update(kwargs)
    session.add(NormalizedFill(**defaults))


def test_persists_reconstructed_trades_from_stored_fills():
    session = _memory_session()
    _add_normalized_fill(session, tid=1, side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=2,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 1
    assert result.symbols == [SYMBOL]
    assert session.query(ReconstructedTrade).count() == 1


def test_rerunning_after_new_fills_replaces_not_duplicates():
    session = _memory_session()
    _add_normalized_fill(session, tid=1, side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=2,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    session.commit()
    reconstruct_and_persist_trades(session, TRADER)

    # a second, independent round-trip trade for the same trader/symbol
    _add_normalized_fill(
        session, tid=3, side="buy", position=0.0, timestamp=T0 + timedelta(hours=2)
    )
    _add_normalized_fill(
        session,
        tid=4,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=3),
        closed_pnl=20.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 2
    assert session.query(ReconstructedTrade).count() == 2


def test_scoping_to_one_symbol_leaves_other_symbols_untouched():
    session = _memory_session()
    _add_normalized_fill(
        session, tid=1, symbol="BTC/USDC:USDC", side="buy", position=0.0, timestamp=T0
    )
    _add_normalized_fill(
        session,
        tid=2,
        symbol="BTC/USDC:USDC",
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    _add_normalized_fill(
        session, tid=3, symbol="ETH/USDC:USDC", side="buy", position=0.0, timestamp=T0
    )
    _add_normalized_fill(
        session,
        tid=4,
        symbol="ETH/USDC:USDC",
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=10.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER, symbol="BTC/USDC:USDC")

    assert result.n_trades == 1
    assert result.symbols == ["BTC/USDC:USDC"]
    assert session.query(ReconstructedTrade).count() == 1


def test_same_timestamp_ties_ordered_by_position_magnitude_not_tid():
    """Regression test for a real bug found validating against a real active wallet's
    fill history (Hyperliquid mainnet, via trader-import): tid is NOT a monotonic
    sequence number. A batch of same-millisecond fills sorted by tid came out in
    EXACTLY REVERSED chronological order (confirmed against the real fills' position
    arithmetic). Reproduced here with tid values deliberately DESCENDING in true
    execution order, so a naive (timestamp, tid) sort processes them backwards.
    (timestamp, abs(position)) recovers the true order."""
    session = _memory_session()
    ts = T0
    # True order: pos 0 -> 3 (entry qty3) -> 8 (entry qty5) -> 0 (exit qty8, closes).
    _add_normalized_fill(
        session, tid=300, side="buy", price=100.0, quantity=3.0, position=0.0, timestamp=ts
    )
    _add_normalized_fill(
        session, tid=200, side="buy", price=110.0, quantity=5.0, position=3.0, timestamp=ts
    )
    _add_normalized_fill(
        session,
        tid=100,
        side="sell",
        price=120.0,
        quantity=8.0,
        position=8.0,
        timestamp=ts,
        closed_pnl=40.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 1
    trade = session.query(ReconstructedTrade).one()
    assert trade.is_truncated_start is False
    # (100*3 + 110*5) / 8 = 106.25 -- under the old tid-sorted bug this would instead
    # produce a spurious truncated-start trade with entry_price wrongly falling back to
    # 120.0 (the sell fill's own price), and the two real entry fills left as an
    # unclosed, unemitted trade.
    assert trade.entry_price == pytest.approx(106.25)
    assert trade.exit_price == pytest.approx(120.0)
    assert trade.gross_pnl == pytest.approx(40.0)


def test_reconstruct_and_persist_trades_reports_reconciled_gaps():
    """The acceptance scenario for the ledger-reconciliation feature: HYPE fills
    mirroring the real gap (large existing holding, a transfer within it, still
    positive throughout -- see the reconstruct_trades-level test above for why),
    plus the ledger event that explains it."""
    import json

    from research.models import RawLedgerEvent

    session = _memory_session()
    _add_normalized_fill(
        session,
        tid=1,
        symbol="HYPE/USDC",
        side="buy",
        price=100.0,
        quantity=8008.0,
        position=70000.0,
        timestamp=T0,
    )
    _add_normalized_fill(
        session,
        tid=2,
        symbol="HYPE/USDC",
        side="sell",
        price=110.0,
        quantity=15744.0,
        position=15744.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=1000.0,
        direction="Close Long",
    )
    session.add(
        RawLedgerEvent(
            trader=TRADER,
            event_id="0xdeadbeef",
            event_type="spotTransfer",
            timestamp=T0 + timedelta(minutes=30),
            info_json=json.dumps(
                {
                    "delta": {
                        "type": "spotTransfer",
                        "token": "HYPE",
                        "amount": "62264.0",
                        "user": TRADER,
                        "destination": "0xother",
                    }
                }
            ),
            retrieved_at=datetime.now(UTC),
        )
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 1
    assert len(result.reconciled_gaps) == 1
    assert "HYPE/USDC" in result.reconciled_gaps[0]
