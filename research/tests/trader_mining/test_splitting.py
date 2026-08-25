from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, ReconstructedTrade
from research.trader_mining.splitting import (
    PeriodBoundaries,
    _to_naive_utc,
    assign_period,
    split_trades,
    straddles_boundary,
)


def test_to_naive_utc_strips_tzinfo_from_utc_datetime():
    assert _to_naive_utc(datetime(2025, 1, 1, tzinfo=UTC)) == datetime(2025, 1, 1)


def test_to_naive_utc_leaves_naive_input_untouched():
    """A naive datetime in this codebase always means 'already UTC' -- the ingestion
    invariant -- so it is never treated as a timezone-unknown error."""
    assert _to_naive_utc(datetime(2025, 1, 1)) == datetime(2025, 1, 1)


def test_to_naive_utc_converts_non_utc_offset_before_stripping():
    plus_five = timezone(timedelta(hours=5))
    dt = datetime(2025, 1, 1, 5, 0, tzinfo=plus_five)  # 2025-01-01T00:00 UTC
    assert _to_naive_utc(dt) == datetime(2025, 1, 1, 0, 0)


def test_boundaries_normalize_tz_aware_input_to_naive_utc():
    b = PeriodBoundaries(
        train_end=datetime(2025, 1, 1, tzinfo=UTC),
        validation_end=datetime(2025, 7, 1, tzinfo=UTC),
        test_end=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert b.train_end == datetime(2025, 1, 1)
    assert b.train_end.tzinfo is None
    assert b.validation_end == datetime(2025, 7, 1)
    assert b.test_end == datetime(2026, 1, 1)


def test_boundaries_treat_naive_input_as_already_utc():
    b = PeriodBoundaries(
        train_end=datetime(2025, 1, 1),
        validation_end=datetime(2025, 7, 1),
        test_end=datetime(2026, 1, 1),
    )

    assert b.train_end == datetime(2025, 1, 1)


def test_boundaries_reject_non_strictly_increasing_dates():
    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        PeriodBoundaries(
            train_end=datetime(2025, 7, 1),
            validation_end=datetime(2025, 1, 1),  # before train_end
            test_end=datetime(2026, 1, 1),
        )


def test_boundaries_reject_equal_dates():
    """Equal boundaries would silently produce an empty period -- rejected, not tolerated."""
    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        PeriodBoundaries(
            train_end=datetime(2025, 1, 1),
            validation_end=datetime(2025, 1, 1),
            test_end=datetime(2026, 1, 1),
        )


BOUNDARIES = PeriodBoundaries(
    train_end=datetime(2025, 1, 1, tzinfo=UTC),
    validation_end=datetime(2025, 7, 1, tzinfo=UTC),
    test_end=datetime(2026, 1, 1, tzinfo=UTC),
)


def _trade(entry_ts, exit_ts=None) -> ReconstructedTrade:
    exit_ts = exit_ts if exit_ts is not None else entry_ts
    return ReconstructedTrade(
        trader="0xAAA",
        symbol="BTC/USDC:USDC",
        direction="long",
        entry_timestamp=entry_ts,
        entry_price=100.0,
        exit_timestamp=exit_ts,
        exit_price=100.0,
        quantity=1.0,
        gross_pnl=10.0,
        fees=0.0,
        net_pnl=10.0,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_assign_period_uses_entry_timestamp_not_exit_timestamp():
    """A trade entering in TRAIN but exiting in VALIDATION is still TRAIN -- assignment is
    entirely a function of the entry decision point, never the (only-knowable-at-exit)
    outcome timing."""
    t = _trade(entry_ts=datetime(2024, 6, 1, tzinfo=UTC), exit_ts=datetime(2025, 3, 1, tzinfo=UTC))

    assert assign_period(t, BOUNDARIES) == "TRAIN"


@pytest.mark.parametrize(
    "entry_ts,expected",
    [
        (datetime(2020, 1, 1, tzinfo=UTC), "TRAIN"),
        (datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC), "TRAIN"),
        (datetime(2025, 1, 1, tzinfo=UTC), "VALIDATION"),  # boundary starts the next period
        (datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC), "VALIDATION"),
        (datetime(2025, 7, 1, tzinfo=UTC), "TEST"),
        (datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC), "TEST"),
        (datetime(2026, 1, 1, tzinfo=UTC), "FORWARD"),
        (datetime(2030, 1, 1, tzinfo=UTC), "FORWARD"),  # FORWARD has no upper bound
    ],
)
def test_assign_period_boundary_semantics_are_half_open(entry_ts, expected):
    assert assign_period(_trade(entry_ts=entry_ts), BOUNDARIES) == expected


def test_assign_period_works_with_naive_entry_timestamp_and_tz_aware_boundaries():
    """Simulated version of the SQLite-naive-read landmine (see the real round-trip test
    below for the end-to-end proof): a tz-aware-boundary comparison against a naive
    (implicitly-UTC) trade timestamp must not raise TypeError."""
    t = _trade(entry_ts=datetime(2025, 3, 1))  # naive

    assert assign_period(t, BOUNDARIES) == "VALIDATION"


def test_straddles_boundary_true_when_entry_and_exit_periods_differ():
    t = _trade(entry_ts=datetime(2024, 12, 1, tzinfo=UTC), exit_ts=datetime(2025, 2, 1, tzinfo=UTC))

    assert straddles_boundary(t, BOUNDARIES) is True


def test_straddles_boundary_false_when_entry_and_exit_in_same_period():
    t = _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC), exit_ts=datetime(2025, 4, 1, tzinfo=UTC))

    assert straddles_boundary(t, BOUNDARIES) is False


def test_split_trades_every_trade_appears_exactly_once():
    """No-overlap guarantee, enforced by test: partition 6 trades and confirm the four
    buckets are disjoint and their union recovers every input trade exactly once."""
    trades = [
        _trade(entry_ts=datetime(2024, 1, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 9, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2026, 3, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 1, 1, tzinfo=UTC)),  # exactly on a boundary
    ]

    result = split_trades(trades, BOUNDARIES)

    all_bucketed = result.train + result.validation + result.test + result.forward
    assert len(all_bucketed) == len(trades)
    assert {id(t) for t in all_bucketed} == {id(t) for t in trades}
    assert len(result.train) == 2
    assert len(result.validation) == 2
    assert len(result.test) == 1
    assert len(result.forward) == 1


def test_split_trades_counts_straddling_trades():
    trades = [
        _trade(
            entry_ts=datetime(2024, 12, 1, tzinfo=UTC), exit_ts=datetime(2025, 2, 1, tzinfo=UTC)
        ),
        _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC), exit_ts=datetime(2025, 4, 1, tzinfo=UTC)),
    ]

    result = split_trades(trades, BOUNDARIES)

    assert result.n_straddling == 1


def test_split_trades_handles_empty_history_without_error():
    """Insufficient/empty-history handling: no trades is not an error condition."""
    result = split_trades([], BOUNDARIES)

    assert result.train == []
    assert result.validation == []
    assert result.test == []
    assert result.forward == []
    assert result.n_straddling == 0


def test_split_trades_handles_history_entirely_before_train_end():
    """All trades landing in one bucket, others legitimately empty -- not an error."""
    trades = [_trade(entry_ts=datetime(2024, 1, 1, tzinfo=UTC))]

    result = split_trades(trades, BOUNDARIES)

    assert len(result.train) == 1
    assert result.validation == [] and result.test == [] and result.forward == []


def test_assign_period_handles_real_sqlite_round_trip_naive_timestamps():
    """End-to-end proof of the documented landmine: entry_timestamp is written tz-aware UTC
    but comes back naive from a fresh query/session. A tz-aware PeriodBoundaries must still
    compare against it without raising TypeError."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    write_session = Session(engine)
    write_session.add(
        ReconstructedTrade(
            trader="0xAAA",
            symbol="BTC/USDC:USDC",
            direction="long",
            entry_timestamp=datetime(2025, 3, 1, tzinfo=UTC),
            entry_price=100.0,
            exit_timestamp=datetime(2025, 3, 1, tzinfo=UTC),
            exit_price=100.0,
            quantity=1.0,
            gross_pnl=10.0,
            fees=0.0,
            net_pnl=10.0,
            holding_time_seconds=3600.0,
            n_fills=2,
            is_truncated_start=False,
            was_liquidated=False,
        )
    )
    write_session.commit()
    write_session.close()

    read_session = Session(engine)
    trade = read_session.query(ReconstructedTrade).one()
    assert trade.entry_timestamp.tzinfo is None  # confirms the landmine actually reproduces

    assert assign_period(trade, BOUNDARIES) == "VALIDATION"  # must not raise TypeError
