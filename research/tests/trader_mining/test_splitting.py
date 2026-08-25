from datetime import UTC, datetime, timedelta, timezone

import pytest

from research.trader_mining.splitting import PeriodBoundaries, _to_naive_utc


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
