from datetime import UTC, datetime

from research.models import ReconstructedTrade
from research.trader_mining.split_report import compute_split_report, format_split_report
from research.trader_mining.splitting import PeriodBoundaries


BOUNDARIES = PeriodBoundaries(
    train_end=datetime(2025, 1, 1, tzinfo=UTC),
    validation_end=datetime(2025, 7, 1, tzinfo=UTC),
    test_end=datetime(2026, 1, 1, tzinfo=UTC),
)


def _trade(net_pnl, entry_ts, exit_ts=None) -> ReconstructedTrade:
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
        gross_pnl=net_pnl,
        fees=0.0,
        net_pnl=net_pnl,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_compute_split_report_populates_all_four_periods_and_whole_history():
    trades = [
        _trade(10.0, datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(20.0, datetime(2024, 8, 1, tzinfo=UTC)),
        _trade(-5.0, datetime(2025, 3, 1, tzinfo=UTC)),
        _trade(15.0, datetime(2025, 9, 1, tzinfo=UTC)),
    ]

    report = compute_split_report(trades, BOUNDARIES)

    assert [p.period for p in report.periods] == ["TRAIN", "VALIDATION", "TEST", "FORWARD"]
    train, validation, test, forward = report.periods
    assert train.n_trades == 2
    assert train.metrics.net_pnl == 30.0
    assert validation.n_trades == 1
    assert validation.metrics.net_pnl == -5.0
    assert test.n_trades == 1
    assert forward.n_trades == 0
    assert forward.metrics.trade_count == 0  # compute_metrics([]) -- reused, not reimplemented
    assert report.whole_history.trade_count == 4
    assert report.whole_history.net_pnl == 40.0
    assert report.n_straddling == 0


def test_compute_split_report_handles_all_periods_empty():
    """Insufficient history: an empty trade list must not raise, and every period's metrics
    must come back as compute_metrics([])'s own well-defined 'undefined' shape."""
    report = compute_split_report([], BOUNDARIES)

    assert all(p.n_trades == 0 for p in report.periods)
    assert all(p.metrics.win_rate is None for p in report.periods)
    assert report.whole_history.trade_count == 0


def test_period_summary_start_end_are_open_for_train_start_and_forward_end():
    report = compute_split_report([], BOUNDARIES)

    train, validation, test, forward = report.periods
    assert train.start is None
    assert train.end == BOUNDARIES.train_end
    assert validation.start == BOUNDARIES.train_end
    assert test.end == BOUNDARIES.test_end
    assert forward.start == BOUNDARIES.test_end
    assert forward.end is None


def test_format_split_report_labels_periods_and_shows_sample_counts_and_dates():
    trades = [
        _trade(10.0, datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(-5.0, datetime(2025, 3, 1, tzinfo=UTC)),
    ]
    report = compute_split_report(trades, BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "TRAIN" in text and "VALIDATION" in text and "TEST" in text and "FORWARD" in text
    assert "n=1" in text  # each populated period's sample count appears
    assert "2025-01-01" in text  # a boundary date appears
    assert "0xAAA" in text


def test_format_split_report_shows_whole_history_as_a_distinct_labeled_section():
    trades = [_trade(10.0, datetime(2024, 6, 1, tzinfo=UTC))]
    report = compute_split_report(trades, BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "whole-history" in text.lower()
    assert "not out-of-sample" in text.lower()


def test_format_split_report_never_auto_rejects_on_expectancy_degradation():
    """Proposal review-notes correction: report TRAIN->VALIDATION expectancy degradation as
    a diagnostic only. No pass/fail verdict, no threshold language."""
    trades = [
        _trade(100.0, datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(-90.0, datetime(2025, 3, 1, tzinfo=UTC)),  # huge TRAIN->VALIDATION degradation
    ]

    report = compute_split_report(trades, BOUNDARIES)
    text = format_split_report(report, "0xAAA")

    lowered = text.lower()
    assert "reject" not in lowered
    assert "fail" not in lowered
    assert "diagnostic" in lowered


def test_format_split_report_handles_empty_periods_without_crashing_or_printing_none():
    report = compute_split_report([], BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "None" not in text
    assert "n/a" in text
