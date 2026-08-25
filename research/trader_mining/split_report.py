"""Per-period performance report for a chronologically split wallet -- computes
research.trader_mining.metrics.compute_metrics once per TRAIN/VALIDATION/TEST/FORWARD
bucket (unmodified) and formats the result alongside a distinctly labeled whole-history
reference section. See docs/superpowers/specs/2026-08-25-trader-mining-release-4-design.md.

TRAIN-to-VALIDATION expectancy change is reported as a diagnostic only -- the proposal's own
review-notes correction is explicit that an arbitrary percentage cutoff is unstable at
realistic sample sizes. There is no threshold, no pass/fail verdict, anywhere in this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from research.models import ReconstructedTrade
from research.trader_mining.metrics import WalletMetrics, compute_metrics, format_report
from research.trader_mining.splitting import PERIODS, PeriodBoundaries, split_trades


@dataclass
class PeriodSummary:
    period: str
    start: datetime | None
    end: datetime | None
    n_trades: int
    metrics: WalletMetrics


@dataclass
class SplitReport:
    boundaries: PeriodBoundaries
    periods: list[PeriodSummary]  # always length 4, PERIODS order
    n_straddling: int
    whole_history: WalletMetrics


def compute_split_report(
    trades: list[ReconstructedTrade], boundaries: PeriodBoundaries
) -> SplitReport:
    split = split_trades(trades, boundaries)
    bucketed = {
        "TRAIN": split.train,
        "VALIDATION": split.validation,
        "TEST": split.test,
        "FORWARD": split.forward,
    }
    bounds: dict[str, tuple[datetime | None, datetime | None]] = {
        "TRAIN": (None, boundaries.train_end),
        "VALIDATION": (boundaries.train_end, boundaries.validation_end),
        "TEST": (boundaries.validation_end, boundaries.test_end),
        "FORWARD": (boundaries.test_end, None),
    }
    periods = [
        PeriodSummary(
            period=p,
            start=bounds[p][0],
            end=bounds[p][1],
            n_trades=len(bucketed[p]),
            metrics=compute_metrics(bucketed[p]),
        )
        for p in PERIODS
    ]
    return SplitReport(
        boundaries=boundaries,
        periods=periods,
        n_straddling=split.n_straddling,
        whole_history=compute_metrics(trades),
    )


def format_split_report(report: SplitReport, trader: str) -> str:
    lines = [
        f"# Chronological Split Report: {trader}",
        "",
        (
            f"Boundaries: TRAIN < {report.boundaries.train_end.date()} <= VALIDATION < "
            f"{report.boundaries.validation_end.date()} <= TEST < "
            f"{report.boundaries.test_end.date()} <= FORWARD"
        ),
        f"Trades spanning a period boundary (counted in their entry period): {report.n_straddling}",
        "",
    ]
    for summary in report.periods:
        start = summary.start.date() if summary.start else "(start of history)"
        end = summary.end.date() if summary.end else "(ongoing)"
        lines.append(f"## {summary.period} [{start} - {end}), n={summary.n_trades}")
        lines.append("")
        lines.append(format_report(summary.metrics, trader))
        lines.append("")

    train_expectancy = report.periods[0].metrics.expectancy
    validation_expectancy = report.periods[1].metrics.expectancy
    if train_expectancy and validation_expectancy is not None:
        delta_pct = (validation_expectancy - train_expectancy) / abs(train_expectancy) * 100
        lines.append(
            f"Diagnostic: TRAIN->VALIDATION expectancy changed by {delta_pct:.1f}% "
            "(reported for awareness only -- no threshold, no automatic verdict)"
        )
    else:
        lines.append(
            "Diagnostic: TRAIN->VALIDATION expectancy change: n/a (insufficient data in "
            "TRAIN or VALIDATION)"
        )
    lines.append("")

    lines.append("## Whole-history (reference only -- NOT out-of-sample)")
    lines.append("")
    lines.append(format_report(report.whole_history, trader))

    return "\n".join(lines)
