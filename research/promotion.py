# research/promotion.py
"""Paper-trading promotion tracker: a state machine tracking a gate-passing candidate's
lifecycle from PASSED_GATE through PAPER_TRADING to a LIVE_ELIGIBLE recommendation or a
REJECTED verdict. LIVE is reachable only via a direct, manual promote_to_live() call --
never from any automated evaluation path (see evaluate_paper_trading_health /
apply_health_evaluation, added in Task 2 of this module's plan). See
docs/superpowers/specs/2026-08-24-paper-trading-promotion-design.md for the full design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

import pandas as pd
from sqlalchemy.orm import Session

from freqtrade.data.metrics import calculate_sharpe
from freqtrade.persistence import Trade, init_db
from research.models import CandidateResult, PromotionRecord


class PromotionState(StrEnum):
    PASSED_GATE = "passed_gate"
    PAPER_TRADING = "paper_trading"
    LIVE_ELIGIBLE = "live_eligible"
    LIVE = "live"
    REJECTED = "rejected"


def _load_promotion_record(session: Session, promotion_id: int) -> PromotionRecord:
    record = session.get(PromotionRecord, promotion_id)
    if record is None:
        raise ValueError(f"No PromotionRecord with id {promotion_id}")
    return record


def create_promotion_record(session: Session, candidate_result_id: int) -> PromotionRecord:
    """Start a new promotion lifecycle for a candidate that already passed the gate.

    Raises ValueError if candidate_result_id doesn't exist, or if that candidate's
    CandidateResult.survived is not True -- only a passing gate result may be promoted.
    """
    candidate = session.get(CandidateResult, candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {candidate_result_id}")
    if not candidate.survived:
        raise ValueError(
            f"CandidateResult {candidate_result_id} did not pass the gate "
            "(survived=False) -- only a passing candidate may be promoted."
        )
    record = PromotionRecord(
        candidate_result_id=candidate_result_id,
        state=PromotionState.PASSED_GATE.value,
        created_at=datetime.now(UTC),
    )
    session.add(record)
    session.flush()
    return record


def start_paper_trading(
    session: Session,
    promotion_id: int,
    dry_run_db_path: str,
    started_at: datetime | None = None,
) -> PromotionRecord:
    """Transition PASSED_GATE -> PAPER_TRADING.

    Raises ValueError if the record doesn't exist or isn't currently PASSED_GATE.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PASSED_GATE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PASSED_GATE.value!r} -- cannot start paper trading."
        )
    record.state = PromotionState.PAPER_TRADING.value
    record.paper_trading_db_path = dry_run_db_path
    record.paper_trading_started_at = started_at or datetime.now(UTC)
    session.flush()
    return record


def promote_to_live(session: Session, promotion_id: int) -> PromotionRecord:
    """Transition LIVE_ELIGIBLE -> LIVE. The only function in this module that can
    produce a LIVE state -- called only by a human, directly, never from
    apply_health_evaluation or any other automated path.

    Raises ValueError if the record doesn't exist or isn't currently LIVE_ELIGIBLE.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE_ELIGIBLE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE_ELIGIBLE.value!r} -- cannot promote to live."
        )
    record.state = PromotionState.LIVE.value
    record.resolved_at = datetime.now(UTC)
    session.flush()
    return record


def reject(session: Session, promotion_id: int, reason: str) -> PromotionRecord:
    """Manually transition PAPER_TRADING or LIVE_ELIGIBLE -> REJECTED.

    Raises ValueError if the record doesn't exist, is currently PASSED_GATE (paper
    trading never started), or is already resolved (REJECTED/LIVE).
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state not in (
        PromotionState.PAPER_TRADING.value,
        PromotionState.LIVE_ELIGIBLE.value,
    ):
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r} -- "
            f"reject() only applies from {PromotionState.PAPER_TRADING.value!r} or "
            f"{PromotionState.LIVE_ELIGIBLE.value!r}."
        )
    record.state = PromotionState.REJECTED.value
    record.resolved_at = datetime.now(UTC)
    record.resolution_reason = reason
    session.flush()
    return record


# ponytail: starting defaults, not derived from any real paper-trading history (none
# exists yet in this fork) -- adjust based on real usage once this runs against real
# strategies.
MIN_PAPER_TRADING_DAYS = 14
MIN_PAPER_TRADES = 10
MIN_DEGRADATION_RATIO = 0.5


def evaluate_paper_trading_health(
    session: Session,
    promotion_id: int,
    starting_balance: float,
    dry_run_db_path: str | None = None,
    periods_per_year: int = 365,
) -> dict:
    """Pure evaluation (no state mutation) of a PAPER_TRADING record's real dry-run
    trade history. Returns a verdict dict; call apply_health_evaluation with the result
    to actually transition state.

    `starting_balance` is required -- the paper-trading bot's own configured wallet
    size, which this function has no other way to discover (see the spec for why
    run_promotion_gate's config isn't available here).

    IMPORTANT: freqtrade.persistence.init_db() sets Trade.session as GLOBAL class-level
    state, not a scoped per-call connection. This function fully materializes its query
    results before returning and must never be called concurrently with, or interleaved
    with, other in-process code relying on Trade.session pointing at a different
    database (e.g. a WalkForwardRunner/Backtesting run in the same process).

    Known limitation: the returned degradation_ratio is a coarse heuristic, not a
    statistically rigorous comparison -- a paper-trading window is typically far
    shorter than the OOS window a candidate was originally evaluated over, so
    paper_sharpe carries materially more estimation noise than the OOS baseline it's
    compared against. Treat this as a first-pass filter for human judgment, not proof.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PAPER_TRADING.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PAPER_TRADING.value!r} -- cannot evaluate health."
        )
    candidate = session.get(CandidateResult, record.candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {record.candidate_result_id}")

    db_path = dry_run_db_path or record.paper_trading_db_path

    now = datetime.now(UTC)
    started_at_aware = record.paper_trading_started_at
    if started_at_aware is None:
        # Unreachable in practice: start_paper_trading() always sets this before a
        # record can reach PAPER_TRADING state. Narrows the nullable-in-schema type
        # for mypy and fails loudly rather than silently if that invariant ever breaks.
        raise ValueError(
            f"PromotionRecord {promotion_id} is PAPER_TRADING but has no "
            "paper_trading_started_at -- this should be impossible."
        )
    if started_at_aware.tzinfo is None:
        started_at_aware = started_at_aware.replace(tzinfo=UTC)
    started_at_naive = started_at_aware.replace(tzinfo=None)

    days_elapsed = (now - started_at_aware).days

    init_db(f"sqlite:///{db_path}")
    closed_trades = (
        Trade.session.query(Trade)
        .filter(
            Trade.strategy == candidate.strategy_id,
            Trade.is_open.is_(False),
            Trade.close_date >= started_at_naive,
        )
        .all()
    )
    n_trades = len(closed_trades)

    if n_trades > 0:
        trades_df = pd.DataFrame({"profit_abs": [t.close_profit_abs for t in closed_trades]})
        paper_sharpe = calculate_sharpe(trades_df, started_at_aware, now, starting_balance)
    else:
        paper_sharpe = 0

    if candidate.oos_sharpe > 0:
        degradation_ratio = max(0.0, min(1.0, paper_sharpe / candidate.oos_sharpe))
    else:
        degradation_ratio = 0.0

    reasons: list[str] = []
    enough_evidence = days_elapsed >= MIN_PAPER_TRADING_DAYS and n_trades >= MIN_PAPER_TRADES
    if not enough_evidence:
        eligible = False
        if days_elapsed < MIN_PAPER_TRADING_DAYS:
            reasons.append(f"only {days_elapsed} days elapsed, need >= {MIN_PAPER_TRADING_DAYS}")
        if n_trades < MIN_PAPER_TRADES:
            reasons.append(f"only {n_trades} trades, need >= {MIN_PAPER_TRADES}")
    elif degradation_ratio < MIN_DEGRADATION_RATIO:
        eligible = False
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below threshold {MIN_DEGRADATION_RATIO}"
        )
    else:
        eligible = True

    return {
        "eligible": eligible,
        "enough_evidence": enough_evidence,
        "days_elapsed": days_elapsed,
        "n_trades": n_trades,
        "paper_sharpe": paper_sharpe,
        "degradation_ratio": degradation_ratio,
        "reasons": reasons,
    }


def apply_health_evaluation(
    session: Session, promotion_id: int, evaluation: dict
) -> PromotionRecord:
    """Apply an evaluate_paper_trading_health() result to the state machine.

    PAPER_TRADING -> LIVE_ELIGIBLE if eligible; PAPER_TRADING -> REJECTED if there's
    enough evidence but it failed the bar; otherwise no state change (stays
    PAPER_TRADING for a future re-evaluation).

    Raises ValueError if the record doesn't exist or isn't currently PAPER_TRADING.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PAPER_TRADING.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PAPER_TRADING.value!r} -- cannot apply a health evaluation."
        )
    if evaluation["eligible"]:
        record.state = PromotionState.LIVE_ELIGIBLE.value
        record.resolved_at = datetime.now(UTC)
        record.resolution_reason = (
            f"paper_sharpe={evaluation['paper_sharpe']:.3f}, "
            f"degradation_ratio={evaluation['degradation_ratio']:.3f}, "
            f"n_trades={evaluation['n_trades']}, days_elapsed={evaluation['days_elapsed']}"
        )
    elif evaluation["enough_evidence"]:
        record.state = PromotionState.REJECTED.value
        record.resolved_at = datetime.now(UTC)
        record.resolution_reason = "; ".join(evaluation["reasons"])
    session.flush()
    return record
