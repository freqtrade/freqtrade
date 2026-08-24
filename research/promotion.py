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

from sqlalchemy.orm import Session

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
