import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from research.models import CandidateResult


# Strategy-name -> family aliasing. Grow this table as strategy variants are added —
# related parameter sweeps must share a family so trial counts compound correctly.
_FAMILY_ALIASES: dict[str, str] = {
    "ema_cross_v3": "trend_following",
}


def family_of(strategy_id: str) -> str:
    return _FAMILY_ALIASES.get(strategy_id, strategy_id)


def log_candidate_result(
    session: Session,
    *,
    strategy_id: str,
    params: dict,
    universe: str,
    timeframe: str,
    discovery_start: str,
    discovery_end: str,
    n_trials_this_run: int,
    is_sharpe: float,
    oos_sharpe: float,
    deflated_sharpe: float,
    permutation_p: float,
    pbo: float,
    survived: bool,
    validation_start: str | None = None,
    validation_end: str | None = None,
    oos_start: str | None = None,
    oos_end: str | None = None,
    evidence: dict | None = None,
    run_stamp: datetime | None = None,
) -> CandidateResult:
    row = CandidateResult(
        run_stamp=run_stamp or datetime.now(UTC),
        strategy_id=strategy_id,
        strategy_family=family_of(strategy_id),
        params_json=json.dumps(params, sort_keys=True),
        universe=universe,
        timeframe=timeframe,
        discovery_start=discovery_start,
        discovery_end=discovery_end,
        validation_start=validation_start,
        validation_end=validation_end,
        oos_start=oos_start,
        oos_end=oos_end,
        n_trials_this_run=n_trials_this_run,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=deflated_sharpe,
        permutation_p=permutation_p,
        pbo=pbo,
        survived=survived,
        evidence_json=json.dumps(evidence or {}),
    )
    session.add(row)
    session.flush()
    return row


def family_trial_count(session: Session, family: str, declared: int = 0) -> int:
    """The number of trials to deflate against: whichever is larger between the
    ledger's accumulated row count for this family and a caller-declared count.
    Call this BEFORE writing the current run's own row (count-then-write) so a run
    never deflates against trials it hasn't finished yet."""
    ledger_count = (
        session.query(CandidateResult).filter(CandidateResult.strategy_family == family).count()
    )
    return max(ledger_count, declared)
