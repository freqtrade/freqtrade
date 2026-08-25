# research/health.py
"""Live strategy health monitor: repeatedly evaluates a PromotionRecord already in LIVE
state against its own real trading history, classifying it into a HEALTHY -> WATCH ->
DEGRADED -> SUSPENDED ladder. Never stops a live bot -- SUSPENDED is a recorded
recommendation for a human, exactly as LIVE itself is reachable only by a manual human
call in research/promotion.py. See
docs/superpowers/specs/2026-08-24-live-strategy-health-design.md for the full design.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from freqtrade.data.metrics import calculate_sharpe
from freqtrade.persistence import Trade, init_db
from research.models import CandidateResult, HealthCheck
from research.promotion import PromotionState, _load_promotion_record


class HealthState(StrEnum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"


_STATE_ORDER = [
    HealthState.HEALTHY.value,
    HealthState.WATCH.value,
    HealthState.DEGRADED.value,
    HealthState.SUSPENDED.value,
]

# ponytail: starting defaults, not derived from any real live-trading history (none
# exists yet in this fork) -- adjust based on real usage once this runs against real
# strategies. Deliberately mirrors promotion.py's MIN_PAPER_TRADING_DAYS/MIN_PAPER_TRADES
# shape, applied to a rolling window instead of a one-time cumulative check.
HEALTH_WINDOW_DAYS = 30
MIN_HEALTH_TRADES = 10
HEALTHY_THRESHOLD = 0.7
WATCH_THRESHOLD = 0.4
DEGRADED_THRESHOLD = 0.15
MIN_HEALTH_CHECK_INTERVAL_HOURS = 24


def evaluate_live_health(
    session: Session,
    promotion_id: int,
    starting_balance: float,
    live_db_path: str | None = None,
) -> dict:
    """Pure evaluation (no state mutation, no HealthCheck row written) of a LIVE
    PromotionRecord's real trade history over the trailing HEALTH_WINDOW_DAYS. Returns a
    verdict dict; call record_health_check with the result to persist an audit row and
    (possibly) move the record's current health state.

    `live_db_path` defaults to the record's own stored paper_trading_db_path -- the same
    field promotion.py uses to remember where a record's trade history lives (its name
    predates LIVE and is not renamed here; it holds a LIVE record's real trading
    database just as well as a PAPER_TRADING one's dry-run database).

    starting_balance is required, matching evaluate_paper_trading_health's own
    established contract -- there is no other way for this function to discover the
    live bot's configured wallet size.

    IMPORTANT: freqtrade.persistence.init_db() sets Trade.session as GLOBAL class-level
    state, not a scoped per-call connection. This function fully materializes its query
    results before returning and disposes the engine/session in a finally block before
    returning -- see FIELD-NOTES.md and research/promotion.py's own equivalent pattern.

    Rolling window, not cumulative: uses only closed trades with
    close_date >= now - HEALTH_WINDOW_DAYS, not the strategy's full LIVE-to-date
    history. Win rate and max drawdown are computed from the same window and returned
    for human context -- they do not affect target_state.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE.value!r} -- cannot evaluate live health."
        )
    candidate = session.get(CandidateResult, record.candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {record.candidate_result_id}")

    db_path = live_db_path or record.paper_trading_db_path
    if not db_path:
        raise ValueError(
            f"PromotionRecord {promotion_id} has no db path to evaluate -- pass "
            "live_db_path explicitly or ensure the record's stored path is set."
        )

    now = datetime.now(UTC)
    window_start_aware = now - timedelta(days=HEALTH_WINDOW_DAYS)
    window_start_naive = window_start_aware.replace(tzinfo=None)

    init_db(f"sqlite:///{db_path}")
    try:
        closed_trades = (
            Trade.session.query(Trade)
            .filter(
                Trade.strategy == candidate.strategy_id,
                Trade.is_open.is_(False),
                Trade.close_date >= window_start_naive,
            )
            .all()
        )
    finally:
        # Connection hygiene: the live database file belongs to a currently-running
        # freqtrade bot process writing to it concurrently -- never leave a lingering
        # handle on it. Release the scoped session and dispose its engine before
        # returning, regardless of whether the query above succeeded. init_db() always
        # binds Trade.session via sessionmaker(bind=engine) with a real Engine (see
        # freqtrade/persistence/models.py's init_db) -- get_bind()'s broader
        # Engine | Connection return type is never a Connection here.
        engine = cast(Engine, Trade.session.get_bind())
        Trade.session.remove()
        engine.dispose()

    n_trades = len(closed_trades)

    if n_trades > 0:
        trades_df = pd.DataFrame(
            {"profit_abs": [cast(float, t.close_profit_abs) for t in closed_trades]}
        )
        live_sharpe = calculate_sharpe(trades_df, window_start_aware, now, starting_balance)
        win_rate = sum(1 for t in closed_trades if cast(float, t.close_profit_abs) > 0) / n_trades
        sorted_trades = sorted(closed_trades, key=lambda t: cast(datetime, t.close_date))
        equity = starting_balance
        peak = starting_balance
        max_drawdown = 0.0
        for t in sorted_trades:
            equity += cast(float, t.close_profit_abs)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
    else:
        live_sharpe = 0
        win_rate = 0.0
        max_drawdown = 0.0

    if candidate.oos_sharpe > 0:
        degradation_ratio = max(0.0, min(1.0, live_sharpe / candidate.oos_sharpe))
    else:
        degradation_ratio = 0.0

    reasons: list[str] = []
    enough_evidence = n_trades >= MIN_HEALTH_TRADES
    if not enough_evidence:
        reasons.append(
            f"only {n_trades} trades in the last {HEALTH_WINDOW_DAYS} days, need >= "
            f"{MIN_HEALTH_TRADES}"
        )
        target_state = None
    elif degradation_ratio >= HEALTHY_THRESHOLD:
        target_state = HealthState.HEALTHY.value
    elif degradation_ratio >= WATCH_THRESHOLD:
        target_state = HealthState.WATCH.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below healthy threshold {HEALTHY_THRESHOLD}"
        )
    elif degradation_ratio >= DEGRADED_THRESHOLD:
        target_state = HealthState.DEGRADED.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below watch threshold {WATCH_THRESHOLD}"
        )
    else:
        target_state = HealthState.SUSPENDED.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below degraded threshold "
            f"{DEGRADED_THRESHOLD}"
        )

    return {
        "enough_evidence": enough_evidence,
        "n_trades": n_trades,
        "live_sharpe": live_sharpe,
        "degradation_ratio": degradation_ratio,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "target_state": target_state,
        "reasons": reasons,
    }


def record_health_check(
    session: Session,
    promotion_id: int,
    evaluation: dict,
    evaluated_at: datetime | None = None,
) -> HealthCheck:
    """Apply an evaluate_live_health() result: write one HealthCheck audit row, computing
    the record's new current state from its previous current state (the latest existing
    HealthCheck row for this promotion_id, or implicit HEALTHY if none exists yet).

    A rung-move requires BOTH evaluation["enough_evidence"] is True AND at least
    MIN_HEALTH_CHECK_INTERVAL_HOURS real hours elapsed since the most recent prior
    HealthCheck row (no prior row satisfies this automatically). Failing either gate
    means the new state equals the previous state exactly -- an audit row is still
    written either way, with a reason explaining why no move happened.

    Raises ValueError if the record doesn't exist or isn't currently LIVE (re-checked
    directly, independent of evaluate_live_health's own guard, since evaluation may be
    stale by the time this is called).
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE.value!r} -- cannot record a health check."
        )

    check_time = evaluated_at or datetime.now(UTC)
    if check_time.tzinfo is None:
        check_time = check_time.replace(tzinfo=UTC)

    prior = (
        session.query(HealthCheck)
        .filter(HealthCheck.promotion_record_id == promotion_id)
        .order_by(HealthCheck.evaluated_at.desc(), HealthCheck.id.desc())
        .first()
    )
    previous_state = prior.state if prior is not None else HealthState.HEALTHY.value

    reasons = list(evaluation["reasons"])
    if not evaluation["enough_evidence"]:
        new_state = previous_state
    else:
        interval_ok = True
        if prior is not None:
            prior_at = prior.evaluated_at
            if prior_at.tzinfo is None:
                prior_at = prior_at.replace(tzinfo=UTC)
            elapsed_hours = (check_time - prior_at).total_seconds() / 3600.0
            if elapsed_hours < MIN_HEALTH_CHECK_INTERVAL_HOURS:
                interval_ok = False
                reasons.append(
                    f"only {elapsed_hours:.1f}h since the last recorded check, need >= "
                    f"{MIN_HEALTH_CHECK_INTERVAL_HOURS}h before a state move"
                )
        if not interval_ok:
            new_state = previous_state
        else:
            cur_idx = _STATE_ORDER.index(previous_state)
            tgt_idx = _STATE_ORDER.index(evaluation["target_state"])
            if tgt_idx > cur_idx:
                new_idx = cur_idx + 1
            elif tgt_idx < cur_idx:
                new_idx = cur_idx - 1
            else:
                new_idx = cur_idx
            new_state = _STATE_ORDER[new_idx]

    check = HealthCheck(
        promotion_record_id=promotion_id,
        evaluated_at=check_time,
        state=new_state,
        enough_evidence=evaluation["enough_evidence"],
        n_trades=evaluation["n_trades"],
        live_sharpe=evaluation["live_sharpe"],
        degradation_ratio=evaluation["degradation_ratio"],
        win_rate=evaluation["win_rate"],
        max_drawdown=evaluation["max_drawdown"],
        reasons_json=json.dumps(reasons),
    )
    session.add(check)
    session.flush()
    return check
