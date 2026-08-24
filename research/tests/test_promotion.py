# research/tests/test_promotion.py
from datetime import UTC, datetime

import pytest

from research.db import get_engine, get_session
from research.models import CandidateResult
from research.promotion import (
    PromotionState,
    create_promotion_record,
    promote_to_live,
    reject,
    start_paper_trading,
)


def _session(tmp_path):
    engine = get_engine(str(tmp_path / "research.sqlite"))
    return get_session(engine)


def _candidate(session, survived=True, oos_sharpe=1.5):
    candidate = CandidateResult(
        run_stamp=datetime.now(UTC),
        strategy_id="TestStrategy",
        strategy_family="TestStrategy",
        params_json="{}",
        universe="BTC/USDT",
        timeframe="1h",
        discovery_start="2024-01-01",
        discovery_end="2024-06-01",
        n_trials_this_run=1,
        is_sharpe=1.0,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        survived=survived,
        evidence_json="{}",
    )
    session.add(candidate)
    session.flush()
    return candidate


def test_create_promotion_record_succeeds_for_a_passing_candidate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, survived=True)

    record = create_promotion_record(session, candidate.id)

    assert record.state == PromotionState.PASSED_GATE.value
    assert record.candidate_result_id == candidate.id


def test_create_promotion_record_raises_for_a_failing_candidate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, survived=False)

    with pytest.raises(ValueError, match="did not pass the gate"):
        create_promotion_record(session, candidate.id)


def test_create_promotion_record_raises_for_a_nonexistent_candidate(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="No CandidateResult"):
        create_promotion_record(session, 999)


def test_start_paper_trading_transitions_passed_gate_to_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    before = datetime.now(UTC)

    updated = start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    assert updated.state == PromotionState.PAPER_TRADING.value
    assert updated.paper_trading_db_path == "tradesv3.dryrun.sqlite"
    assert updated.paper_trading_started_at.replace(tzinfo=UTC) >= before


def test_start_paper_trading_raises_when_not_in_passed_gate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    with pytest.raises(ValueError, match="cannot start paper trading"):
        start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")


def test_promote_to_live_transitions_live_eligible_to_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")
    record.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()

    updated = promote_to_live(session, record.id)

    assert updated.state == PromotionState.LIVE.value
    assert updated.resolved_at is not None


def test_promote_to_live_raises_from_paper_trading_directly(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    with pytest.raises(ValueError, match="cannot promote to live"):
        promote_to_live(session, record.id)


def test_reject_transitions_paper_trading_and_live_eligible_to_rejected(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)

    record_a = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record_a.id, "a.sqlite")
    rejected_a = reject(session, record_a.id, "degraded in paper trading")
    assert rejected_a.state == PromotionState.REJECTED.value
    assert rejected_a.resolution_reason == "degraded in paper trading"

    record_b = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record_b.id, "b.sqlite")
    record_b.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()
    rejected_b = reject(session, record_b.id, "manual override")
    assert rejected_b.state == PromotionState.REJECTED.value


def test_reject_raises_from_passed_gate_and_already_resolved_states(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match=r"reject\(\) only applies"):
        reject(session, record.id, "too early")

    start_paper_trading(session, record.id, "a.sqlite")
    reject(session, record.id, "first rejection")
    with pytest.raises(ValueError, match=r"reject\(\) only applies"):
        reject(session, record.id, "second rejection")


def test_transitions_raise_from_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "a.sqlite")
    record.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()
    promote_to_live(session, record.id)

    with pytest.raises(ValueError):
        start_paper_trading(session, record.id, "b.sqlite")

    with pytest.raises(ValueError):
        promote_to_live(session, record.id)

    with pytest.raises(ValueError):
        reject(session, record.id, "too late")
