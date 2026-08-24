# research/tests/test_promotion.py
from datetime import UTC, datetime, timedelta

import pytest

from freqtrade.persistence import Trade, init_db
from research.db import get_engine, get_session
from research.models import CandidateResult
from research.promotion import (
    PromotionState,
    apply_health_evaluation,
    create_promotion_record,
    evaluate_paper_trading_health,
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


@pytest.fixture(autouse=True)
def _reset_trade_session_after_evaluation_tests():
    """evaluate_paper_trading_health() calls freqtrade's own init_db(), which sets
    Trade.session as GLOBAL class-level state (see the plan's Global Constraints for
    why) -- reset it to a fresh in-memory DB after every test in this file so no later
    test (in this file or elsewhere in the same pytest-xdist worker) can see
    Trade.session still pointed at one of this file's throwaway dry-run databases."""
    yield
    init_db("sqlite://")


def _insert_dry_run_trades(dry_run_db_path, strategy_id, started_at, profits_abs):
    """Directly construct and insert closed Trade rows into a fresh dry-run-style
    sqlite database -- one trade per entry in profits_abs, spaced evenly across the
    period from started_at to now, matching how research/tests/test_regime.py and
    test_scoring.py construct real rows directly rather than running a full backtest."""
    init_db(f"sqlite:///{dry_run_db_path}")
    now = datetime.now(UTC).replace(tzinfo=None)
    started_naive = started_at.replace(tzinfo=None) if started_at.tzinfo else started_at
    span = now - started_naive
    step = span / max(1, len(profits_abs))
    for i, profit in enumerate(profits_abs):
        open_dt = started_naive + step * i
        close_dt = open_dt + timedelta(minutes=30)
        trade = Trade(
            pair="BTC/USDT",
            strategy=strategy_id,
            exchange="binance",
            is_open=False,
            open_date=open_dt,
            close_date=close_dt,
            close_profit_abs=float(profit),
            stake_amount=100.0,
            amount=1.0,
            open_rate=100.0,
            close_rate=100.0 + float(profit) / 100.0,
            fee_open=0.001,
            fee_close=0.001,
        )
        Trade.session.add(trade)
    Trade.session.flush()
    Trade.session.commit()
    init_db("sqlite://")  # release this function's own Trade.session redirect immediately


def test_evaluate_paper_trading_health_not_enough_evidence_when_too_few_trades(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_thin.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_thin.sqlite", "TestStrategy", started_at, [8, 6, 7]
    )  # only 3 trades, MIN_PAPER_TRADES is 10

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is False
    assert evaluation["eligible"] is False
    assert evaluation["n_trades"] == 3

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.PAPER_TRADING.value


def test_evaluate_paper_trading_health_eligible_when_degradation_ratio_clears_bar(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_good.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_good.sqlite",
        "TestStrategy",
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["n_trades"] == 10
    assert evaluation["paper_sharpe"] == pytest.approx(67.54628043053148)
    assert evaluation["degradation_ratio"] == pytest.approx(1.0)
    assert evaluation["eligible"] is True

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.LIVE_ELIGIBLE.value
    assert updated.resolved_at is not None


def test_evaluate_paper_trading_health_rejected_when_degradation_ratio_fails_bar(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_bad.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_bad.sqlite",
        "TestStrategy",
        started_at,
        [-3, 2, -4, 1, -2, 3, -5, 2, -3, 1],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["paper_sharpe"] == pytest.approx(-3.9705208944418664)
    assert evaluation["degradation_ratio"] == pytest.approx(0.0)
    assert evaluation["eligible"] is False

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.REJECTED.value
    assert updated.resolution_reason is not None


def test_evaluate_paper_trading_health_zero_trades_does_not_raise(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=1)
    dry_run_path = tmp_path / "dryrun_empty.sqlite"
    start_paper_trading(session, record.id, str(dry_run_path), started_at)
    init_db(f"sqlite:///{dry_run_path}")  # create the (empty) schema, no trades inserted
    init_db("sqlite://")

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["n_trades"] == 0
    assert evaluation["paper_sharpe"] == 0
    assert evaluation["enough_evidence"] is False


def test_evaluate_paper_trading_health_non_positive_oos_sharpe_is_zero_degradation(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=-0.5)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_neg.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_neg.sqlite",
        "TestStrategy",
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == 0.0


def test_evaluate_paper_trading_health_raises_when_not_in_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match="cannot evaluate health"):
        evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)


def test_apply_health_evaluation_raises_when_not_in_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    canned_evaluation = {
        "eligible": True,
        "enough_evidence": True,
        "days_elapsed": 30,
        "n_trades": 20,
        "paper_sharpe": 1.5,
        "degradation_ratio": 0.9,
        "reasons": [],
    }

    with pytest.raises(ValueError, match="cannot apply a health evaluation"):
        apply_health_evaluation(session, record.id, canned_evaluation)


def test_apply_health_evaluation_raises_when_eligible_without_enough_evidence(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")
    inconsistent_evaluation = {
        "eligible": True,
        "enough_evidence": False,
        "days_elapsed": 3,
        "n_trades": 1,
        "paper_sharpe": 1.5,
        "degradation_ratio": 0.9,
        "reasons": [],
    }

    with pytest.raises(ValueError, match="internally inconsistent"):
        apply_health_evaluation(session, record.id, inconsistent_evaluation)


def test_full_promotion_chain_from_passed_gate_through_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    assert record.state == PromotionState.PASSED_GATE.value

    started_at = datetime.now(UTC) - timedelta(days=14)
    dry_run_db_path = tmp_path / "dryrun_full_chain.sqlite"
    start_paper_trading(session, record.id, str(dry_run_db_path), started_at)
    assert record.state == PromotionState.PAPER_TRADING.value

    _insert_dry_run_trades(
        dry_run_db_path,
        "TestStrategy",
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.LIVE_ELIGIBLE.value

    live = promote_to_live(session, record.id)
    assert live.state == PromotionState.LIVE.value
    assert live.resolved_at is not None
