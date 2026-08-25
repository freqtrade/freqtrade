# research/tests/test_health.py
import json
from datetime import UTC, datetime, timedelta

import pytest

from freqtrade.persistence import Trade, init_db
from research.db import get_engine, get_session
from research.health import (
    HEALTH_WINDOW_DAYS,
    MIN_HEALTH_CHECK_INTERVAL_HOURS,
    HealthState,
    evaluate_live_health,
    record_health_check,
)
from research.models import CandidateResult, HealthCheck
from research.promotion import (
    PromotionState,
    apply_health_evaluation,
    create_promotion_record,
    evaluate_paper_trading_health,
    promote_to_live,
    start_paper_trading,
)


def _session(tmp_path):
    engine = get_engine(str(tmp_path / "research.sqlite"))
    return get_session(engine)


def _candidate(session, oos_sharpe=30.0):
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
        survived=True,
        evidence_json="{}",
    )
    session.add(candidate)
    session.flush()
    return candidate


def _live_record(session, tmp_path, oos_sharpe=30.0, db_name="live.sqlite"):
    """Reach a genuine LIVE PromotionRecord via research/promotion.py's own real
    functions -- no manual state injection."""
    candidate = _candidate(session, oos_sharpe=oos_sharpe)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, str(tmp_path / db_name))
    record.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()
    promote_to_live(session, record.id)
    return record, candidate


@pytest.fixture(autouse=True)
def _reset_trade_session_after_health_tests():
    """evaluate_live_health() calls freqtrade's own init_db(), which sets Trade.session
    as GLOBAL class-level state -- reset it to a fresh in-memory DB after every test in
    this file so no later test (in this file or elsewhere in the same pytest-xdist
    worker) can see Trade.session still pointed at one of this file's throwaway
    databases. See FIELD-NOTES.md."""
    yield
    init_db("sqlite://")


def _insert_live_trades(live_db_path, strategy_id, started_at, profits_abs):
    """Directly construct and insert closed Trade rows into a fresh live-style sqlite
    database -- one trade per entry in profits_abs, spaced evenly across the period from
    started_at to now. Mirrors research/tests/test_promotion.py's
    _insert_dry_run_trades helper exactly."""
    init_db(f"sqlite:///{live_db_path}")
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


def test_evaluate_live_health_raises_for_missing_record(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="No PromotionRecord"):
        evaluate_live_health(session, 999, starting_balance=1000.0)


def test_evaluate_live_health_raises_when_not_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, str(tmp_path / "a.sqlite"))

    with pytest.raises(ValueError, match="cannot evaluate live health"):
        evaluate_live_health(session, record.id, starting_balance=1000.0)


def test_evaluate_live_health_not_enough_evidence(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS),
        [8, 6, 7],
    )  # only 3 trades, MIN_HEALTH_TRADES is 10

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is False
    assert evaluation["n_trades"] == 3
    assert evaluation["target_state"] is None
    assert evaluation["win_rate"] == pytest.approx(1.0)  # all 3 profits are positive


def test_evaluate_live_health_excludes_trades_outside_the_window(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    init_db(f"sqlite:///{tmp_path / 'live.sqlite'}")
    now_naive = datetime.now(UTC).replace(tzinfo=None)
    inside = now_naive - timedelta(days=HEALTH_WINDOW_DAYS - 1)
    outside = now_naive - timedelta(days=HEALTH_WINDOW_DAYS + 1)
    for close_dt in (inside, outside):
        Trade.session.add(
            Trade(
                pair="BTC/USDT",
                strategy=candidate.strategy_id,
                exchange="binance",
                is_open=False,
                open_date=close_dt - timedelta(minutes=30),
                close_date=close_dt,
                close_profit_abs=10.0,
                stake_amount=100.0,
                amount=1.0,
                open_rate=100.0,
                close_rate=100.1,
                fee_open=0.001,
                fee_close=0.001,
            )
        )
    Trade.session.flush()
    Trade.session.commit()
    init_db("sqlite://")

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["n_trades"] == 1


def test_evaluate_live_health_healthy_fixture_hand_verified_sharpe(tmp_path):
    session = _session(tmp_path)
    # oos_sharpe=30.0 chosen so degradation_ratio clips to exactly 1.0 (live_sharpe
    # 31.521597534248023 > oos_sharpe 30.0) -- matches HEALTHY_THRESHOLD=0.7 comfortably.
    record, candidate = _live_record(session, tmp_path, oos_sharpe=30.0)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["n_trades"] == 10
    # Hand-derived from freqtrade's real calculate_sharpe formula: total_return=0.07,
    # days_period=30 (HEALTH_WINDOW_DAYS), mean_daily_return=0.07/30, population stdev
    # of the per-trade normalized returns=0.0014142135623730951 (same fixture already
    # hand-verified in test_promotion.py over a 14-day window as 67.54628043053148;
    # Sharpe scales linearly with 1/days_period, so the 30-day value is
    # 67.54628043053148 * 14 / 30).
    assert evaluation["live_sharpe"] == pytest.approx(31.521597534248023)
    assert evaluation["degradation_ratio"] == pytest.approx(1.0)
    assert evaluation["target_state"] == HealthState.HEALTHY.value


def test_evaluate_live_health_degraded_fixture(tmp_path):
    session = _session(tmp_path)
    # Same trades/live_sharpe as the healthy fixture (31.521597534248023), but a much
    # larger oos_sharpe so degradation_ratio = 31.521597534248023 / 100.0 ~= 0.315,
    # landing inside [DEGRADED_THRESHOLD, WATCH_THRESHOLD) = [0.15, 0.4).
    record, candidate = _live_record(session, tmp_path, oos_sharpe=100.0)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == pytest.approx(0.31521597534248023)
    assert evaluation["target_state"] == HealthState.DEGRADED.value


def test_evaluate_live_health_non_positive_oos_sharpe_is_zero_degradation(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path, oos_sharpe=-0.5)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == 0.0
    assert evaluation["target_state"] == HealthState.SUSPENDED.value


def _suspended_target_evaluation(n_trades=10):
    """A canned evaluation dict with target_state=SUSPENDED and enough_evidence=True --
    used by the damping tests below, which test record_health_check's own transition
    logic directly rather than re-deriving a real SUSPENDED-target evaluation from
    scratch each time."""
    return {
        "enough_evidence": True,
        "n_trades": n_trades,
        "live_sharpe": -5.0,
        "degradation_ratio": 0.05,
        "win_rate": 0.2,
        "max_drawdown": 0.5,
        "target_state": HealthState.SUSPENDED.value,
        "reasons": ["degradation_ratio 0.050 below degraded threshold 0.15"],
    }


def _healthy_target_evaluation(n_trades=10):
    return {
        "enough_evidence": True,
        "n_trades": n_trades,
        "live_sharpe": 40.0,
        "degradation_ratio": 1.0,
        "win_rate": 0.8,
        "max_drawdown": 0.05,
        "target_state": HealthState.HEALTHY.value,
        "reasons": [],
    }


def _not_enough_evidence_evaluation():
    return {
        "enough_evidence": False,
        "n_trades": 2,
        "live_sharpe": -999.0,  # deliberately alarming -- must be ignored
        "degradation_ratio": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "target_state": None,
        "reasons": ["only 2 trades in the last 30 days, need >= 10"],
    }


def test_record_health_check_no_prior_row_healthy_target_stays_healthy(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)

    check = record_health_check(session, record.id, _healthy_target_evaluation())

    assert check.state == HealthState.HEALTHY.value
    assert check.promotion_record_id == record.id


def test_record_health_check_no_prior_row_suspended_target_moves_one_rung(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)

    check = record_health_check(session, record.id, _suspended_target_evaluation())

    assert check.state == HealthState.WATCH.value  # not SUSPENDED -- one rung from HEALTHY


def test_record_health_check_sustained_degradation_spaced_apart_reaches_suspended(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)

    c1 = record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    c2 = record_health_check(session, record.id, _suspended_target_evaluation(), t0 + interval)
    c3 = record_health_check(session, record.id, _suspended_target_evaluation(), t0 + interval * 2)

    assert c1.state == HealthState.WATCH.value
    assert c2.state == HealthState.DEGRADED.value
    assert c3.state == HealthState.SUSPENDED.value


def test_record_health_check_rapid_calls_do_not_move_state(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    minutes_apart = timedelta(minutes=5)

    c1 = record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    c2 = record_health_check(session, record.id, _suspended_target_evaluation(), t0 + minutes_apart)
    c3 = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + minutes_apart * 2
    )

    assert c1.state == HealthState.WATCH.value  # first call always allowed (no prior row)
    assert c2.state == HealthState.WATCH.value  # interval gate blocks the second move
    assert c3.state == HealthState.WATCH.value  # and the third
    assert any("since the last recorded check" in r for r in json.loads(c2.reasons_json))


def test_record_health_check_recovery_moves_one_rung_at_a_time(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0 + interval)
    suspended_check = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + interval * 2
    )
    assert suspended_check.state == HealthState.SUSPENDED.value

    recovering = record_health_check(
        session, record.id, _healthy_target_evaluation(), t0 + interval * 3
    )

    assert recovering.state == HealthState.DEGRADED.value  # not straight back to HEALTHY


def test_record_health_check_not_enough_evidence_never_moves_state(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0)  # -> WATCH

    check = record_health_check(
        session, record.id, _not_enough_evidence_evaluation(), t0 + interval
    )

    assert check.state == HealthState.WATCH.value  # unchanged despite the alarming dict


def test_record_health_check_raises_when_not_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match="cannot record a health check"):
        record_health_check(session, record.id, _healthy_target_evaluation())


def test_win_rate_and_max_drawdown_hand_verified(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    started_at = datetime.now(UTC) - timedelta(days=2)
    _insert_live_trades(
        tmp_path / "live.sqlite", candidate.strategy_id, started_at, [100, -50, 80, -120, 60]
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    # win_rate: 3 of 5 trades positive (100, 80, 60) -> 0.6
    assert evaluation["win_rate"] == pytest.approx(0.6)
    # max_drawdown: equity 1000 -> 1100 -> 1050 -> 1130 -> 1010 -> 1070 (peak 1130 after
    # trade 3, low point 1010 after trade 4) -> (1130-1010)/1130
    assert evaluation["max_drawdown"] == pytest.approx(0.10619469026548672)


def test_full_promotion_and_health_chain_reaches_suspended(tmp_path):
    """Reuses research/promotion.py's own real functions to reach a genuine LIVE
    record (no new functions involved in this prefix), then exercises this task's two
    new functions against real inserted live trades."""
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    paper_started = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "paper.sqlite"), paper_started)
    _insert_live_trades(
        tmp_path / "paper.sqlite",
        candidate.strategy_id,
        paper_started,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )
    paper_eval = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)
    apply_health_evaluation(session, record.id, paper_eval)
    assert record.state == PromotionState.LIVE_ELIGIBLE.value
    promote_to_live(session, record.id)
    assert record.state == PromotionState.LIVE.value

    live_started = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        live_started,
        [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],  # sustained losses, well below oos_sharpe
    )
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    live_db = str(tmp_path / "live.sqlite")
    c1 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=live_db),
        t0,
    )
    c2 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=live_db),
        t0 + interval,
    )
    c3 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=live_db),
        t0 + interval * 2,
    )

    assert [c1.state, c2.state, c3.state] == [
        HealthState.WATCH.value,
        HealthState.DEGRADED.value,
        HealthState.SUSPENDED.value,
    ]
    # "current state" is queryable as the latest row for this promotion_id
    latest = (
        session.query(HealthCheck)
        .filter(HealthCheck.promotion_record_id == record.id)
        .order_by(HealthCheck.evaluated_at.desc(), HealthCheck.id.desc())
        .first()
    )
    assert latest.id == c3.id
