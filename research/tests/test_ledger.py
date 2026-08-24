"""Test suite for research ledger module."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.ledger import family_of, family_trial_count, log_candidate_result
from research.models import Base, CandidateResult


def _memory_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_log_candidate_result_round_trips() -> None:
    """Test that log_candidate_result stores and retrieves data correctly."""
    session = _memory_session()
    row = log_candidate_result(
        session,
        strategy_id="ema_cross_v3",
        params={"buy_rsi": 30},
        universe="BTC/USDT",
        timeframe="1h",
        discovery_start="2020-01-01",
        discovery_end="2024-12-31",
        n_trials_this_run=48,
        is_sharpe=1.2,
        oos_sharpe=0.8,
        deflated_sharpe=0.6,
        permutation_p=0.03,
        pbo=0.2,
        survived=True,
    )
    session.commit()

    fetched = session.query(CandidateResult).filter_by(id=row.id).one()
    assert fetched.strategy_id == "ema_cross_v3"
    assert fetched.strategy_family == "trend_following"
    assert fetched.params_json == '{"buy_rsi": 30}'
    assert fetched.survived is True


def test_family_of_maps_known_alias_and_falls_back_to_strategy_id() -> None:
    """Test family_of mapping and fallback behavior."""
    assert family_of("ema_cross_v3") == "trend_following"
    assert family_of("some_unmapped_strategy") == "some_unmapped_strategy"


def test_family_trial_count_prefers_ledger_count_when_higher_than_declared() -> None:
    """Test that family_trial_count uses the larger of ledger and declared counts."""
    session = _memory_session()
    for i in range(5):
        log_candidate_result(
            session,
            strategy_id="ema_cross_v3",
            params={"buy_rsi": i},
            universe="BTC/USDT",
            timeframe="1h",
            discovery_start="2020-01-01",
            discovery_end="2024-12-31",
            n_trials_this_run=1,
            is_sharpe=0.1,
            oos_sharpe=0.0,
            deflated_sharpe=0.0,
            permutation_p=1.0,
            pbo=1.0,
            survived=False,
        )
    session.commit()

    assert family_trial_count(session, "trend_following") == 5
    assert family_trial_count(session, "trend_following", declared=2) == 5
    assert family_trial_count(session, "trend_following", declared=10) == 10


def test_get_engine_creates_sqlite_file_and_tables(tmp_path: Path) -> None:
    """Test that get_engine creates database file and initializes tables."""
    from research.db import get_engine, get_session

    db_path = tmp_path / "research.sqlite"
    engine = get_engine(str(db_path))
    session = get_session(engine)

    assert db_path.exists()
    assert session.query(CandidateResult).count() == 0
