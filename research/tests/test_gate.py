from pathlib import Path

import pytest

from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode
from research.db import get_engine, get_session
from research.gate import run_promotion_gate
from research.models import CandidateResult
from tests.conftest import get_default_conf, patch_exchange


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def _patch(mocker):
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])


def test_run_promotion_gate_raises_with_too_few_windows(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)

    with pytest.raises(ValueError, match="walk-forward windows"):
        run_promotion_gate(
            config=conf,
            strategy_id="StrategyTestV3",
            pairs=["UNITTEST/BTC"],
            timeframe="5m",
            datadir=TESTDATADIR,
            start=min_date,
            end=max_date,
            train_days=3650,  # deliberately far larger than the available data span
            test_days=3650,
            param_grid=[{"buy_rsi": 30}],
            db_path=str(tmp_path / "research.sqlite"),
        )


def test_run_promotion_gate_returns_result_and_writes_ledger_row(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)
    db_path = str(tmp_path / "research.sqlite")

    result = run_promotion_gate(
        config=conf,
        strategy_id="StrategyTestV3",
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        start=min_date,
        end=max_date,
        train_days=train_days,
        test_days=test_days,
        param_grid=[{"buy_rsi": 25}, {"buy_rsi": 35}],
        db_path=db_path,
    )

    assert result.strategy_id == "StrategyTestV3"
    assert isinstance(result.passed, bool)
    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert 0.0 <= result.permutation_p <= 1.0
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_trials >= 1

    session = get_session(get_engine(db_path))
    assert session.query(CandidateResult).filter_by(strategy_id="StrategyTestV3").count() == 1
