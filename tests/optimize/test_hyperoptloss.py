from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_omega import OmegaHyperOptLoss
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_short_trade_dur import ShortTradeDurHyperOptLoss
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver


def test_hyperoptlossresolver_noname(default_conf):
    with pytest.raises(
        OperationalException,
        match=r"No Hyperopt loss set. Please use `--hyperopt-loss` to specify "
        r"the Hyperopt-Loss class to use\.",
    ):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_hyperoptlossresolver(mocker, default_conf) -> None:
    hl = ShortTradeDurHyperOptLoss
    mocker.patch(
        "freqtrade.resolvers.hyperopt_resolver.HyperOptLossResolver.load_object",
        MagicMock(return_value=hl()),
    )
    default_conf.update({"hyperopt_loss": "SharpeHyperOptLossDaily"})
    x = HyperOptLossResolver.load_hyperoptloss(default_conf)
    assert hasattr(x, "hyperopt_loss_function")


def test_hyperoptlossresolver_wrongname(default_conf) -> None:
    default_conf.update({"hyperopt_loss": "NonExistingLossClass"})

    with pytest.raises(OperationalException, match=r"Impossible to load HyperoptLoss.*"):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_loss_calculation_prefer_correct_trade_count(hyperopt_conf, hyperopt_results) -> None:
    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600 + 100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600 - 100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(hyperopt_conf, hyperopt_results) -> None:
    resultsb = hyperopt_results.copy()
    resultsb.loc[1, "trade_duration"] = 20

    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    longer = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    shorter = hl.hyperopt_loss_function(
        results=resultsb,
        trade_count=100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": resultsb["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert shorter < longer


def test_loss_calculation_has_limited_profit(hyperopt_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over["profit_ratio"] = hyperopt_results["profit_ratio"] * 2
    results_under = hyperopt_results.copy()
    results_under["profit_ratio"] = hyperopt_results["profit_ratio"] / 2

    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=results_over,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": results_over["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=results_under,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": results_under["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert over < correct
    assert under > correct


@pytest.mark.parametrize(
    "lossfunction",
    [
        "OnlyProfitHyperOptLoss",
        "SortinoHyperOptLoss",
        "SortinoHyperOptLossDaily",
        "SharpeHyperOptLoss",
        "SharpeHyperOptLossDaily",
        "MaxDrawDownHyperOptLoss",
        "MaxDrawDownRelativeHyperOptLoss",
        "MaxDrawDownPerPairHyperOptLoss",
        "CalmarHyperOptLoss",
        "OmegaHyperOptLoss",
        "ProfitDrawDownHyperOptLoss",
        "MultiMetricHyperOptLoss",
    ],
)
def test_loss_functions_better_profits(default_conf, hyperopt_results, lossfunction) -> None:
    results_over = hyperopt_results.copy()
    results_over["profit_abs"] = hyperopt_results["profit_abs"] * 2 + 0.2
    results_over["profit_ratio"] = hyperopt_results["profit_ratio"] * 2
    results_under = hyperopt_results.copy()
    results_under["profit_abs"] = hyperopt_results["profit_abs"] / 2 - 0.2
    results_under["profit_ratio"] = hyperopt_results["profit_ratio"] / 2
    pair_results = [
        {
            "key": "ETH/USDT",
            "max_drawdown_abs": 50.0,
            "profit_total_abs": 100.0,
        },
        {
            "key": "BTC/USDT",
            "max_drawdown_abs": 50.0,
            "profit_total_abs": 100.0,
        },
    ]
    pair_results_over = [
        {
            **p,
            "max_drawdown_abs": p["max_drawdown_abs"] * 0.5,
            "profit_total_abs": p["profit_total_abs"] * 2,
        }
        for p in pair_results
    ]
    pair_results_under = [
        {
            **p,
            "max_drawdown_abs": p["max_drawdown_abs"] * 2,
            "profit_total_abs": p["profit_total_abs"] * 0.5,
        }
        for p in pair_results
    ]

    default_conf.update({"hyperopt_loss": lossfunction})
    hl = HyperOptLossResolver.load_hyperoptloss(default_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=len(hyperopt_results),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": hyperopt_results["profit_abs"].sum(),
            "results_per_pair": pair_results,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=results_over,
        trade_count=len(results_over),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": results_over["profit_abs"].sum(),
            "results_per_pair": pair_results_over,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=results_under,
        trade_count=len(results_under),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": results_under["profit_abs"].sum(),
            "results_per_pair": pair_results_under,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    assert over < correct
    assert under > correct


def _omega_loss(default_conf, results, min_date, max_date, starting_balance=1000):
    default_conf.update({"hyperopt_loss": "OmegaHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(default_conf)
    return hl.hyperopt_loss_function(
        results=results,
        trade_count=len(results),
        min_date=min_date,
        max_date=max_date,
        config=default_conf,
        processed=None,
        backtest_stats={},
        starting_balance=starting_balance,
    )


def test_omega_loss_hand_computed(default_conf, mocker) -> None:
    # Four trading days and one flat day on a 1000 balance. Day 3 nets +30 from two trades.
    results = pd.DataFrame(
        {
            "profit_abs": [20.0, -10.0, 40.0, -10.0, -20.0],
            "close_date": [
                datetime(2019, 1, 1, 10),
                datetime(2019, 1, 2, 10),
                datetime(2019, 1, 3, 10),
                datetime(2019, 1, 3, 15),
                datetime(2019, 1, 4, 10),
            ],
        }
    )
    min_date, max_date = datetime(2019, 1, 1), datetime(2019, 1, 5)
    # Balances at the start of each day: 1000, 1020, 1010, 1040, 1020.
    # Daily returns: 20/1000, -10/1020, 30/1010, -20/1040, 0.
    gains = 20 / 1000 + 30 / 1010
    losses = 10 / 1020 + 20 / 1040
    loss = _omega_loss(default_conf, results, min_date, max_date)
    assert loss == pytest.approx(-gains / losses)
    assert loss == pytest.approx(-1.711848, rel=1e-6)

    # With a 1% daily threshold the flat day counts as a 1% shortfall.
    mocker.patch("freqtrade.optimize.hyperopt_loss.hyperopt_loss_omega.OMEGA_THRESHOLD", 0.01)
    gains_l = (20 / 1000 - 0.01) + (30 / 1010 - 0.01)
    losses_l = (0.01 + 10 / 1020) + (0.01 + 20 / 1040) + 0.01
    loss_l = OmegaHyperOptLoss.hyperopt_loss_function(
        results=results, min_date=min_date, max_date=max_date, starting_balance=1000
    )
    assert loss_l == pytest.approx(-gains_l / losses_l)


def test_omega_loss_edge_cases(default_conf) -> None:
    min_date, max_date = datetime(2019, 1, 1), datetime(2019, 1, 3)
    empty = pd.DataFrame({"profit_abs": [], "close_date": pd.to_datetime([])})
    assert _omega_loss(default_conf, empty, min_date, max_date) == 0.0

    # No losing day: the ratio is undefined and the epoch is treated as not optimal.
    winners = pd.DataFrame(
        {"profit_abs": [5.0, 7.0], "close_date": [datetime(2019, 1, 1), datetime(2019, 1, 2)]}
    )
    assert _omega_loss(default_conf, winners, min_date, max_date) == 100.0

    # Account wiped out.
    wiped = pd.DataFrame(
        {"profit_abs": [-1000.0, 5.0], "close_date": [datetime(2019, 1, 1), datetime(2019, 1, 2)]}
    )
    assert _omega_loss(default_conf, wiped, min_date, max_date) == 100.0
