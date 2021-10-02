from datetime import datetime
from unittest.mock import MagicMock

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_loss_short_trade_dur import ShortTradeDurHyperOptLoss
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver


def test_hyperoptlossresolver_noname(default_conf):
    with pytest.raises(OperationalException,
                       match="No Hyperopt loss set. Please use `--hyperopt-loss` to specify "
                             "the Hyperopt-Loss class to use."):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_hyperoptlossresolver(mocker, default_conf) -> None:

    hl = ShortTradeDurHyperOptLoss
    mocker.patch(
        'freqtrade.resolvers.hyperopt_resolver.HyperOptLossResolver.load_object',
        MagicMock(return_value=hl())
    )
    default_conf.update({'hyperopt_loss': 'SharpeHyperOptLossDaily'})
    x = HyperOptLossResolver.load_hyperoptloss(default_conf)
    assert hasattr(x, "hyperopt_loss_function")


def test_hyperoptlossresolver_wrongname(default_conf) -> None:
    default_conf.update({'hyperopt_loss': "NonExistingLossClass"})

    with pytest.raises(OperationalException, match=r'Impossible to load HyperoptLoss.*'):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_loss_calculation_prefer_correct_trade_count(hyperopt_conf, hyperopt_results) -> None:
    hyperopt_conf.update({'hyperopt_loss': "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(hyperopt_results, 600,
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hl.hyperopt_loss_function(hyperopt_results, 600 + 100,
                                     datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hl.hyperopt_loss_function(hyperopt_results, 600 - 100,
                                      datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(hyperopt_conf, hyperopt_results) -> None:
    resultsb = hyperopt_results.copy()
    resultsb.loc[1, 'trade_duration'] = 20

    hyperopt_conf.update({'hyperopt_loss': "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    longer = hl.hyperopt_loss_function(hyperopt_results, 100,
                                       datetime(2019, 1, 1), datetime(2019, 5, 1))
    shorter = hl.hyperopt_loss_function(resultsb, 100,
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert shorter < longer


def test_loss_calculation_has_limited_profit(hyperopt_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_ratio'] = hyperopt_results['profit_ratio'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_ratio'] = hyperopt_results['profit_ratio'] / 2

    hyperopt_conf.update({'hyperopt_loss': "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(hyperopt_results, 600,
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hl.hyperopt_loss_function(results_over, 600,
                                     datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hl.hyperopt_loss_function(results_under, 600,
                                      datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over < correct
    assert under > correct


@pytest.mark.parametrize('lossfunction', [
    "OnlyProfitHyperOptLoss",
    "SortinoHyperOptLoss",
    "SortinoHyperOptLossDaily",
    "SharpeHyperOptLoss",
    "SharpeHyperOptLossDaily",
])
def test_loss_functions_better_profits(default_conf, hyperopt_results, lossfunction) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_abs'] = hyperopt_results['profit_abs'] * 2
    results_over['profit_ratio'] = hyperopt_results['profit_ratio'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_abs'] = hyperopt_results['profit_abs'] / 2
    results_under['profit_ratio'] = hyperopt_results['profit_ratio'] / 2

    default_conf.update({'hyperopt_loss': lossfunction})
    hl = HyperOptLossResolver.load_hyperoptloss(default_conf)
    correct = hl.hyperopt_loss_function(hyperopt_results, len(hyperopt_results),
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hl.hyperopt_loss_function(results_over, len(results_over),
                                     datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hl.hyperopt_loss_function(results_under, len(results_under),
                                      datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over < correct
    assert under > correct
