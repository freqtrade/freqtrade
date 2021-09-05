import logging
from typing import List

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def hyperopt_filter_epochs(epochs: List, filteroptions: dict, log: bool = True) -> List:
    """
    Filter our items from the list of hyperopt results
    """
    if filteroptions['only_best']:
        epochs = [x for x in epochs if x['is_best']]
    if filteroptions['only_profitable']:
        epochs = [x for x in epochs
                  if x['results_metrics'].get('profit_total', 0) > 0]

    epochs = _hyperopt_filter_epochs_trade_count(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_duration(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_profit(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_objective(epochs, filteroptions)
    if log:
        logger.info(f"{len(epochs)} " +
                    ("best " if filteroptions['only_best'] else "") +
                    ("profitable " if filteroptions['only_profitable'] else "") +
                    "epochs found.")
    return epochs


def _hyperopt_filter_epochs_trade(epochs: List, trade_count: int):
    """
    Filter epochs with trade-counts > trades
    """
    return [
        x for x in epochs if x['results_metrics'].get('total_trades', 0) > trade_count
    ]


def _hyperopt_filter_epochs_trade_count(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_trades'] > 0:
        epochs = _hyperopt_filter_epochs_trade(epochs, filteroptions['filter_min_trades'])

    if filteroptions['filter_max_trades'] > 0:
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('total_trades') < filteroptions['filter_max_trades']
        ]
    return epochs


def _hyperopt_filter_epochs_duration(epochs: List, filteroptions: dict) -> List:

    def get_duration_value(x):
        # Duration in minutes ...
        if 'holding_avg_s' in x['results_metrics']:
            avg = x['results_metrics']['holding_avg_s']
            return avg // 60
        raise OperationalException(
            "Holding-average not available. Please omit the filter on average time, "
            "or rerun hyperopt with this version")

    if filteroptions['filter_min_avg_time'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if get_duration_value(x) > filteroptions['filter_min_avg_time']
        ]
    if filteroptions['filter_max_avg_time'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if get_duration_value(x) < filteroptions['filter_max_avg_time']
        ]

    return epochs


def _hyperopt_filter_epochs_profit(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_avg_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_mean', 0) * 100
            > filteroptions['filter_min_avg_profit']
        ]
    if filteroptions['filter_max_avg_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_mean', 0) * 100
            < filteroptions['filter_max_avg_profit']
        ]
    if filteroptions['filter_min_total_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_total_abs', 0)
            > filteroptions['filter_min_total_profit']
        ]
    if filteroptions['filter_max_total_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_total_abs', 0)
            < filteroptions['filter_max_total_profit']
        ]
    return epochs


def _hyperopt_filter_epochs_objective(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_objective'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x['loss'] < filteroptions['filter_min_objective']]
    if filteroptions['filter_max_objective'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x['loss'] > filteroptions['filter_max_objective']]

    return epochs
