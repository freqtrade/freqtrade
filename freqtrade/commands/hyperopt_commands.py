import logging
from operator import itemgetter
from typing import Any, Dict, List

from colorama import init as colorama_init

from freqtrade.configuration import setup_utils_configuration
from freqtrade.data.btanalysis import get_latest_hyperopt_file
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode


logger = logging.getLogger(__name__)


def start_hyperopt_list(args: Dict[str, Any]) -> None:
    """
    List hyperopt epochs previously evaluated
    """
    from freqtrade.optimize.hyperopt_tools import HyperoptTools

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_colorized = config.get('print_colorized', False)
    print_json = config.get('print_json', False)
    export_csv = config.get('export_csv', None)
    no_details = config.get('hyperopt_list_no_details', False)
    no_header = False

    filteroptions = {
        'only_best': config.get('hyperopt_list_best', False),
        'only_profitable': config.get('hyperopt_list_profitable', False),
        'filter_min_trades': config.get('hyperopt_list_min_trades', 0),
        'filter_max_trades': config.get('hyperopt_list_max_trades', 0),
        'filter_min_avg_time': config.get('hyperopt_list_min_avg_time', None),
        'filter_max_avg_time': config.get('hyperopt_list_max_avg_time', None),
        'filter_min_avg_profit': config.get('hyperopt_list_min_avg_profit', None),
        'filter_max_avg_profit': config.get('hyperopt_list_max_avg_profit', None),
        'filter_min_total_profit': config.get('hyperopt_list_min_total_profit', None),
        'filter_max_total_profit': config.get('hyperopt_list_max_total_profit', None),
        'filter_min_objective': config.get('hyperopt_list_min_objective', None),
        'filter_max_objective': config.get('hyperopt_list_max_objective', None),
    }

    results_file = get_latest_hyperopt_file(
        config['user_data_dir'] / 'hyperopt_results',
        config.get('hyperoptexportfilename'))

    # Previous evaluations
    epochs = HyperoptTools.load_previous_results(results_file)
    total_epochs = len(epochs)

    epochs = hyperopt_filter_epochs(epochs, filteroptions)

    if print_colorized:
        colorama_init(autoreset=True)

    if not export_csv:
        try:
            print(HyperoptTools.get_result_table(config, epochs, total_epochs,
                                                 not filteroptions['only_best'],
                                                 print_colorized, 0))
        except KeyboardInterrupt:
            print('User interrupted..')

    if epochs and not no_details:
        sorted_epochs = sorted(epochs, key=itemgetter('loss'))
        results = sorted_epochs[0]
        HyperoptTools.print_epoch_details(results, total_epochs, print_json, no_header)

    if epochs and export_csv:
        HyperoptTools.export_csv_file(
            config, epochs, total_epochs, not filteroptions['only_best'], export_csv
        )


def start_hyperopt_show(args: Dict[str, Any]) -> None:
    """
    Show details of a hyperopt epoch previously evaluated
    """
    from freqtrade.optimize.hyperopt_tools import HyperoptTools

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_json = config.get('print_json', False)
    no_header = config.get('hyperopt_show_no_header', False)
    results_file = get_latest_hyperopt_file(
        config['user_data_dir'] / 'hyperopt_results',
        config.get('hyperoptexportfilename'))

    n = config.get('hyperopt_show_index', -1)

    filteroptions = {
        'only_best': config.get('hyperopt_list_best', False),
        'only_profitable': config.get('hyperopt_list_profitable', False),
        'filter_min_trades': config.get('hyperopt_list_min_trades', 0),
        'filter_max_trades': config.get('hyperopt_list_max_trades', 0),
        'filter_min_avg_time': config.get('hyperopt_list_min_avg_time', None),
        'filter_max_avg_time': config.get('hyperopt_list_max_avg_time', None),
        'filter_min_avg_profit': config.get('hyperopt_list_min_avg_profit', None),
        'filter_max_avg_profit': config.get('hyperopt_list_max_avg_profit', None),
        'filter_min_total_profit': config.get('hyperopt_list_min_total_profit', None),
        'filter_max_total_profit': config.get('hyperopt_list_max_total_profit', None),
        'filter_min_objective': config.get('hyperopt_list_min_objective', None),
        'filter_max_objective': config.get('hyperopt_list_max_objective', None)
    }

    # Previous evaluations
    epochs = HyperoptTools.load_previous_results(results_file)
    total_epochs = len(epochs)

    epochs = hyperopt_filter_epochs(epochs, filteroptions)
    filtered_epochs = len(epochs)

    if n > filtered_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be less than {filtered_epochs + 1}.")
    if n < -filtered_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be greater than {-filtered_epochs - 1}.")

    # Translate epoch index from human-readable format to pythonic
    if n > 0:
        n -= 1

    if epochs:
        val = epochs[n]
        HyperoptTools.print_epoch_details(val, total_epochs, print_json, no_header,
                                          header_str="Epoch details")


def hyperopt_filter_epochs(epochs: List, filteroptions: dict) -> List:
    """
    Filter our items from the list of hyperopt results
    """
    if filteroptions['only_best']:
        epochs = [x for x in epochs if x['is_best']]
    if filteroptions['only_profitable']:
        epochs = [x for x in epochs if x['results_metrics']['profit'] > 0]

    epochs = _hyperopt_filter_epochs_trade_count(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_duration(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_profit(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_objective(epochs, filteroptions)

    logger.info(f"{len(epochs)} " +
                ("best " if filteroptions['only_best'] else "") +
                ("profitable " if filteroptions['only_profitable'] else "") +
                "epochs found.")
    return epochs


def _hyperopt_filter_epochs_trade_count(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_trades'] > 0:
        epochs = [
            x for x in epochs
            if x['results_metrics']['trade_count'] > filteroptions['filter_min_trades']
        ]
    if filteroptions['filter_max_trades'] > 0:
        epochs = [
            x for x in epochs
            if x['results_metrics']['trade_count'] < filteroptions['filter_max_trades']
        ]
    return epochs


def _hyperopt_filter_epochs_duration(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_avg_time'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['duration'] > filteroptions['filter_min_avg_time']
        ]
    if filteroptions['filter_max_avg_time'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['duration'] < filteroptions['filter_max_avg_time']
        ]

    return epochs


def _hyperopt_filter_epochs_profit(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_avg_profit'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['avg_profit'] > filteroptions['filter_min_avg_profit']
        ]
    if filteroptions['filter_max_avg_profit'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['avg_profit'] < filteroptions['filter_max_avg_profit']
        ]
    if filteroptions['filter_min_total_profit'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['profit'] > filteroptions['filter_min_total_profit']
        ]
    if filteroptions['filter_max_total_profit'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]
        epochs = [
            x for x in epochs
            if x['results_metrics']['profit'] < filteroptions['filter_max_total_profit']
        ]
    return epochs


def _hyperopt_filter_epochs_objective(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_objective'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]

        epochs = [x for x in epochs if x['loss'] < filteroptions['filter_min_objective']]
    if filteroptions['filter_max_objective'] is not None:
        epochs = [x for x in epochs if x['results_metrics']['trade_count'] > 0]

        epochs = [x for x in epochs if x['loss'] > filteroptions['filter_max_objective']]

    return epochs
