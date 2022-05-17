import logging
from operator import itemgetter
from typing import Any, Dict

from colorama import init as colorama_init

from freqtrade.configuration import setup_utils_configuration
from freqtrade.data.btanalysis import get_latest_hyperopt_file
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.optimize_reports import show_backtest_result


logger = logging.getLogger(__name__)


def start_hyperopt_list(args: Dict[str, Any]) -> None:
    """
    List hyperopt epochs previously evaluated
    """
    from freqtrade.optimize.hyperopt_tools import HyperoptTools

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_colorized = config.get('print_colorized', False)
    print_json = config.get('print_json', False)
    export_csv = config.get('export_csv')
    no_details = config.get('hyperopt_list_no_details', False)
    no_header = False

    results_file = get_latest_hyperopt_file(
        config['user_data_dir'] / 'hyperopt_results',
        config.get('hyperoptexportfilename'))

    # Previous evaluations
    epochs, total_epochs = HyperoptTools.load_filtered_results(results_file, config)

    if print_colorized:
        colorama_init(autoreset=True)

    if not export_csv:
        try:
            print(HyperoptTools.get_result_table(config, epochs, total_epochs,
                                                 not config.get('hyperopt_list_best', False),
                                                 print_colorized, 0))
        except KeyboardInterrupt:
            print('User interrupted..')

    if epochs and not no_details:
        sorted_epochs = sorted(epochs, key=itemgetter('loss'))
        results = sorted_epochs[0]
        HyperoptTools.show_epoch_details(results, total_epochs, print_json, no_header)

    if epochs and export_csv:
        HyperoptTools.export_csv_file(
            config, epochs, export_csv
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

    # Previous evaluations
    epochs, total_epochs = HyperoptTools.load_filtered_results(results_file, config)

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

        metrics = val['results_metrics']
        if 'strategy_name' in metrics:
            strategy_name = metrics['strategy_name']
            show_backtest_result(strategy_name, metrics,
                                 metrics['stake_currency'], config.get('backtest_breakdown', []))

            HyperoptTools.try_export_params(config, strategy_name, val)

        HyperoptTools.show_epoch_details(val, total_epochs, print_json, no_header,
                                         header_str="Epoch details")
