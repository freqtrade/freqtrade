import logging
from operator import itemgetter
from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def start_hyperopt_list(args: dict[str, Any]) -> None:
    """
    List hyperopt epochs previously evaluated
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.btanalysis import get_latest_hyperopt_file
    from freqtrade.optimize.hyperopt.hyperopt_output import HyperoptOutput
    from freqtrade.optimize.hyperopt_tools import HyperoptTools

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_colorized = config.get("print_colorized", False)
    print_json = config.get("print_json", False)
    export_csv = config.get("export_csv")
    no_details = config.get("hyperopt_list_no_details", False)
    no_header = False

    results_file = get_latest_hyperopt_file(
        config["user_data_dir"] / "hyperopt_results", config.get("hyperoptexportfilename")
    )

    # Previous evaluations
    epochs, total_epochs = HyperoptTools.load_filtered_results(results_file, config)

    if not export_csv:
        try:
            h_out = HyperoptOutput()
            h_out.add_data(
                config,
                epochs,
                total_epochs,
                not config.get("hyperopt_list_best", False),
            )
            h_out.print(print_colorized=print_colorized)

        except KeyboardInterrupt:
            print("User interrupted..")

    if epochs and not no_details:
        sorted_epochs = sorted(epochs, key=itemgetter("loss"))
        results = sorted_epochs[0]
        HyperoptTools.show_epoch_details(results, total_epochs, print_json, no_header)

    if epochs and export_csv:
        HyperoptTools.export_csv_file(config, epochs, export_csv)


def start_hyperopt_show(args: dict[str, Any]) -> None:
    """
    Show details of a hyperopt epoch previously evaluated
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.btanalysis import get_latest_hyperopt_file
    from freqtrade.optimize.hyperopt_tools import HyperoptTools
    from freqtrade.optimize.optimize_reports import show_backtest_result

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_json = config.get("print_json", False)
    no_header = config.get("hyperopt_show_no_header", False)
    results_file = get_latest_hyperopt_file(
        config["user_data_dir"] / "hyperopt_results", config.get("hyperoptexportfilename")
    )

    n = config.get("hyperopt_show_index", -1)

    # Previous evaluations
    epochs, total_epochs = HyperoptTools.load_filtered_results(results_file, config)

    filtered_epochs = len(epochs)

    if n > filtered_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be less than {filtered_epochs + 1}."
        )
    if n < -filtered_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be greater than {-filtered_epochs - 1}."
        )

    # Translate epoch index from human-readable format to pythonic
    if n > 0:
        n -= 1

    if epochs:
        val = epochs[n]

        metrics = val["results_metrics"]
        if "strategy_name" in metrics:
            strategy_name = metrics["strategy_name"]
            show_backtest_result(
                strategy_name,
                metrics,
                metrics["stake_currency"],
                config.get("backtest_breakdown", []),
            )

            HyperoptTools.try_export_params(config, strategy_name, val)

        HyperoptTools.show_epoch_details(
            val, total_epochs, print_json, no_header, header_str="Epoch details"
        )
