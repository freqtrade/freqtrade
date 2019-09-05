from argparse import Namespace
from freqtrade import OperationalException
from freqtrade.state import RunMode
from freqtrade.utils import setup_utils_configuration


def validate_plot_args(args: Namespace):
    args_tmp = vars(args)
    if not args_tmp.get('datadir') and not args_tmp.get('config'):
        raise OperationalException(
            "You need to specify either `--datadir` or `--config` "
            "for plot-profit and plot-dataframe.")


def start_plot_dataframe(args: Namespace) -> None:
    """
    Entrypoint for dataframe plotting
    """
    # Import here to avoid errors if plot-dependencies are not installed.
    from freqtrade.plot.plotting import load_and_plot_trades
    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)

    load_and_plot_trades(config)


def start_plot_profit(args: Namespace) -> None:
    """
    Entrypoint for plot_profit
    """
    # Import here to avoid errors if plot-dependencies are not installed.
    from freqtrade.plot.plotting import plot_profit
    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)

    plot_profit(config)
