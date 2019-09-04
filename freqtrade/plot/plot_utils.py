from argparse import Namespace

from freqtrade.state import RunMode
from freqtrade.utils import setup_utils_configuration


def start_plot_dataframe(args: Namespace) -> None:
    """
    Entrypoint for dataframe plotting
    """
    # Import here to avoid errors if plot-dependencies are not installed.
    from freqtrade.plot.plotting import load_and_plot_trades
    config = setup_utils_configuration(args, RunMode.PLOT)

    load_and_plot_trades(config)


def start_plot_profit(args: Namespace) -> None:
    """
    Entrypoint for plot_profit
    """
    # Import here to avoid errors if plot-dependencies are not installed.
    from freqtrade.plot.plotting import plot_profit
    config = setup_utils_configuration(args, RunMode.PLOT)

    plot_profit(config)
