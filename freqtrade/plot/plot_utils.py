from argparse import Namespace

from freqtrade.state import RunMode
from freqtrade.utils import setup_utils_configuration


def start_plot_dataframe(args: Namespace) -> None:
    """
    Plotting dataframe helper
    """
    # Import here to avoid errors if plot-dependencies are not installed.
    from freqtrade.plot.plotting import analyse_and_plot_pairs
    config = setup_utils_configuration(args, RunMode.OTHER)

    analyse_and_plot_pairs(config)
