import logging
from typing import Any, Dict

from freqtrade import constants
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.util import round_coin_value


logger = logging.getLogger(__name__)


def setup_optimize_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for the Hyperopt module
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    config = setup_utils_configuration(args, method)

    no_unlimited_runmodes = {
        RunMode.BACKTEST: 'backtesting',
        RunMode.HYPEROPT: 'hyperoptimization',
    }
    if method in no_unlimited_runmodes.keys():
        wallet_size = config['dry_run_wallet'] * config['tradable_balance_ratio']
        # tradable_balance_ratio
        if (config['stake_amount'] != constants.UNLIMITED_STAKE_AMOUNT
                and config['stake_amount'] > wallet_size):
            wallet = round_coin_value(wallet_size, config['stake_currency'])
            stake = round_coin_value(config['stake_amount'], config['stake_currency'])
            raise OperationalException(
                f"Starting balance ({wallet}) is smaller than stake_amount {stake}. "
                f"Wallet is calculated as `dry_run_wallet * tradable_balance_ratio`."
                )

    return config


def start_backtesting(args: Dict[str, Any]) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Import here to avoid loading backtesting module when it's not used
    from freqtrade.optimize.backtesting import Backtesting

    # Initialize configuration
    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()


def start_backtesting_show(args: Dict[str, Any]) -> None:
    """
    Show previous backtest result
    """

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.btanalysis import load_backtest_stats
    from freqtrade.optimize.optimize_reports import show_backtest_results, show_sorted_pairlist

    results = load_backtest_stats(config['exportfilename'])

    show_backtest_results(config, results)
    show_sorted_pairlist(config, results)


def start_hyperopt(args: Dict[str, Any]) -> None:
    """
    Start hyperopt script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Import here to avoid loading hyperopt module when it's not used
    try:
        from filelock import FileLock, Timeout

        from freqtrade.optimize.hyperopt import Hyperopt
    except ImportError as e:
        raise OperationalException(
            f"{e}. Please ensure that the hyperopt dependencies are installed.") from e
    # Initialize configuration
    config = setup_optimize_configuration(args, RunMode.HYPEROPT)

    logger.info('Starting freqtrade in Hyperopt mode')

    lock = FileLock(Hyperopt.get_lock_filename(config))

    try:
        with lock.acquire(timeout=1):

            # Remove noisy log messages
            logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)
            logging.getLogger('filelock').setLevel(logging.WARNING)

            # Initialize backtesting object
            hyperopt = Hyperopt(config)
            hyperopt.start()

    except Timeout:
        logger.info("Another running instance of freqtrade Hyperopt detected.")
        logger.info("Simultaneous execution of multiple Hyperopt commands is not supported. "
                    "Hyperopt module is resource hungry. Please run your Hyperopt sequentially "
                    "or on separate machines.")
        logger.info("Quitting now.")
        # TODO: return False here in order to help freqtrade to exit
        # with non-zero exit code...
        # Same in Edge and Backtesting start() functions.


def start_edge(args: Dict[str, Any]) -> None:
    """
    Start Edge script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.optimize.edge_cli import EdgeCli

    # Initialize configuration
    config = setup_optimize_configuration(args, RunMode.EDGE)
    logger.info('Starting freqtrade in Edge mode')

    # Initialize Edge object
    edge_cli = EdgeCli(config)
    edge_cli.start()


def start_lookahead_analysis(args: Dict[str, Any]) -> None:
    """
    Start the backtest bias tester script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    LookaheadAnalysisSubFunctions.start(config)


def start_recursive_analysis(args: Dict[str, Any]) -> None:
    """
    Start the backtest recursive tester script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    RecursiveAnalysisSubFunctions.start(config)
