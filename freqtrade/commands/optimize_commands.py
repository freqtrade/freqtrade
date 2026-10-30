import logging
from typing import Any

from freqtrade import constants
from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


def setup_optimize_configuration(args: dict[str, Any], method: RunMode) -> dict[str, Any]:
    """
    Prepare the configuration for the Hyperopt module
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.util import fmt_coin, get_dry_run_wallet

    config = setup_utils_configuration(args, method)

    no_unlimited_runmodes = {
        RunMode.BACKTEST: "backtesting",
        RunMode.HYPEROPT: "hyperoptimization",
    }
    if method in no_unlimited_runmodes:
        wallet_size = get_dry_run_wallet(config) * config["tradable_balance_ratio"]
        # tradable_balance_ratio
        if (
            config["stake_amount"] != constants.UNLIMITED_STAKE_AMOUNT
            and config["stake_amount"] > wallet_size
        ):
            wallet = fmt_coin(wallet_size, config["stake_currency"])
            stake = fmt_coin(config["stake_amount"], config["stake_currency"])
            raise ConfigurationError(
                f"Starting balance ({wallet}) is smaller than stake_amount {stake}. "
                f"Wallet is calculated as `dry_run_wallet * tradable_balance_ratio`."
            )

    return config


def start_backtesting(args: dict[str, Any]) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Import here to avoid loading backtesting module when it's not used
    from freqtrade.optimize.backtesting import Backtesting

    # Initialize configuration
    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    logger.info("Starting freqtrade in Backtesting mode")

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()


def start_backtesting_show(args: dict[str, Any]) -> None:
    """
    Show previous backtest result
    """
    from freqtrade.configuration import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.btanalysis import (
        get_backtest_filename,
        get_backtest_strategy_params,
        load_backtest_stats,
    )
    from freqtrade.optimize.optimize_reports import (
        show_backtest_params,
        show_backtest_results,
        show_sorted_pairlist,
    )

    filename = get_backtest_filename(config["exportdirectory"], config["exportfilename"])
    results = load_backtest_stats(filename)

    show_backtest_results(config, results)
    show_sorted_pairlist(config, results)

    show_params = config.get("backtest_show_params", False)
    export_params = config.get("backtest_export_params", False)
    if show_params or export_params:
        for strategy_name in results["strategy"]:
            params = get_backtest_strategy_params(filename, strategy_name)
            if not params:
                logger.warning(f"No parameters stored for {strategy_name}.")
                continue
            if show_params:
                show_backtest_params(strategy_name, params.get("params", {}))
            if export_params:
                _export_backtest_params(config, strategy_name, params)


def _export_backtest_params(config: Config, strategy_name: str, params: dict[str, Any]) -> None:
    """
    Write the parameters of a stored backtest to the strategy's parameter file.
    """
    from freqtrade.optimize.hyperopt_tools import HyperoptTools

    fn = HyperoptTools.get_strategy_filename(config, strategy_name)
    if not fn:
        logger.warning(f"Strategy {strategy_name} not found, not exporting parameter file.")
        return
    filename = fn.with_suffix(".json")
    logger.info(f"Dumping parameters to {filename}")
    with filename.open("w") as f:
        HyperoptTools.serialize_params(params, f)


def start_hyperopt(args: dict[str, Any]) -> None:
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
            f"{e}. Please ensure that the hyperopt dependencies are installed."
        ) from e
    # Initialize configuration
    config = setup_optimize_configuration(args, RunMode.HYPEROPT)

    logger.info("Starting freqtrade in Hyperopt mode")

    lock = FileLock(Hyperopt.get_lock_filename(config))

    try:
        with lock.acquire(timeout=1):
            # Remove noisy log messages
            logging.getLogger("hyperopt.tpe").setLevel(logging.WARNING)
            logging.getLogger("filelock").setLevel(logging.WARNING)

            # Initialize backtesting object
            hyperopt = Hyperopt(config)
            hyperopt.start()

    except Timeout:
        logger.info("Another running instance of freqtrade Hyperopt detected.")
        logger.info(
            "Simultaneous execution of multiple Hyperopt commands is not supported. "
            "Hyperopt module is resource hungry. Please run your Hyperopt sequentially "
            "or on separate machines."
        )
        logger.info("Quitting now.")
        # TODO: return False here in order to help freqtrade to exit
        # with non-zero exit code...
        # Same in Edge and Backtesting start() functions.


def start_edge(args: dict[str, Any]) -> None:
    """
    Start Edge script
    :param args: Cli args from Arguments()
    :return: None
    """
    raise ConfigurationError(
        "The Edge module has been deprecated in 2023.9 and removed in 2025.6. "
        "All functionalities of edge have been removed."
    )


def start_lookahead_analysis(args: dict[str, Any]) -> None:
    """
    Start the backtest bias tester script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    LookaheadAnalysisSubFunctions.start(config)


def start_recursive_analysis(args: dict[str, Any]) -> None:
    """
    Start the backtest recursive tester script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    RecursiveAnalysisSubFunctions.start(config)
