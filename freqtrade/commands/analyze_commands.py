import logging
from pathlib import Path
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def setup_analyze_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for the entry/exit reason analysis module
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    config = setup_utils_configuration(args, method)

    no_unlimited_runmodes = {
        RunMode.BACKTEST: 'backtesting',
    }
    if method in no_unlimited_runmodes.keys():
        from freqtrade.data.btanalysis import get_latest_backtest_filename

        btfile = Path(get_latest_backtest_filename(config['user_data_dir'] / 'backtest_results'))
        signals_file = f"{btfile.stem}_signals.pkl"

        if (not (config['user_data_dir'] / 'backtest_results' / signals_file).exists()):
            raise OperationalException(
                "Cannot find latest backtest signals file. Run backtesting with --export signals."
            )

        if ('strategy' not in config):
            raise OperationalException(
                "No strategy defined. Use --strategy or supply in config."
            )

    return config


def start_analysis_entries_exits(args: Dict[str, Any]) -> None:
    """
    Start analysis script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.data.entryexitanalysis import process_entry_exit_reasons

    # Initialize configuration
    config = setup_analyze_configuration(args, RunMode.BACKTEST)

    print(config)

    logger.info('Starting freqtrade in analysis mode')

    process_entry_exit_reasons(Path(config['user_data_dir'], 'backtest_results'),
                               config['exchange']['pair_whitelist'],
                               config['strategy'],
                               config['analysis_groups'],
                               config['enter_reason_list'],
                               config['exit_reason_list'],
                               config['indicator_list']
                               )
