import logging
import os

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

        btp = Path(config.get('user_data_dir'), "backtest_results")
        btfile = get_latest_backtest_filename(btp)
        signals_file = f"{os.path.basename(os.path.splitext(btfile)[0])}_signals.pkl"

        if (not os.path.exists(Path(btp, signals_file))):
            raise OperationalException(
                "Cannot find latest backtest signals file. Run backtesting with --export signals."
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

    logger.info('Starting freqtrade in analysis mode')

    process_entry_exit_reasons(Path(config['user_data_dir'], 'backtest_results'),
                               config['exchange']['pair_whitelist'],
                               config['strategy'],
                               config['analysis_groups'],
                               config['enter_reason_list'],
                               config['exit_reason_list'],
                               config['indicator_list']
                               )
