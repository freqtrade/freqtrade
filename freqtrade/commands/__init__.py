# flake8: noqa: F401
"""
Commands module.
Contains all start-commands, subcommands and CLI Interface creation.

Note: Be careful with file-scoped imports in these subfiles.
    as they are parsed on startup, nothing containing optional modules should be loaded.
"""

from typing import Any, Dict

from freqtrade.commands.arguments import Arguments


def start_analysis_entries_exits(args: Dict[str, Any]) -> None:
    from freqtrade.commands.analyze_commands import start_analysis_entries_exits

    start_analysis_entries_exits(args)


def start_new_config(args: Dict[str, Any]) -> None:
    from freqtrade.commands.build_config_commands import start_new_config

    start_new_config(args)


def start_show_config(args: Dict[str, Any]) -> None:
    from freqtrade.commands.build_config_commands import start_show_config

    start_show_config(args)


def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    from freqtrade.commands.data_commands import start_convert_data

    start_convert_data(args, ohlcv)


def start_convert_trades(args: Dict[str, Any]) -> None:
    from freqtrade.commands.data_commands import start_convert_trades

    start_convert_trades(args)


def start_download_data(args: Dict[str, Any]) -> None:
    from freqtrade.commands.data_commands import start_download_data

    start_download_data(args)


def start_list_data(args: Dict[str, Any]) -> None:
    from freqtrade.commands.data_commands import start_list_data

    start_list_data(args)


def start_list_trades_data(args: Dict[str, Any]) -> None:
    from freqtrade.commands.data_commands import start_list_trades_data

    start_list_trades_data(args)


def start_convert_db(args: Dict[str, Any]) -> None:
    from freqtrade.commands.db_commands import start_convert_db

    start_convert_data(args)


def start_create_userdir(args: Dict[str, Any]) -> None:
    from freqtrade.commands.deploy_commands import start_create_userdir

    start_create_userdir(args)


def start_install_ui(args: Dict[str, Any]) -> None:
    from freqtrade.commands.deploy_commands import start_install_ui

    start_install_ui(args)


def start_new_strategy(args: Dict[str, Any]) -> None:
    from freqtrade.commands.deploy_commands import start_new_strategy

    start_new_strategy(args)


def start_hyperopt_list(args: Dict[str, Any]) -> None:
    from freqtrade.commands.hyperopt_commands import start_hyperopt_list

    start_hyperopt_list(args)


def start_hyperopt_show(args: Dict[str, Any]) -> None:
    from freqtrade.commands.hyperopt_commands import start_hyperopt_show

    start_hyperopt_show(args)


def start_list_exchanges(args: Dict[str, Any]) -> None:
    from freqtrade.commands.list_commands import start_list_exchanges

    start_list_exchanges(args)


def start_list_freqAI_models(args: Dict[str, Any]) -> None:
    from freqtrade.commands.list_commands import start_list_freqAI_models

    start_list_freqAI_models(args)


def start_list_markets(args: Dict[str, Any], pairs_only: bool = False) -> None:
    from freqtrade.commands.list_commands import start_list_markets

    start_list_markets(args, pairs_only)


def start_list_strategies(args: Dict[str, Any]) -> None:
    from freqtrade.commands.list_commands import start_list_strategies

    start_list_strategies(args)


def start_list_timeframes(args: Dict[str, Any]) -> None:
    from freqtrade.commands.list_commands import start_list_timeframes

    start_list_timeframes(args)


def start_show_trades(args: Dict[str, Any]) -> None:
    from freqtrade.commands.list_commands import start_show_trades

    start_show_trades(args)


def start_backtesting(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_backtesting

    start_backtesting(args)


def start_backtesting_show(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_backtesting_show

    start_backtesting_show(args)


def start_edge(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_edge

    start_edge(args)


def start_hyperopt(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_hyperopt

    start_hyperopt(args)


def start_lookahead_analysis(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_lookahead_analysis

    start_lookahead_analysis(args)


def start_recursive_analysis(args: Dict[str, Any]) -> None:
    from freqtrade.commands.optimize_commands import start_recursive_analysis

    start_recursive_analysis(args)


def start_test_pairlist(args: Dict[str, Any]) -> None:
    from freqtrade.commands.pairlist_commands import start_test_pairlist

    start_test_pairlist(args)


def start_plot_dataframe(args: Dict[str, Any]) -> None:
    from freqtrade.commands.plot_commands import start_plot_dataframe

    start_plot_dataframe(args)


def start_plot_profit(args: Dict[str, Any]) -> None:
    from freqtrade.commands.plot_commands import start_plot_profit

    start_plot_profit(args)


def start_strategy_update(args: Dict[str, Any]) -> None:
    from freqtrade.commands.strategy_utils_commands import start_strategy_update

    start_strategy_update(args)


def start_trading(args: Dict[str, Any]) -> int:
    from freqtrade.commands.trade_commands import start_trading

    return start_trading(args)


def start_webserver(args: Dict[str, Any]) -> None:
    from freqtrade.commands.webserver_commands import start_webserver

    start_webserver(args)
