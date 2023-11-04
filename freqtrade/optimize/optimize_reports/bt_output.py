import logging
from typing import Any, Dict, List

from tabulate import tabulate

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config
from freqtrade.misc import decimals_per_coin, round_coin_value
from freqtrade.optimize.optimize_reports.optimize_reports import generate_periodic_breakdown_stats
from freqtrade.types import BacktestResultType


logger = logging.getLogger(__name__)


def _get_line_floatfmt(stake_currency: str) -> List[str]:
    """
    Generate floatformat (goes in line with _generate_result_line())
    """
    return ['s', 'd', '.2f', '.2f', f'.{decimals_per_coin(stake_currency)}f',
            '.2f', 'd', 's', 's']


def _get_line_header(first_column: str, stake_currency: str,
                     direction: str = 'Entries') -> List[str]:
    """
    Generate header lines (goes in line with _generate_result_line())
    """
    return [first_column, direction, 'Avg Profit %', 'Cum Profit %',
            f'Tot Profit {stake_currency}', 'Tot Profit %', 'Avg Duration',
            'Win  Draw  Loss  Win%']


def generate_wins_draws_losses(wins, draws, losses):
    if wins > 0 and losses == 0:
        wl_ratio = '100'
    elif wins == 0:
        wl_ratio = '0'
    else:
        wl_ratio = f'{100.0 / (wins + draws + losses) * wins:.1f}' if losses > 0 else '100'
    return f'{wins:>4}  {draws:>4}  {losses:>4}  {wl_ratio:>4}'


def text_table_bt_results(pair_results: List[Dict[str, Any]], stake_currency: str) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :param pair_results: List of Dictionaries - one entry per pair + final TOTAL row
    :param stake_currency: stake-currency - used to correctly name headers
    :return: pretty printed table with tabulate as string
    """

    headers = _get_line_header('Pair', stake_currency)
    floatfmt = _get_line_floatfmt(stake_currency)
    output = [[
        t['key'], t['trades'], t['profit_mean_pct'], t['profit_sum_pct'], t['profit_total_abs'],
        t['profit_total_pct'], t['duration_avg'],
        generate_wins_draws_losses(t['wins'], t['draws'], t['losses'])
    ] for t in pair_results]
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(output, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")


def text_table_exit_reason(exit_reason_stats: List[Dict[str, Any]], stake_currency: str) -> str:
    """
    Generate small table outlining Backtest results
    :param sell_reason_stats: Exit reason metrics
    :param stake_currency: Stakecurrency used
    :return: pretty printed table with tabulate as string
    """
    headers = [
        'Exit Reason',
        'Exits',
        'Win  Draws  Loss  Win%',
        'Avg Profit %',
        'Cum Profit %',
        f'Tot Profit {stake_currency}',
        'Tot Profit %',
    ]

    output = [[
        t.get('exit_reason', t.get('sell_reason')), t['trades'],
        generate_wins_draws_losses(t['wins'], t['draws'], t['losses']),
        t['profit_mean_pct'], t['profit_sum_pct'],
        round_coin_value(t['profit_total_abs'], stake_currency, False),
        t['profit_total_pct'],
    ] for t in exit_reason_stats]
    return tabulate(output, headers=headers, tablefmt="orgtbl", stralign="right")


def text_table_tags(tag_type: str, tag_results: List[Dict[str, Any]], stake_currency: str) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :param pair_results: List of Dictionaries - one entry per pair + final TOTAL row
    :param stake_currency: stake-currency - used to correctly name headers
    :return: pretty printed table with tabulate as string
    """
    if (tag_type == "enter_tag"):
        headers = _get_line_header("TAG", stake_currency)
    else:
        headers = _get_line_header("TAG", stake_currency, 'Exits')
    floatfmt = _get_line_floatfmt(stake_currency)
    output = [
        [
            t['key'] if t['key'] is not None and len(
                t['key']) > 0 else "OTHER",
            t['trades'],
            t['profit_mean_pct'],
            t['profit_sum_pct'],
            t['profit_total_abs'],
            t['profit_total_pct'],
            t['duration_avg'],
            generate_wins_draws_losses(
                t['wins'],
                t['draws'],
                t['losses'])] for t in tag_results]
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(output, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")


def text_table_periodic_breakdown(days_breakdown_stats: List[Dict[str, Any]],
                                  stake_currency: str, period: str) -> str:
    """
    Generate small table with Backtest results by days
    :param days_breakdown_stats: Days breakdown metrics
    :param stake_currency: Stakecurrency used
    :return: pretty printed table with tabulate as string
    """
    headers = [
        period.capitalize(),
        f'Tot Profit {stake_currency}',
        'Wins',
        'Draws',
        'Losses',
    ]
    output = [[
        d['date'], round_coin_value(d['profit_abs'], stake_currency, False),
        d['wins'], d['draws'], d['loses'],
    ] for d in days_breakdown_stats]
    return tabulate(output, headers=headers, tablefmt="orgtbl", stralign="right")


def text_table_strategy(strategy_results, stake_currency: str) -> str:
    """
    Generate summary table per strategy
    :param strategy_results: Dict of <Strategyname: DataFrame> containing results for all strategies
    :param stake_currency: stake-currency - used to correctly name headers
    :return: pretty printed table with tabulate as string
    """
    floatfmt = _get_line_floatfmt(stake_currency)
    headers = _get_line_header('Strategy', stake_currency)
    # _get_line_header() is also used for per-pair summary. Per-pair drawdown is mostly useless
    # therefore we slip this column in only for strategy summary here.
    headers.append('Drawdown')

    # Align drawdown string on the center two space separator.
    if 'max_drawdown_account' in strategy_results[0]:
        drawdown = [f'{t["max_drawdown_account"] * 100:.2f}' for t in strategy_results]
    else:
        # Support for prior backtest results
        drawdown = [f'{t["max_drawdown_per"]:.2f}' for t in strategy_results]

    dd_pad_abs = max([len(t['max_drawdown_abs']) for t in strategy_results])
    dd_pad_per = max([len(dd) for dd in drawdown])
    drawdown = [f'{t["max_drawdown_abs"]:>{dd_pad_abs}} {stake_currency}  {dd:>{dd_pad_per}}%'
                for t, dd in zip(strategy_results, drawdown)]

    output = [[
        t['key'], t['trades'], t['profit_mean_pct'], t['profit_sum_pct'], t['profit_total_abs'],
        t['profit_total_pct'], t['duration_avg'],
        generate_wins_draws_losses(t['wins'], t['draws'], t['losses']), drawdown]
        for t, drawdown in zip(strategy_results, drawdown)]
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(output, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")


def text_table_add_metrics(strat_results: Dict) -> str:
    if len(strat_results['trades']) > 0:
        best_trade = max(strat_results['trades'], key=lambda x: x['profit_ratio'])
        worst_trade = min(strat_results['trades'], key=lambda x: x['profit_ratio'])

        short_metrics = [
            ('', ''),  # Empty line to improve readability
            ('Long / Short',
             f"{strat_results.get('trade_count_long', 'total_trades')} / "
             f"{strat_results.get('trade_count_short', 0)}"),
            ('Total profit Long %', f"{strat_results['profit_total_long']:.2%}"),
            ('Total profit Short %', f"{strat_results['profit_total_short']:.2%}"),
            ('Absolute profit Long', round_coin_value(strat_results['profit_total_long_abs'],
                                                      strat_results['stake_currency'])),
            ('Absolute profit Short', round_coin_value(strat_results['profit_total_short_abs'],
                                                       strat_results['stake_currency'])),
        ] if strat_results.get('trade_count_short', 0) > 0 else []

        drawdown_metrics = []
        if 'max_relative_drawdown' in strat_results:
            # Compatibility to show old hyperopt results
            drawdown_metrics.append(
                ('Max % of account underwater', f"{strat_results['max_relative_drawdown']:.2%}")
            )
        drawdown_metrics.extend([
            ('Absolute Drawdown (Account)', f"{strat_results['max_drawdown_account']:.2%}")
            if 'max_drawdown_account' in strat_results else (
                'Drawdown', f"{strat_results['max_drawdown']:.2%}"),
            ('Absolute Drawdown', round_coin_value(strat_results['max_drawdown_abs'],
                                                   strat_results['stake_currency'])),
            ('Drawdown high', round_coin_value(strat_results['max_drawdown_high'],
                                               strat_results['stake_currency'])),
            ('Drawdown low', round_coin_value(strat_results['max_drawdown_low'],
                                              strat_results['stake_currency'])),
            ('Drawdown Start', strat_results['drawdown_start']),
            ('Drawdown End', strat_results['drawdown_end']),
        ])

        entry_adjustment_metrics = [
            ('Canceled Trade Entries', strat_results.get('canceled_trade_entries', 'N/A')),
            ('Canceled Entry Orders', strat_results.get('canceled_entry_orders', 'N/A')),
            ('Replaced Entry Orders', strat_results.get('replaced_entry_orders', 'N/A')),
        ] if strat_results.get('canceled_entry_orders', 0) > 0 else []

        # Newly added fields should be ignored if they are missing in strat_results. hyperopt-show
        # command stores these results and newer version of freqtrade must be able to handle old
        # results with missing new fields.
        metrics = [
            ('Backtesting from', strat_results['backtest_start']),
            ('Backtesting to', strat_results['backtest_end']),
            ('Max open trades', strat_results['max_open_trades']),
            ('', ''),  # Empty line to improve readability
            ('Total/Daily Avg Trades',
                f"{strat_results['total_trades']} / {strat_results['trades_per_day']}"),

            ('Starting balance', round_coin_value(strat_results['starting_balance'],
                                                  strat_results['stake_currency'])),
            ('Final balance', round_coin_value(strat_results['final_balance'],
                                               strat_results['stake_currency'])),
            ('Absolute profit ', round_coin_value(strat_results['profit_total_abs'],
                                                  strat_results['stake_currency'])),
            ('Total profit %', f"{strat_results['profit_total']:.2%}"),
            ('CAGR %', f"{strat_results['cagr']:.2%}" if 'cagr' in strat_results else 'N/A'),
            ('Sortino', f"{strat_results['sortino']:.2f}" if 'sortino' in strat_results else 'N/A'),
            ('Sharpe', f"{strat_results['sharpe']:.2f}" if 'sharpe' in strat_results else 'N/A'),
            ('Calmar', f"{strat_results['calmar']:.2f}" if 'calmar' in strat_results else 'N/A'),
            ('Profit factor', f'{strat_results["profit_factor"]:.2f}' if 'profit_factor'
                              in strat_results else 'N/A'),
            ('Expectancy (Ratio)', (
                f"{strat_results['expectancy']:.2f} ({strat_results['expectancy_ratio']:.2f})" if
                'expectancy_ratio' in strat_results else 'N/A')),
            ('Trades per day', strat_results['trades_per_day']),
            ('Avg. daily profit %',
             f"{(strat_results['profit_total'] / strat_results['backtest_days']):.2%}"),
            ('Avg. stake amount', round_coin_value(strat_results['avg_stake_amount'],
                                                   strat_results['stake_currency'])),
            ('Total trade volume', round_coin_value(strat_results['total_volume'],
                                                    strat_results['stake_currency'])),
            *short_metrics,
            ('', ''),  # Empty line to improve readability
            ('Best Pair', f"{strat_results['best_pair']['key']} "
                          f"{strat_results['best_pair']['profit_sum']:.2%}"),
            ('Worst Pair', f"{strat_results['worst_pair']['key']} "
                           f"{strat_results['worst_pair']['profit_sum']:.2%}"),
            ('Best trade', f"{best_trade['pair']} {best_trade['profit_ratio']:.2%}"),
            ('Worst trade', f"{worst_trade['pair']} "
                            f"{worst_trade['profit_ratio']:.2%}"),

            ('Best day', round_coin_value(strat_results['backtest_best_day_abs'],
                                          strat_results['stake_currency'])),
            ('Worst day', round_coin_value(strat_results['backtest_worst_day_abs'],
                                           strat_results['stake_currency'])),
            ('Days win/draw/lose', f"{strat_results['winning_days']} / "
                f"{strat_results['draw_days']} / {strat_results['losing_days']}"),
            ('Avg. Duration Winners', f"{strat_results['winner_holding_avg']}"),
            ('Avg. Duration Loser', f"{strat_results['loser_holding_avg']}"),
            ('Max Consecutive Wins / Loss',
             f"{strat_results['max_consecutive_wins']} / {strat_results['max_consecutive_losses']}"
             if 'max_consecutive_losses' in strat_results else 'N/A'),
            ('Rejected Entry signals', strat_results.get('rejected_signals', 'N/A')),
            ('Entry/Exit Timeouts',
             f"{strat_results.get('timedout_entry_orders', 'N/A')} / "
             f"{strat_results.get('timedout_exit_orders', 'N/A')}"),
            *entry_adjustment_metrics,
            ('', ''),  # Empty line to improve readability

            ('Min balance', round_coin_value(strat_results['csum_min'],
                                             strat_results['stake_currency'])),
            ('Max balance', round_coin_value(strat_results['csum_max'],
                                             strat_results['stake_currency'])),

            *drawdown_metrics,
            ('Market change', f"{strat_results['market_change']:.2%}"),
        ]

        return tabulate(metrics, headers=["Metric", "Value"], tablefmt="orgtbl")
    else:
        start_balance = round_coin_value(strat_results['starting_balance'],
                                         strat_results['stake_currency'])
        stake_amount = round_coin_value(
            strat_results['stake_amount'], strat_results['stake_currency']
        ) if strat_results['stake_amount'] != UNLIMITED_STAKE_AMOUNT else 'unlimited'

        message = ("No trades made. "
                   f"Your starting balance was {start_balance}, "
                   f"and your stake was {stake_amount}."
                   )
        return message


def show_backtest_result(strategy: str, results: Dict[str, Any], stake_currency: str,
                         backtest_breakdown=[]):
    """
    Print results for one strategy
    """
    # Print results
    print(f"Result for strategy {strategy}")
    table = text_table_bt_results(results['results_per_pair'], stake_currency=stake_currency)
    if isinstance(table, str):
        print(' BACKTESTING REPORT '.center(len(table.splitlines()[0]), '='))
    print(table)

    table = text_table_bt_results(results['left_open_trades'], stake_currency=stake_currency)
    if isinstance(table, str) and len(table) > 0:
        print(' LEFT OPEN TRADES REPORT '.center(len(table.splitlines()[0]), '='))
    print(table)

    if (results.get('results_per_enter_tag') is not None
            or results.get('results_per_buy_tag') is not None):
        # results_per_buy_tag is deprecated and should be removed 2 versions after short golive.
        table = text_table_tags(
            "enter_tag",
            results.get('results_per_enter_tag', results.get('results_per_buy_tag')),
            stake_currency=stake_currency)

        if isinstance(table, str) and len(table) > 0:
            print(' ENTER TAG STATS '.center(len(table.splitlines()[0]), '='))
        print(table)

    exit_reasons = results.get('exit_reason_summary', results.get('sell_reason_summary'))
    table = text_table_exit_reason(exit_reason_stats=exit_reasons,
                                   stake_currency=stake_currency)
    if isinstance(table, str) and len(table) > 0:
        print(' EXIT REASON STATS '.center(len(table.splitlines()[0]), '='))
    print(table)

    for period in backtest_breakdown:
        if period in results.get('periodic_breakdown', {}):
            days_breakdown_stats = results['periodic_breakdown'][period]
        else:
            days_breakdown_stats = generate_periodic_breakdown_stats(
                trade_list=results['trades'], period=period)
        table = text_table_periodic_breakdown(days_breakdown_stats=days_breakdown_stats,
                                              stake_currency=stake_currency, period=period)
        if isinstance(table, str) and len(table) > 0:
            print(f' {period.upper()} BREAKDOWN '.center(len(table.splitlines()[0]), '='))
        print(table)

    table = text_table_add_metrics(results)
    if isinstance(table, str) and len(table) > 0:
        print(' SUMMARY METRICS '.center(len(table.splitlines()[0]), '='))
    print(table)

    if isinstance(table, str) and len(table) > 0:
        print('=' * len(table.splitlines()[0]))

    print()


def show_backtest_results(config: Config, backtest_stats: BacktestResultType):
    stake_currency = config['stake_currency']

    for strategy, results in backtest_stats['strategy'].items():
        show_backtest_result(
            strategy, results, stake_currency,
            config.get('backtest_breakdown', []))

    if len(backtest_stats['strategy']) > 0:
        # Print Strategy summary table

        table = text_table_strategy(backtest_stats['strategy_comparison'], stake_currency)
        print(f"Backtested {results['backtest_start']} -> {results['backtest_end']} |"
              f" Max open trades : {results['max_open_trades']}")
        print(' STRATEGY SUMMARY '.center(len(table.splitlines()[0]), '='))
        print(table)
        print('=' * len(table.splitlines()[0]))
        print('\nFor more details, please look at the detail tables above')


def show_sorted_pairlist(config: Config, backtest_stats: BacktestResultType):
    if config.get('backtest_show_pair_list', False):
        for strategy, results in backtest_stats['strategy'].items():
            print(f"Pairs for Strategy {strategy}: \n[")
            for result in results['results_per_pair']:
                if result["key"] != 'TOTAL':
                    print(f'"{result["key"]}",  // {result["profit_mean"]:.2%}')
            print("]")


def generate_edge_table(results: dict) -> str:
    floatfmt = ('s', '.10g', '.2f', '.2f', '.2f', '.2f', 'd', 'd', 'd')
    tabular_data = []
    headers = ['Pair', 'Stoploss', 'Win Rate', 'Risk Reward Ratio',
               'Required Risk Reward', 'Expectancy', 'Total Number of Trades',
               'Average Duration (min)']

    for result in results.items():
        if result[1].nb_trades > 0:
            tabular_data.append([
                result[0],
                result[1].stoploss,
                result[1].winrate,
                result[1].risk_reward_ratio,
                result[1].required_risk_reward,
                result[1].expectancy,
                result[1].nb_trades,
                round(result[1].avg_trade_duration)
            ])

    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(tabular_data, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")
