import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from pandas import DataFrame
from tabulate import tabulate

from freqtrade.misc import file_dump_json

logger = logging.getLogger(__name__)


def store_backtest_result(recordfilename: Path, all_results: Dict[str, DataFrame]) -> None:
    """
    Stores backtest results to file (one file per strategy)
    :param recordfilename: Destination filename
    :param all_results: Dict of Dataframes, one results dataframe per strategy
    """
    for strategy, results in all_results.items():
        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end, t.sell_reason.value)
                   for index, t in results.iterrows()]

        if records:
            filename = recordfilename
            if len(all_results) > 1:
                # Inject strategy to filename
                filename = Path.joinpath(
                    recordfilename.parent,
                    f'{recordfilename.stem}-{strategy}').with_suffix(recordfilename.suffix)
            logger.info(f'Dumping backtest results to {filename}')
            file_dump_json(filename, records)


def _get_line_floatfmt() -> List[str]:
    """
    Generate floatformat (goes in line with _generate_result_line())
    """
    return ['s', 'd', '.2f', '.2f', '.8f', '.2f', 'd', 'd', 'd', 'd']


def _get_line_header(first_column: str, stake_currency: str) -> List[str]:
    """
    Generate header lines (goes in line with _generate_result_line())
    """
    return [first_column, 'Buys', 'Avg Profit %', 'Cum Profit %',
            f'Tot Profit {stake_currency}', 'Tot Profit %', 'Avg Duration',
            'Wins', 'Draws', 'Losses']


def _generate_result_line(result: DataFrame, max_open_trades: int, first_column: str) -> Dict:
    """
    Generate one result dict, with "first_column" as key.
    """
    return {
        'key': first_column,
        'trades': len(result.index),
        'profit_mean': result.profit_percent.mean(),
        'profit_mean_pct': result.profit_percent.mean() * 100.0,
        'profit_sum': result.profit_percent.sum(),
        'profit_sum_pct': result.profit_percent.sum() * 100.0,
        'profit_total_abs': result.profit_abs.sum(),
        'profit_total_pct': result.profit_percent.sum() * 100.0 / max_open_trades,
        'duration_avg': str(timedelta(
                            minutes=round(result.trade_duration.mean()))
                            ) if not result.empty else '0:00',
        # 'duration_max': str(timedelta(
        #                     minutes=round(result.trade_duration.max()))
        #                     ) if not result.empty else '0:00',
        # 'duration_min': str(timedelta(
        #                     minutes=round(result.trade_duration.min()))
        #                     ) if not result.empty else '0:00',
        'wins': len(result[result.profit_abs > 0]),
        'draws': len(result[result.profit_abs == 0]),
        'losses': len(result[result.profit_abs < 0]),
    }


def generate_pair_metrics(data: Dict[str, Dict], stake_currency: str, max_open_trades: int,
                          results: DataFrame, skip_nan: bool = False) -> List[Dict]:
    """
    Generates and returns a list  for the given backtest data and the results dataframe
    :param data: Dict of <pair: dataframe> containing data that was used during backtesting.
    :param stake_currency: stake-currency - used to correctly name headers
    :param max_open_trades: Maximum allowed open trades
    :param results: Dataframe containing the backtest results
    :param skip_nan: Print "left open" open trades
    :return: List of Dicts containing the metrics per pair
    """

    tabular_data = []

    for pair in data:
        result = results[results.pair == pair]
        if skip_nan and result.profit_abs.isnull().all():
            continue

        tabular_data.append(_generate_result_line(result, max_open_trades, pair))

    # Append Total
    tabular_data.append(_generate_result_line(results, max_open_trades, 'TOTAL'))
    return tabular_data


def generate_text_table(pair_results: List[Dict[str, Any]], stake_currency: str) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :param pair_results: List of Dictionaries - one entry per pair + final TOTAL row
    :param stake_currency: stake-currency - used to correctly name headers
    :return: pretty printed table with tabulate as string
    """

    headers = _get_line_header('Pair', stake_currency)
    floatfmt = _get_line_floatfmt()
    output = [[
        t['key'], t['trades'], t['profit_mean_pct'], t['profit_sum_pct'], t['profit_total_abs'],
        t['profit_total_pct'], t['duration_avg'], t['wins'], t['draws'], t['losses']
    ] for t in pair_results]
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(output, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")  # type: ignore


def generate_sell_reason_stats(max_open_trades: int, results: DataFrame) -> List[Dict]:
    """
    Generate small table outlining Backtest results
    :param max_open_trades: Max_open_trades parameter
    :param results: Dataframe containing the backtest result for one strategy
    :return: List of Dicts containing the metrics per Sell reason
    """
    tabular_data = []

    for reason, count in results['sell_reason'].value_counts().iteritems():
        result = results.loc[results['sell_reason'] == reason]

        profit_mean = result['profit_percent'].mean()
        profit_sum = result["profit_percent"].sum()
        profit_percent_tot = round(result['profit_percent'].sum() * 100.0 / max_open_trades, 2)

        tabular_data.append(
            {
                'sell_reason': reason.value,
                'trades': count,
                'wins': len(result[result['profit_abs'] > 0]),
                'draws': len(result[result['profit_abs'] == 0]),
                'losses': len(result[result['profit_abs'] < 0]),
                'profit_mean': profit_mean,
                'profit_mean_pct': round(profit_mean * 100, 2),
                'profit_sum': profit_sum,
                'profit_sum_pct': round(profit_sum * 100, 2),
                'profit_total_abs': result['profit_abs'].sum(),
                'profit_pct_total': profit_percent_tot,
            }
        )
    return tabular_data


def generate_text_table_sell_reason(sell_reason_stats: List[Dict[str, Any]],
                                    stake_currency: str) -> str:
    """
    Generate small table outlining Backtest results
    :param sell_reason_stats: Sell reason metrics
    :param stake_currency: Stakecurrency used
    :return: pretty printed table with tabulate as string
    """
    headers = [
        'Sell Reason',
        'Sells',
        'Wins',
        'Draws',
        'Losses',
        'Avg Profit %',
        'Cum Profit %',
        f'Tot Profit {stake_currency}',
        'Tot Profit %',
    ]

    output = [[
        t['sell_reason'], t['trades'], t['wins'], t['draws'], t['losses'],
        t['profit_mean_pct'], t['profit_sum_pct'], t['profit_total_abs'], t['profit_pct_total'],
     ] for t in sell_reason_stats]
    return tabulate(output, headers=headers, tablefmt="orgtbl", stralign="right")


def generate_strategy_metrics(stake_currency: str, max_open_trades: int,
                              all_results: Dict) -> List[Dict]:
    """
    Generate summary per strategy
    :param stake_currency: stake-currency - used to correctly name headers
    :param max_open_trades: Maximum allowed open trades used for backtest
    :param all_results: Dict of <Strategyname: BacktestResult> containing results for all strategies
    :return: List of Dicts containing the metrics per Strategy
    """

    tabular_data = []
    for strategy, results in all_results.items():
        tabular_data.append(_generate_result_line(results, max_open_trades, strategy))
    return tabular_data


def generate_text_table_strategy(strategy_results, stake_currency: str) -> str:
    """
    Generate summary table per strategy
    :param stake_currency: stake-currency - used to correctly name headers
    :param max_open_trades: Maximum allowed open trades used for backtest
    :param all_results: Dict of <Strategyname: BacktestResult> containing results for all strategies
    :return: pretty printed table with tabulate as string
    """
    floatfmt = _get_line_floatfmt()
    headers = _get_line_header('Strategy', stake_currency)

    output = [[
        t['key'], t['trades'], t['profit_mean_pct'], t['profit_sum_pct'], t['profit_total_abs'],
        t['profit_total_pct'], t['duration_avg'], t['wins'], t['draws'], t['losses']
    ] for t in strategy_results]
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(output, headers=headers,
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")  # type: ignore


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
                    floatfmt=floatfmt, tablefmt="orgtbl", stralign="right")  # type: ignore


def show_backtest_results(config: Dict, btdata: Dict[str, DataFrame],
                          all_results: Dict[str, DataFrame]):
    stake_currency = config['stake_currency']
    max_open_trades = config['max_open_trades']

    for strategy, results in all_results.items():
        pair_results = generate_pair_metrics(btdata, stake_currency=stake_currency,
                                             max_open_trades=max_open_trades,
                                             results=results, skip_nan=False)
        sell_reason_stats = generate_sell_reason_stats(max_open_trades=max_open_trades,
                                                       results=results)
        left_open_results = generate_pair_metrics(btdata, stake_currency=stake_currency,
                                                  max_open_trades=max_open_trades,
                                                  results=results.loc[results['open_at_end']],
                                                  skip_nan=True)
        # Print results
        print(f"Result for strategy {strategy}")
        table = generate_text_table(pair_results, stake_currency=stake_currency)
        if isinstance(table, str):
            print(' BACKTESTING REPORT '.center(len(table.splitlines()[0]), '='))
        print(table)

        table = generate_text_table_sell_reason(sell_reason_stats=sell_reason_stats,
                                                stake_currency=stake_currency,
                                                )
        if isinstance(table, str):
            print(' SELL REASON STATS '.center(len(table.splitlines()[0]), '='))
        print(table)

        table = generate_text_table(left_open_results, stake_currency=stake_currency)
        if isinstance(table, str):
            print(' LEFT OPEN TRADES REPORT '.center(len(table.splitlines()[0]), '='))
        print(table)
        if isinstance(table, str):
            print('=' * len(table.splitlines()[0]))
        print()

    if len(all_results) > 1:
        # Print Strategy summary table
        strategy_results = generate_strategy_metrics(stake_currency=stake_currency,
                                                     max_open_trades=max_open_trades,
                                                     all_results=all_results)

        table = generate_text_table_strategy(strategy_results, stake_currency)
        print(' STRATEGY SUMMARY '.center(len(table.splitlines()[0]), '='))
        print(table)
        print('=' * len(table.splitlines()[0]))
        print('\nFor more details, please look at the detail tables above')
