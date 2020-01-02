from datetime import timedelta
from typing import Dict

from pandas import DataFrame
from tabulate import tabulate


def generate_text_table(data: Dict[str, Dict], stake_currency: str, max_open_trades: int,
                        results: DataFrame, skip_nan: bool = False) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :return: pretty printed table with tabulate as str
    """

    floatfmt = ('s', 'd', '.2f', '.2f', '.8f', '.2f', 'd', '.1f', '.1f')
    tabular_data = []
    headers = ['pair', 'buy count', 'avg profit %', 'cum profit %',
               f'tot profit {stake_currency}', 'tot profit %', 'avg duration',
               'profit', 'loss']
    for pair in data:
        result = results[results.pair == pair]
        if skip_nan and result.profit_abs.isnull().all():
            continue

        tabular_data.append([
            pair,
            len(result.index),
            result.profit_percent.mean() * 100.0,
            result.profit_percent.sum() * 100.0,
            result.profit_abs.sum(),
            result.profit_percent.sum() * 100.0 / max_open_trades,
            str(timedelta(
                minutes=round(result.trade_duration.mean()))) if not result.empty else '0:00',
            len(result[result.profit_abs > 0]),
            len(result[result.profit_abs < 0])
        ])

    # Append Total
    tabular_data.append([
        'TOTAL',
        len(results.index),
        results.profit_percent.mean() * 100.0,
        results.profit_percent.sum() * 100.0,
        results.profit_abs.sum(),
        results.profit_percent.sum() * 100.0 / max_open_trades,
        str(timedelta(
            minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
        len(results[results.profit_abs > 0]),
        len(results[results.profit_abs < 0])
    ])
    # Ignore type as floatfmt does allow tuples but mypy does not know that
    return tabulate(tabular_data, headers=headers,
                    floatfmt=floatfmt, tablefmt="pipe")  # type: ignore


def generate_text_table_sell_reason(data: Dict[str, Dict], results: DataFrame) -> str:
    """
    Generate small table outlining Backtest results
    """
    tabular_data = []
    headers = ['Sell Reason', 'Count', 'Profit', 'Loss']
    for reason, count in results['sell_reason'].value_counts().iteritems():
        profit = len(results[(results['sell_reason'] == reason) & (results['profit_abs'] >= 0)])
        loss = len(results[(results['sell_reason'] == reason) & (results['profit_abs'] < 0)])
        tabular_data.append([reason.value, count, profit, loss])
    return tabulate(tabular_data, headers=headers, tablefmt="pipe")
