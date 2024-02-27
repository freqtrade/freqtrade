import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from pandas import DataFrame, Series, concat, to_datetime

from freqtrade.constants import BACKTEST_BREAKDOWNS, DATETIME_PRINT_FORMAT, IntOrInf
from freqtrade.data.metrics import (calculate_cagr, calculate_calmar, calculate_csum,
                                    calculate_expectancy, calculate_market_change,
                                    calculate_max_drawdown, calculate_sharpe, calculate_sortino)
from freqtrade.types import BacktestResultType
from freqtrade.util import decimals_per_coin, fmt_coin


logger = logging.getLogger(__name__)


def generate_trade_signal_candles(preprocessed_df: Dict[str, DataFrame],
                                  bt_results: Dict[str, Any]) -> DataFrame:
    signal_candles_only = {}
    for pair in preprocessed_df.keys():
        signal_candles_only_df = DataFrame()

        pairdf = preprocessed_df[pair]
        resdf = bt_results['results']
        pairresults = resdf.loc[(resdf["pair"] == pair)]

        if pairdf.shape[0] > 0:
            for t, v in pairresults.open_date.items():
                allinds = pairdf.loc[(pairdf['date'] < v)]
                signal_inds = allinds.iloc[[-1]]
                signal_candles_only_df = concat([
                    signal_candles_only_df.infer_objects(),
                    signal_inds.infer_objects()])

            signal_candles_only[pair] = signal_candles_only_df
    return signal_candles_only


def generate_rejected_signals(preprocessed_df: Dict[str, DataFrame],
                              rejected_dict: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    rejected_candles_only = {}
    for pair, signals in rejected_dict.items():
        rejected_signals_only_df = DataFrame()
        pairdf = preprocessed_df[pair]

        for t in signals:
            data_df_row = pairdf.loc[(pairdf['date'] == t[0])].copy()
            data_df_row['pair'] = pair
            data_df_row['enter_tag'] = t[1]

            rejected_signals_only_df = concat([
                rejected_signals_only_df.infer_objects(),
                data_df_row.infer_objects()])

        rejected_candles_only[pair] = rejected_signals_only_df
    return rejected_candles_only


def _generate_result_line(result: DataFrame, starting_balance: int, first_column: str) -> Dict:
    """
    Generate one result dict, with "first_column" as key.
    """
    profit_sum = result['profit_ratio'].sum()
    # (end-capital - starting capital) / starting capital
    profit_total = result['profit_abs'].sum() / starting_balance

    return {
        'key': first_column,
        'trades': len(result),
        'profit_mean': result['profit_ratio'].mean() if len(result) > 0 else 0.0,
        'profit_mean_pct': result['profit_ratio'].mean() * 100.0 if len(result) > 0 else 0.0,
        'profit_sum': profit_sum,
        'profit_sum_pct': round(profit_sum * 100.0, 2),
        'profit_total_abs': result['profit_abs'].sum(),
        'profit_total': profit_total,
        'profit_total_pct': round(profit_total * 100.0, 2),
        'duration_avg': str(timedelta(
                            minutes=round(result['trade_duration'].mean()))
                            ) if not result.empty else '0:00',
        # 'duration_max': str(timedelta(
        #                     minutes=round(result['trade_duration'].max()))
        #                     ) if not result.empty else '0:00',
        # 'duration_min': str(timedelta(
        #                     minutes=round(result['trade_duration'].min()))
        #                     ) if not result.empty else '0:00',
        'wins': len(result[result['profit_abs'] > 0]),
        'draws': len(result[result['profit_abs'] == 0]),
        'losses': len(result[result['profit_abs'] < 0]),
        'winrate': len(result[result['profit_abs'] > 0]) / len(result) if len(result) else 0.0,
    }


def generate_pair_metrics(pairlist: List[str], stake_currency: str, starting_balance: int,
                          results: DataFrame, skip_nan: bool = False) -> List[Dict]:
    """
    Generates and returns a list  for the given backtest data and the results dataframe
    :param pairlist: Pairlist used
    :param stake_currency: stake-currency - used to correctly name headers
    :param starting_balance: Starting balance
    :param results: Dataframe containing the backtest results
    :param skip_nan: Print "left open" open trades
    :return: List of Dicts containing the metrics per pair
    """

    tabular_data = []

    for pair in pairlist:
        result = results[results['pair'] == pair]
        if skip_nan and result['profit_abs'].isnull().all():
            continue

        tabular_data.append(_generate_result_line(result, starting_balance, pair))

    # Sort by total profit %:
    tabular_data = sorted(tabular_data, key=lambda k: k['profit_total_abs'], reverse=True)

    # Append Total
    tabular_data.append(_generate_result_line(results, starting_balance, 'TOTAL'))
    return tabular_data


def generate_tag_metrics(tag_type: str,
                         starting_balance: int,
                         results: DataFrame,
                         skip_nan: bool = False) -> List[Dict]:
    """
    Generates and returns a list of metrics for the given tag trades and the results dataframe
    :param starting_balance: Starting balance
    :param results: Dataframe containing the backtest results
    :param skip_nan: Print "left open" open trades
    :return: List of Dicts containing the metrics per pair
    """

    tabular_data = []

    if tag_type in results.columns:
        for tag, count in results[tag_type].value_counts().items():
            result = results[results[tag_type] == tag]
            if skip_nan and result['profit_abs'].isnull().all():
                continue

            tabular_data.append(_generate_result_line(result, starting_balance, tag))

        # Sort by total profit %:
        tabular_data = sorted(tabular_data, key=lambda k: k['profit_total_abs'], reverse=True)

        # Append Total
        tabular_data.append(_generate_result_line(results, starting_balance, 'TOTAL'))
        return tabular_data
    else:
        return []


def generate_exit_reason_stats(max_open_trades: IntOrInf, results: DataFrame) -> List[Dict]:
    """
    Generate small table outlining Backtest results
    :param max_open_trades: Max_open_trades parameter
    :param results: Dataframe containing the backtest result for one strategy
    :return: List of Dicts containing the metrics per Sell reason
    """
    tabular_data = []

    for reason, count in results['exit_reason'].value_counts().items():
        result = results.loc[results['exit_reason'] == reason]

        profit_mean = result['profit_ratio'].mean()
        profit_sum = result['profit_ratio'].sum()
        profit_total = profit_sum / max_open_trades

        tabular_data.append(
            {
                'exit_reason': reason,
                'trades': count,
                'wins': len(result[result['profit_abs'] > 0]),
                'draws': len(result[result['profit_abs'] == 0]),
                'losses': len(result[result['profit_abs'] < 0]),
                'winrate': len(result[result['profit_abs'] > 0]) / count if count else 0.0,
                'profit_mean': profit_mean,
                'profit_mean_pct': round(profit_mean * 100, 2),
                'profit_sum': profit_sum,
                'profit_sum_pct': round(profit_sum * 100, 2),
                'profit_total_abs': result['profit_abs'].sum(),
                'profit_total': profit_total,
                'profit_total_pct': round(profit_total * 100, 2),
            }
        )
    return tabular_data


def generate_strategy_comparison(bt_stats: Dict) -> List[Dict]:
    """
    Generate summary per strategy
    :param bt_stats: Dict of <Strategyname: DataFrame> containing results for all strategies
    :return: List of Dicts containing the metrics per Strategy
    """

    tabular_data = []
    for strategy, result in bt_stats.items():
        tabular_data.append(deepcopy(result['results_per_pair'][-1]))
        # Update "key" to strategy (results_per_pair has it as "Total").
        tabular_data[-1]['key'] = strategy
        tabular_data[-1]['max_drawdown_account'] = result['max_drawdown_account']
        tabular_data[-1]['max_drawdown_abs'] = fmt_coin(
            result['max_drawdown_abs'], result['stake_currency'], False)
    return tabular_data


def _get_resample_from_period(period: str) -> str:
    if period == 'day':
        return '1d'
    if period == 'week':
        # Weekly defaulting to Monday.
        return '1W-MON'
    if period == 'month':
        return '1ME'
    raise ValueError(f"Period {period} is not supported.")


def generate_periodic_breakdown_stats(
        trade_list: Union[List,  DataFrame], period: str) -> List[Dict[str, Any]]:

    results = trade_list if not isinstance(trade_list, list) else DataFrame.from_records(trade_list)
    if len(results) == 0:
        return []
    results['close_date'] = to_datetime(results['close_date'], utc=True)
    resample_period = _get_resample_from_period(period)
    resampled = results.resample(resample_period, on='close_date')
    stats = []
    for name, day in resampled:
        profit_abs = day['profit_abs'].sum().round(10)
        wins = sum(day['profit_abs'] > 0)
        draws = sum(day['profit_abs'] == 0)
        loses = sum(day['profit_abs'] < 0)
        trades = (wins + draws + loses)
        stats.append(
            {
                'date': name.strftime('%d/%m/%Y'),
                'date_ts': int(name.to_pydatetime().timestamp() * 1000),
                'profit_abs': profit_abs,
                'wins': wins,
                'draws': draws,
                'loses': loses,
                'winrate': wins / trades if trades else 0.0,
            }
        )
    return stats


def generate_all_periodic_breakdown_stats(trade_list: List) -> Dict[str, List]:
    result = {}
    for period in BACKTEST_BREAKDOWNS:
        result[period] = generate_periodic_breakdown_stats(trade_list, period)
    return result


def calc_streak(dataframe: DataFrame) -> Tuple[int, int]:
    """
    Calculate consecutive win and loss streaks
    :param dataframe: Dataframe containing the trades dataframe, with profit_ratio column
    :return: Tuple containing consecutive wins and losses
    """

    df = Series(np.where(dataframe['profit_ratio'] > 0, 'win', 'loss')).to_frame('result')
    df['streaks'] = df['result'].ne(df['result'].shift()).cumsum().rename('streaks')
    df['counter'] = df['streaks'].groupby(df['streaks']).cumcount() + 1
    res = df.groupby(df['result']).max()
    #
    cons_wins = int(res.loc['win', 'counter']) if 'win' in res.index else 0
    cons_losses = int(res.loc['loss', 'counter']) if 'loss' in res.index else 0
    return cons_wins, cons_losses


def generate_trading_stats(results: DataFrame) -> Dict[str, Any]:
    """ Generate overall trade statistics """
    if len(results) == 0:
        return {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'winrate': 0,
            'holding_avg': timedelta(),
            'winner_holding_avg': timedelta(),
            'loser_holding_avg': timedelta(),
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
        }

    winning_trades = results.loc[results['profit_ratio'] > 0]
    draw_trades = results.loc[results['profit_ratio'] == 0]
    losing_trades = results.loc[results['profit_ratio'] < 0]

    holding_avg = (timedelta(minutes=round(results['trade_duration'].mean()))
                   if not results.empty else timedelta())
    winner_holding_avg = (timedelta(minutes=round(winning_trades['trade_duration'].mean()))
                          if not winning_trades.empty else timedelta())
    loser_holding_avg = (timedelta(minutes=round(losing_trades['trade_duration'].mean()))
                         if not losing_trades.empty else timedelta())
    winstreak, loss_streak = calc_streak(results)

    return {
        'wins': len(winning_trades),
        'losses': len(losing_trades),
        'draws': len(draw_trades),
        'winrate': len(winning_trades) / len(results) if len(results) else 0.0,
        'holding_avg': holding_avg,
        'holding_avg_s': holding_avg.total_seconds(),
        'winner_holding_avg': winner_holding_avg,
        'winner_holding_avg_s': winner_holding_avg.total_seconds(),
        'loser_holding_avg': loser_holding_avg,
        'loser_holding_avg_s': loser_holding_avg.total_seconds(),
        'max_consecutive_wins': winstreak,
        'max_consecutive_losses': loss_streak,
    }


def generate_daily_stats(results: DataFrame) -> Dict[str, Any]:
    """ Generate daily statistics """
    if len(results) == 0:
        return {
            'backtest_best_day': 0,
            'backtest_worst_day': 0,
            'backtest_best_day_abs': 0,
            'backtest_worst_day_abs': 0,
            'winning_days': 0,
            'draw_days': 0,
            'losing_days': 0,
            'daily_profit_list': [],
        }
    daily_profit_rel = results.resample('1d', on='close_date')['profit_ratio'].sum()
    daily_profit = results.resample('1d', on='close_date')['profit_abs'].sum().round(10)
    worst_rel = min(daily_profit_rel)
    best_rel = max(daily_profit_rel)
    worst = min(daily_profit)
    best = max(daily_profit)
    winning_days = sum(daily_profit > 0)
    draw_days = sum(daily_profit == 0)
    losing_days = sum(daily_profit < 0)
    daily_profit_list = [(str(idx.date()), val) for idx, val in daily_profit.items()]

    return {
        'backtest_best_day': best_rel,
        'backtest_worst_day': worst_rel,
        'backtest_best_day_abs': best,
        'backtest_worst_day_abs': worst,
        'winning_days': winning_days,
        'draw_days': draw_days,
        'losing_days': losing_days,
        'daily_profit': daily_profit_list,
    }


def generate_strategy_stats(pairlist: List[str],
                            strategy: str,
                            content: Dict[str, Any],
                            min_date: datetime, max_date: datetime,
                            market_change: float,
                            is_hyperopt: bool = False,
                            ) -> Dict[str, Any]:
    """
    :param pairlist: List of pairs to backtest
    :param strategy: Strategy name
    :param content: Backtest result data in the format:
                    {'results: results, 'config: config}}.
    :param min_date: Backtest start date
    :param max_date: Backtest end date
    :param market_change: float indicating the market change
    :return: Dictionary containing results per strategy and a strategy summary.
    """
    results: Dict[str, DataFrame] = content['results']
    if not isinstance(results, DataFrame):
        return {}
    config = content['config']
    max_open_trades = min(config['max_open_trades'], len(pairlist))
    start_balance = config['dry_run_wallet']
    stake_currency = config['stake_currency']

    pair_results = generate_pair_metrics(pairlist, stake_currency=stake_currency,
                                         starting_balance=start_balance,
                                         results=results, skip_nan=False)

    enter_tag_results = generate_tag_metrics("enter_tag", starting_balance=start_balance,
                                             results=results, skip_nan=False)

    exit_reason_stats = generate_exit_reason_stats(max_open_trades=max_open_trades,
                                                   results=results)
    left_open_results = generate_pair_metrics(
        pairlist, stake_currency=stake_currency, starting_balance=start_balance,
        results=results.loc[results['exit_reason'] == 'force_exit'], skip_nan=True)

    daily_stats = generate_daily_stats(results)
    trade_stats = generate_trading_stats(results)

    periodic_breakdown = {}
    if not is_hyperopt:
        periodic_breakdown = {'periodic_breakdown': generate_all_periodic_breakdown_stats(results)}

    best_pair = max([pair for pair in pair_results if pair['key'] != 'TOTAL'],
                    key=lambda x: x['profit_sum']) if len(pair_results) > 1 else None
    worst_pair = min([pair for pair in pair_results if pair['key'] != 'TOTAL'],
                     key=lambda x: x['profit_sum']) if len(pair_results) > 1 else None
    winning_profit = results.loc[results['profit_abs'] > 0, 'profit_abs'].sum()
    losing_profit = results.loc[results['profit_abs'] < 0, 'profit_abs'].sum()
    profit_factor = winning_profit / abs(losing_profit) if losing_profit else 0.0

    expectancy, expectancy_ratio = calculate_expectancy(results)
    backtest_days = (max_date - min_date).days or 1
    strat_stats = {
        'trades': results.to_dict(orient='records'),
        'locks': [lock.to_json() for lock in content['locks']],
        'best_pair': best_pair,
        'worst_pair': worst_pair,
        'results_per_pair': pair_results,
        'results_per_enter_tag': enter_tag_results,
        'exit_reason_summary': exit_reason_stats,
        'left_open_trades': left_open_results,

        'total_trades': len(results),
        'trade_count_long': len(results.loc[~results['is_short']]),
        'trade_count_short': len(results.loc[results['is_short']]),
        'total_volume': float(results['stake_amount'].sum()),
        'avg_stake_amount': results['stake_amount'].mean() if len(results) > 0 else 0,
        'profit_mean': results['profit_ratio'].mean() if len(results) > 0 else 0,
        'profit_median': results['profit_ratio'].median() if len(results) > 0 else 0,
        'profit_total': results['profit_abs'].sum() / start_balance,
        'profit_total_long': results.loc[~results['is_short'], 'profit_abs'].sum() / start_balance,
        'profit_total_short': results.loc[results['is_short'], 'profit_abs'].sum() / start_balance,
        'profit_total_abs': results['profit_abs'].sum(),
        'profit_total_long_abs': results.loc[~results['is_short'], 'profit_abs'].sum(),
        'profit_total_short_abs': results.loc[results['is_short'], 'profit_abs'].sum(),
        'cagr': calculate_cagr(backtest_days, start_balance, content['final_balance']),
        'expectancy': expectancy,
        'expectancy_ratio': expectancy_ratio,
        'sortino': calculate_sortino(results, min_date, max_date, start_balance),
        'sharpe': calculate_sharpe(results, min_date, max_date, start_balance),
        'calmar': calculate_calmar(results, min_date, max_date, start_balance),
        'profit_factor': profit_factor,
        'backtest_start': min_date.strftime(DATETIME_PRINT_FORMAT),
        'backtest_start_ts': int(min_date.timestamp() * 1000),
        'backtest_end': max_date.strftime(DATETIME_PRINT_FORMAT),
        'backtest_end_ts': int(max_date.timestamp() * 1000),
        'backtest_days': backtest_days,

        'backtest_run_start_ts': content['backtest_start_time'],
        'backtest_run_end_ts': content['backtest_end_time'],

        'trades_per_day': round(len(results) / backtest_days, 2),
        'market_change': market_change,
        'pairlist': pairlist,
        'stake_amount': config['stake_amount'],
        'stake_currency': config['stake_currency'],
        'stake_currency_decimals': decimals_per_coin(config['stake_currency']),
        'starting_balance': start_balance,
        'dry_run_wallet': start_balance,
        'final_balance': content['final_balance'],
        'rejected_signals': content['rejected_signals'],
        'timedout_entry_orders': content['timedout_entry_orders'],
        'timedout_exit_orders': content['timedout_exit_orders'],
        'canceled_trade_entries': content['canceled_trade_entries'],
        'canceled_entry_orders': content['canceled_entry_orders'],
        'replaced_entry_orders': content['replaced_entry_orders'],
        'max_open_trades': max_open_trades,
        'max_open_trades_setting': (config['max_open_trades']
                                    if config['max_open_trades'] != float('inf') else -1),
        'timeframe': config['timeframe'],
        'timeframe_detail': config.get('timeframe_detail', ''),
        'timerange': config.get('timerange', ''),
        'enable_protections': config.get('enable_protections', False),
        'strategy_name': strategy,
        # Parameters relevant for backtesting
        'stoploss': config['stoploss'],
        'trailing_stop': config.get('trailing_stop', False),
        'trailing_stop_positive': config.get('trailing_stop_positive'),
        'trailing_stop_positive_offset': config.get('trailing_stop_positive_offset', 0.0),
        'trailing_only_offset_is_reached': config.get('trailing_only_offset_is_reached', False),
        'use_custom_stoploss': config.get('use_custom_stoploss', False),
        'minimal_roi': config['minimal_roi'],
        'use_exit_signal': config['use_exit_signal'],
        'exit_profit_only': config['exit_profit_only'],
        'exit_profit_offset': config['exit_profit_offset'],
        'ignore_roi_if_entry_signal': config['ignore_roi_if_entry_signal'],
        **periodic_breakdown,
        **daily_stats,
        **trade_stats
    }

    try:
        max_drawdown_legacy, _, _, _, _, _ = calculate_max_drawdown(
            results, value_col='profit_ratio')
        (drawdown_abs, drawdown_start, drawdown_end, high_val, low_val,
         max_drawdown) = calculate_max_drawdown(
             results, value_col='profit_abs', starting_balance=start_balance)
        # max_relative_drawdown = Underwater
        (_, _, _, _, _, max_relative_drawdown) = calculate_max_drawdown(
             results, value_col='profit_abs', starting_balance=start_balance, relative=True)

        strat_stats.update({
            'max_drawdown': max_drawdown_legacy,  # Deprecated - do not use
            'max_drawdown_account': max_drawdown,
            'max_relative_drawdown': max_relative_drawdown,
            'max_drawdown_abs': drawdown_abs,
            'drawdown_start': drawdown_start.strftime(DATETIME_PRINT_FORMAT),
            'drawdown_start_ts': drawdown_start.timestamp() * 1000,
            'drawdown_end': drawdown_end.strftime(DATETIME_PRINT_FORMAT),
            'drawdown_end_ts': drawdown_end.timestamp() * 1000,

            'max_drawdown_low': low_val,
            'max_drawdown_high': high_val,
        })

        csum_min, csum_max = calculate_csum(results, start_balance)
        strat_stats.update({
            'csum_min': csum_min,
            'csum_max': csum_max
        })

    except ValueError:
        strat_stats.update({
            'max_drawdown': 0.0,
            'max_drawdown_account': 0.0,
            'max_relative_drawdown': 0.0,
            'max_drawdown_abs': 0.0,
            'max_drawdown_low': 0.0,
            'max_drawdown_high': 0.0,
            'drawdown_start': datetime(1970, 1, 1, tzinfo=timezone.utc),
            'drawdown_start_ts': 0,
            'drawdown_end': datetime(1970, 1, 1, tzinfo=timezone.utc),
            'drawdown_end_ts': 0,
            'csum_min': 0,
            'csum_max': 0
        })

    return strat_stats


def generate_backtest_stats(btdata: Dict[str, DataFrame],
                            all_results: Dict[str, Dict[str, Union[DataFrame, Dict]]],
                            min_date: datetime, max_date: datetime
                            ) -> BacktestResultType:
    """
    :param btdata: Backtest data
    :param all_results: backtest result - dictionary in the form:
                     { Strategy: {'results: results, 'config: config}}.
    :param min_date: Backtest start date
    :param max_date: Backtest end date
    :return: Dictionary containing results per strategy and a strategy summary.
    """
    result: BacktestResultType = {
        'metadata': {},
        'strategy': {},
        'strategy_comparison': [],
    }
    market_change = calculate_market_change(btdata, 'close')
    metadata = {}
    pairlist = list(btdata.keys())
    for strategy, content in all_results.items():
        strat_stats = generate_strategy_stats(pairlist, strategy, content,
                                              min_date, max_date, market_change=market_change)
        metadata[strategy] = {
            'run_id': content['run_id'],
            'backtest_start_time': content['backtest_start_time'],
            'timeframe': content['config']['timeframe'],
            'timeframe_detail': content['config'].get('timeframe_detail', None),
            'backtest_start_ts': int(min_date.timestamp()),
            'backtest_end_ts': int(max_date.timestamp()),
        }
        result['strategy'][strategy] = strat_stats

    strategy_results = generate_strategy_comparison(bt_stats=result['strategy'])

    result['metadata'] = metadata
    result['strategy_comparison'] = strategy_results

    return result
