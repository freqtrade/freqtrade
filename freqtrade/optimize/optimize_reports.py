import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from pandas import DataFrame, concat, to_datetime
from tabulate import tabulate

from freqtrade.constants import (BACKTEST_BREAKDOWNS, DATETIME_PRINT_FORMAT, LAST_BT_RESULT_FN,
                                 UNLIMITED_STAKE_AMOUNT, Config, IntOrInf)
from freqtrade.data.metrics import (calculate_cagr, calculate_calmar, calculate_csum,
                                    calculate_expectancy, calculate_market_change,
                                    calculate_max_drawdown, calculate_sharpe, calculate_sortino)
from freqtrade.misc import decimals_per_coin, file_dump_joblib, file_dump_json, round_coin_value
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename


logger = logging.getLogger(__name__)


def store_backtest_stats(
        recordfilename: Path, stats: Dict[str, DataFrame], dtappendix: str) -> None:
    """
    Stores backtest results
    :param recordfilename: Path object, which can either be a filename or a directory.
        Filenames will be appended with a timestamp right before the suffix
        while for directories, <directory>/backtest-result-<datetime>.json will be used as filename
    :param stats: Dataframe containing the backtesting statistics
    :param dtappendix: Datetime to use for the filename
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f'backtest-result-{dtappendix}.json')
    else:
        filename = Path.joinpath(
            recordfilename.parent, f'{recordfilename.stem}-{dtappendix}'
        ).with_suffix(recordfilename.suffix)

    # Store metadata separately.
    file_dump_json(get_backtest_metadata_filename(filename), stats['metadata'])
    del stats['metadata']

    file_dump_json(filename, stats)

    latest_filename = Path.joinpath(filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {'latest_backtest': str(filename.name)})


def _store_backtest_analysis_data(
        recordfilename: Path, data: Dict[str, Dict],
        dtappendix: str, name: str) -> Path:
    """
    Stores backtest trade candles for analysis
    :param recordfilename: Path object, which can either be a filename or a directory.
        Filenames will be appended with a timestamp right before the suffix
        while for directories, <directory>/backtest-result-<datetime>_<name>.pkl will be used
        as filename
    :param candles: Dict containing the backtesting data for analysis
    :param dtappendix: Datetime to use for the filename
    :param name: Name to use for the file, e.g. signals, rejected
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f'backtest-result-{dtappendix}_{name}.pkl')
    else:
        filename = Path.joinpath(
            recordfilename.parent, f'{recordfilename.stem}-{dtappendix}_{name}.pkl'
        )

    file_dump_joblib(filename, data)

    return filename


def store_backtest_analysis_results(
        recordfilename: Path, candles: Dict[str, Dict], trades: Dict[str, Dict],
        dtappendix: str) -> None:
    _store_backtest_analysis_data(recordfilename, candles, dtappendix, "signals")
    _store_backtest_analysis_data(recordfilename, trades, dtappendix, "rejected")


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
        tabular_data[-1]['max_drawdown_abs'] = round_coin_value(
            result['max_drawdown_abs'], result['stake_currency'], False)
    return tabular_data


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


def _get_resample_from_period(period: str) -> str:
    if period == 'day':
        return '1d'
    if period == 'week':
        # Weekly defaulting to Monday.
        return '1W-MON'
    if period == 'month':
        return '1M'
    raise ValueError(f"Period {period} is not supported.")


def generate_periodic_breakdown_stats(trade_list: List, period: str) -> List[Dict[str, Any]]:
    results = DataFrame.from_records(trade_list)
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
        stats.append(
            {
                'date': name.strftime('%d/%m/%Y'),
                'date_ts': int(name.to_pydatetime().timestamp() * 1000),
                'profit_abs': profit_abs,
                'wins': wins,
                'draws': draws,
                'loses': loses
            }
        )
    return stats


def generate_all_periodic_breakdown_stats(trade_list: List) -> Dict[str, List]:
    result = {}
    for period in BACKTEST_BREAKDOWNS:
        result[period] = generate_periodic_breakdown_stats(trade_list, period)
    return result


def generate_trading_stats(results: DataFrame) -> Dict[str, Any]:
    """ Generate overall trade statistics """
    if len(results) == 0:
        return {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'holding_avg': timedelta(),
            'winner_holding_avg': timedelta(),
            'loser_holding_avg': timedelta(),
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

    return {
        'wins': len(winning_trades),
        'losses': len(losing_trades),
        'draws': len(draw_trades),
        'holding_avg': holding_avg,
        'holding_avg_s': holding_avg.total_seconds(),
        'winner_holding_avg': winner_holding_avg,
        'winner_holding_avg_s': winner_holding_avg.total_seconds(),
        'loser_holding_avg': loser_holding_avg,
        'loser_holding_avg_s': loser_holding_avg.total_seconds(),
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
        'expectancy': calculate_expectancy(results),
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
                            ) -> Dict[str, Any]:
    """
    :param btdata: Backtest data
    :param all_results: backtest result - dictionary in the form:
                     { Strategy: {'results: results, 'config: config}}.
    :param min_date: Backtest start date
    :param max_date: Backtest end date
    :return: Dictionary containing results per strategy and a strategy summary.
    """
    result: Dict[str, Any] = {
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
        }
        result['strategy'][strategy] = strat_stats

    strategy_results = generate_strategy_comparison(bt_stats=result['strategy'])

    result['metadata'] = metadata
    result['strategy_comparison'] = strategy_results

    return result


###
# Start output section
###

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
            ('Expectancy', f"{strat_results['expectancy']:.2f}" if 'expectancy'
                           in strat_results else 'N/A'),
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


def show_backtest_results(config: Config, backtest_stats: Dict):
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


def show_sorted_pairlist(config: Config, backtest_stats: Dict):
    if config.get('backtest_show_pair_list', False):
        for strategy, results in backtest_stats['strategy'].items():
            print(f"Pairs for Strategy {strategy}: \n[")
            for result in results['results_per_pair']:
                if result["key"] != 'TOTAL':
                    print(f'"{result["key"]}",  // {result["profit_mean"]:.2%}')
            print("]")
