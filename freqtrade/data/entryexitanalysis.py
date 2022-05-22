import joblib
import logging
import os

from pathlib import Path
from typing import List, Optional

import pandas as pd
from tabulate import tabulate

from freqtrade.data.btanalysis import (load_backtest_data, get_latest_backtest_filename)
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def _load_signal_candles(backtest_dir: Path):
    scpf = Path(backtest_dir,
                os.path.splitext(
                    get_latest_backtest_filename(backtest_dir))[0] + "_signals.pkl"
                )
    try:
        scp = open(scpf, "rb")
        signal_candles = joblib.load(scp)
        logger.info(f"Loaded signal candles: {str(scpf)}")
    except Exception as e:
        logger.error("Cannot load signal candles from pickled results: ", e)

    return signal_candles


def _process_candles_and_indicators(pairlist, strategy_name, trades, signal_candles):
    analysed_trades_dict = {}
    analysed_trades_dict[strategy_name] = {}

    try:
        logger.info(f"Processing {strategy_name} : {len(pairlist)} pairs")

        for pair in pairlist:
            if pair in signal_candles[strategy_name]:
                analysed_trades_dict[strategy_name][pair] = _analyze_candles_and_indicators(
                                                              pair,
                                                              trades,
                                                              signal_candles[strategy_name][pair])
    except Exception:
        pass

    return analysed_trades_dict


def _analyze_candles_and_indicators(pair, trades, signal_candles):
    buyf = signal_candles

    if len(buyf) > 0:
        buyf = buyf.set_index('date', drop=False)
        trades_red = trades.loc[trades['pair'] == pair].copy()

        trades_inds = pd.DataFrame()

        if trades_red.shape[0] > 0 and buyf.shape[0] > 0:
            for t, v in trades_red.open_date.items():
                allinds = buyf.loc[(buyf['date'] < v)]
                if allinds.shape[0] > 0:
                    tmp_inds = allinds.iloc[[-1]]

                    trades_red.loc[t, 'signal_date'] = tmp_inds['date'].values[0]
                    trades_red.loc[t, 'enter_reason'] = trades_red.loc[t, 'enter_tag']
                    tmp_inds.index.rename('signal_date', inplace=True)
                    trades_inds = pd.concat([trades_inds, tmp_inds])

            if 'signal_date' in trades_red:
                trades_red['signal_date'] = pd.to_datetime(trades_red['signal_date'], utc=True)
                trades_red.set_index('signal_date', inplace=True)

                try:
                    trades_red = pd.merge(trades_red, trades_inds, on='signal_date', how='outer')
                except Exception as e:
                    print(e)
        return trades_red
    else:
        return pd.DataFrame()


def _do_group_table_output(bigdf, glist):
    if "0" in glist:
        wins = bigdf.loc[bigdf['profit_abs'] >= 0] \
                    .groupby(['enter_reason']) \
                    .agg({'profit_abs': ['sum']})

        wins.columns = ['profit_abs_wins']
        loss = bigdf.loc[bigdf['profit_abs'] < 0] \
                    .groupby(['enter_reason']) \
                    .agg({'profit_abs': ['sum']})
        loss.columns = ['profit_abs_loss']

        new = bigdf.groupby(['enter_reason']).agg({'profit_abs': [
                                                   'count',
                                                   lambda x: sum(x > 0),
                                                   lambda x: sum(x <= 0)]})

        new = pd.merge(new, wins, left_index=True, right_index=True)
        new = pd.merge(new, loss, left_index=True, right_index=True)

        new['profit_tot'] = new['profit_abs_wins'] - abs(new['profit_abs_loss'])

        new['wl_ratio_pct'] = (new.iloc[:, 1] / new.iloc[:, 0] * 100)
        new['avg_win'] = (new['profit_abs_wins'] / new.iloc[:, 1])
        new['avg_loss'] = (new['profit_abs_loss'] / new.iloc[:, 2])

        new.columns = ['total_num_buys', 'wins', 'losses', 'profit_abs_wins', 'profit_abs_loss',
                       'profit_tot', 'wl_ratio_pct', 'avg_win', 'avg_loss']

        sortcols = ['total_num_buys']

        _print_table(new, sortcols, show_index=True)
    if "1" in glist:
        new = bigdf.groupby(['enter_reason']) \
                   .agg({'profit_abs': ['count', 'sum', 'median', 'mean'],
                         'profit_ratio': ['sum', 'median', 'mean']}
                        ).reset_index()
        new.columns = ['enter_reason', 'num_buys', 'profit_abs_sum', 'profit_abs_median',
                       'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct',
                       'total_profit_pct']
        sortcols = ['profit_abs_sum', 'enter_reason']

        new['median_profit_pct'] = new['median_profit_pct'] * 100
        new['mean_profit_pct'] = new['mean_profit_pct'] * 100
        new['total_profit_pct'] = new['total_profit_pct'] * 100

        _print_table(new, sortcols)
    if "2" in glist:
        new = bigdf.groupby(['enter_reason', 'exit_reason']) \
                   .agg({'profit_abs': ['count', 'sum', 'median', 'mean'],
                         'profit_ratio': ['sum', 'median', 'mean']}
                        ).reset_index()
        new.columns = ['enter_reason', 'exit_reason', 'num_buys', 'profit_abs_sum',
                       'profit_abs_median', 'profit_abs_mean', 'median_profit_pct',
                       'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'enter_reason']

        new['median_profit_pct'] = new['median_profit_pct'] * 100
        new['mean_profit_pct'] = new['mean_profit_pct'] * 100
        new['total_profit_pct'] = new['total_profit_pct'] * 100

        _print_table(new, sortcols)
    if "3" in glist:
        new = bigdf.groupby(['pair', 'enter_reason']) \
                   .agg({'profit_abs': ['count', 'sum', 'median', 'mean'],
                        'profit_ratio': ['sum', 'median', 'mean']}
                        ).reset_index()
        new.columns = ['pair', 'enter_reason', 'num_buys', 'profit_abs_sum',
                       'profit_abs_median', 'profit_abs_mean', 'median_profit_pct',
                       'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'enter_reason']

        new['median_profit_pct'] = new['median_profit_pct'] * 100
        new['mean_profit_pct'] = new['mean_profit_pct'] * 100
        new['total_profit_pct'] = new['total_profit_pct'] * 100

        _print_table(new, sortcols)
    if "4" in glist:
        new = bigdf.groupby(['pair', 'enter_reason', 'exit_reason']) \
                   .agg({'profit_abs': ['count', 'sum', 'median', 'mean'],
                         'profit_ratio': ['sum', 'median', 'mean']}
                        ).reset_index()
        new.columns = ['pair', 'enter_reason', 'exit_reason', 'num_buys', 'profit_abs_sum',
                       'profit_abs_median', 'profit_abs_mean', 'median_profit_pct',
                       'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'enter_reason']

        new['median_profit_pct'] = new['median_profit_pct'] * 100
        new['mean_profit_pct'] = new['mean_profit_pct'] * 100
        new['total_profit_pct'] = new['total_profit_pct'] * 100

        _print_table(new, sortcols)


def _print_results(analysed_trades, stratname, group,
                   enter_reason_list, exit_reason_list,
                   indicator_list, columns=None):

    if columns is None:
        columns = ['pair', 'open_date', 'close_date', 'profit_abs', 'enter_reason', 'exit_reason']

    bigdf = pd.DataFrame()
    for pair, trades in analysed_trades[stratname].items():
        bigdf = pd.concat([bigdf, trades], ignore_index=True)

    if bigdf.shape[0] > 0 and ('enter_reason' in bigdf.columns):
        if group is not None:
            glist = group.split(",")
            _do_group_table_output(bigdf, glist)

        if enter_reason_list is not None and not enter_reason_list == "all":
            enter_reason_list = enter_reason_list.split(",")
            bigdf = bigdf.loc[(bigdf['enter_reason'].isin(enter_reason_list))]

        if exit_reason_list is not None and not exit_reason_list == "all":
            exit_reason_list = exit_reason_list.split(",")
            bigdf = bigdf.loc[(bigdf['exit_reason'].isin(exit_reason_list))]

        if indicator_list is not None:
            if indicator_list == "all":
                print(bigdf)
            else:
                available_inds = []
                for ind in indicator_list.split(","):
                    if ind in bigdf:
                        available_inds.append(ind)
                ilist = ["pair", "enter_reason", "exit_reason"] + available_inds
                print(tabulate(bigdf[ilist].sort_values(['exit_reason']),
                      headers='keys', tablefmt='psql', showindex=False))
        else:
            print(tabulate(bigdf[columns].sort_values(['pair']),
                  headers='keys', tablefmt='psql', showindex=False))
    else:
        print("\\_ No trades to show")


def _print_table(df, sortcols=None, show_index=False):
    if (sortcols is not None):
        data = df.sort_values(sortcols)
    else:
        data = df

    print(
        tabulate(
            data,
            headers='keys',
            tablefmt='psql',
            showindex=show_index
        )
    )


def process_entry_exit_reasons(backtest_dir: Path,
                               pairlist: List[str],
                               strategy_name: str,
                               analysis_groups: Optional[str] = "0,1,2",
                               enter_reason_list: Optional[str] = "all",
                               exit_reason_list: Optional[str] = "all",
                               indicator_list: Optional[str] = None):

    try:
        trades = load_backtest_data(backtest_dir, strategy_name)
    except ValueError as e:
        raise OperationalException(e) from e
    if not trades.empty:
        signal_candles = _load_signal_candles(backtest_dir)
        analysed_trades_dict = _process_candles_and_indicators(pairlist, strategy_name,
                                                               trades, signal_candles)
        _print_results(analysed_trades_dict,
                       strategy_name,
                       analysis_groups,
                       enter_reason_list,
                       exit_reason_list,
                       indicator_list)
