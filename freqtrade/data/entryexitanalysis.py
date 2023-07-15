import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from tabulate import tabulate

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.btanalysis import (get_latest_backtest_filename, load_backtest_data,
                                       load_backtest_stats)
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def _load_backtest_analysis_data(backtest_dir: Path, name: str):
    if backtest_dir.is_dir():
        scpf = Path(backtest_dir,
                    Path(get_latest_backtest_filename(backtest_dir)).stem + "_" + name + ".pkl"
                    )
    else:
        scpf = Path(backtest_dir.parent / f"{backtest_dir.stem}_{name}.pkl")

    try:
        with scpf.open("rb") as scp:
            loaded_data = joblib.load(scp)
            logger.info(f"Loaded {name} candles: {str(scpf)}")
    except Exception as e:
        logger.error(f"Cannot load {name} data from pickled results: ", e)
        return None

    return loaded_data


def _load_rejected_signals(backtest_dir: Path):
    return _load_backtest_analysis_data(backtest_dir, "rejected")


def _load_signal_candles(backtest_dir: Path):
    return _load_backtest_analysis_data(backtest_dir, "signals")


def _process_candles_and_indicators(pairlist, strategy_name, trades, signal_candles):
    analysed_trades_dict = {}
    analysed_trades_dict[strategy_name] = {}

    try:
        logger.info(f"Processing {strategy_name} : {len(pairlist)} pairs")

        for pair in pairlist:
            if pair in signal_candles[strategy_name]:
                analysed_trades_dict[strategy_name][pair] = _analyze_candles_and_indicators(
                    pair, trades, signal_candles[strategy_name][pair])
    except Exception as e:
        print(f"Cannot process entry/exit reasons for {strategy_name}: ", e)

    return analysed_trades_dict


def _analyze_candles_and_indicators(pair, trades: pd.DataFrame, signal_candles: pd.DataFrame):
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
                    raise e
        return trades_red
    else:
        return pd.DataFrame()


def _do_group_table_output(bigdf, glist, csv_path: Path, to_csv=False, ):
    for g in glist:
        # 0: summary wins/losses grouped by enter tag
        if g == "0":
            group_mask = ['enter_reason']
            wins = bigdf.loc[bigdf['profit_abs'] >= 0] \
                        .groupby(group_mask) \
                        .agg({'profit_abs': ['sum']})

            wins.columns = ['profit_abs_wins']
            loss = bigdf.loc[bigdf['profit_abs'] < 0] \
                        .groupby(group_mask) \
                        .agg({'profit_abs': ['sum']})
            loss.columns = ['profit_abs_loss']

            new = bigdf.groupby(group_mask).agg({'profit_abs': [
                                                    'count',
                                                    lambda x: sum(x > 0),
                                                    lambda x: sum(x <= 0)]})
            new = pd.concat([new, wins, loss], axis=1).fillna(0)

            new['profit_tot'] = new['profit_abs_wins'] - abs(new['profit_abs_loss'])
            new['wl_ratio_pct'] = (new.iloc[:, 1] / new.iloc[:, 0] * 100).fillna(0)
            new['avg_win'] = (new['profit_abs_wins'] / new.iloc[:, 1]).fillna(0)
            new['avg_loss'] = (new['profit_abs_loss'] / new.iloc[:, 2]).fillna(0)

            new.columns = ['total_num_buys', 'wins', 'losses', 'profit_abs_wins', 'profit_abs_loss',
                           'profit_tot', 'wl_ratio_pct', 'avg_win', 'avg_loss']

            sortcols = ['total_num_buys']

            _print_table(new, sortcols, show_index=True, name="Group 0:",
                         to_csv=to_csv, csv_path=csv_path)

        else:
            agg_mask = {'profit_abs': ['count', 'sum', 'median', 'mean'],
                        'profit_ratio': ['median', 'mean', 'sum']}
            agg_cols = ['num_buys', 'profit_abs_sum', 'profit_abs_median',
                        'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct',
                        'total_profit_pct']
            sortcols = ['profit_abs_sum', 'enter_reason']

            # 1: profit summaries grouped by enter_tag
            if g == "1":
                group_mask = ['enter_reason']

            # 2: profit summaries grouped by enter_tag and exit_tag
            if g == "2":
                group_mask = ['enter_reason', 'exit_reason']

            # 3: profit summaries grouped by pair and enter_tag
            if g == "3":
                group_mask = ['pair', 'enter_reason']

            # 4: profit summaries grouped by pair, enter_ and exit_tag (this can get quite large)
            if g == "4":
                group_mask = ['pair', 'enter_reason', 'exit_reason']

            # 5: profit summaries grouped by exit_tag
            if g == "5":
                group_mask = ['exit_reason']
                sortcols = ['exit_reason']

            if group_mask:
                new = bigdf.groupby(group_mask).agg(agg_mask).reset_index()
                new.columns = group_mask + agg_cols
                new['median_profit_pct'] = new['median_profit_pct'] * 100
                new['mean_profit_pct'] = new['mean_profit_pct'] * 100
                new['total_profit_pct'] = new['total_profit_pct'] * 100

                _print_table(new, sortcols, name=f"Group {g}:",
                             to_csv=to_csv, csv_path=csv_path)
            else:
                logger.warning("Invalid group mask specified.")


def _do_rejected_signals_output(rejected_signals_df: pd.DataFrame,
                                to_csv: bool = False, csv_path=None) -> None:
    cols = ['pair', 'date', 'enter_tag']
    sortcols = ['date', 'pair', 'enter_tag']
    _print_table(rejected_signals_df[cols],
                 sortcols,
                 show_index=False,
                 name="Rejected Signals:",
                 to_csv=to_csv,
                 csv_path=csv_path)


def _select_rows_within_dates(df, timerange=None, df_date_col: str = 'date'):
    if timerange:
        if timerange.starttype == 'date':
            df = df.loc[(df[df_date_col] >= timerange.startdt)]
        if timerange.stoptype == 'date':
            df = df.loc[(df[df_date_col] < timerange.stopdt)]
    return df


def _select_rows_by_tags(df, enter_reason_list, exit_reason_list):
    if enter_reason_list and "all" not in enter_reason_list:
        df = df.loc[(df['enter_reason'].isin(enter_reason_list))]

    if exit_reason_list and "all" not in exit_reason_list:
        df = df.loc[(df['exit_reason'].isin(exit_reason_list))]
    return df


def prepare_results(analysed_trades, stratname,
                    enter_reason_list, exit_reason_list,
                    timerange=None):
    res_df = pd.DataFrame()
    for pair, trades in analysed_trades[stratname].items():
        res_df = pd.concat([res_df, trades], ignore_index=True)

    res_df = _select_rows_within_dates(res_df, timerange)

    if res_df is not None and res_df.shape[0] > 0 and ('enter_reason' in res_df.columns):
        res_df = _select_rows_by_tags(res_df, enter_reason_list, exit_reason_list)

    return res_df


def print_results(res_df: pd.DataFrame, analysis_groups: List[str], indicator_list: List[str],
                  csv_path: Path, rejected_signals=None, to_csv=False):
    if res_df.shape[0] > 0:
        if analysis_groups:
            _do_group_table_output(res_df, analysis_groups, to_csv=to_csv, csv_path=csv_path)

        if rejected_signals is not None:
            if rejected_signals.empty:
                print("There were no rejected signals.")
            else:
                _do_rejected_signals_output(rejected_signals, to_csv=to_csv, csv_path=csv_path)

        # NB this can be large for big dataframes!
        if "all" in indicator_list:
            _print_table(res_df,
                         show_index=False,
                         name="Indicators:",
                         to_csv=to_csv,
                         csv_path=csv_path)
        elif indicator_list is not None and indicator_list:
            available_inds = []
            for ind in indicator_list:
                if ind in res_df:
                    available_inds.append(ind)
            ilist = ["pair", "enter_reason", "exit_reason"] + available_inds
            _print_table(res_df[ilist],
                         sortcols=['exit_reason'],
                         show_index=False,
                         name="Indicators:",
                         to_csv=to_csv,
                         csv_path=csv_path)
    else:
        print("\\No trades to show")


def _print_table(df: pd.DataFrame, sortcols=None, *, show_index=False, name=None,
                 to_csv=False, csv_path: Path):
    if (sortcols is not None):
        data = df.sort_values(sortcols)
    else:
        data = df

    if to_csv:
        safe_name = Path(csv_path, name.lower().replace(" ", "_").replace(":", "") + ".csv")
        data.to_csv(safe_name)
        print(f"Saved {name} to {safe_name}")
    else:
        if name is not None:
            print(name)

        print(
            tabulate(
                data,
                headers='keys',
                tablefmt='psql',
                showindex=show_index
            )
        )


def process_entry_exit_reasons(config: Config):
    try:
        analysis_groups = config.get('analysis_groups', [])
        enter_reason_list = config.get('enter_reason_list', ["all"])
        exit_reason_list = config.get('exit_reason_list', ["all"])
        indicator_list = config.get('indicator_list', [])
        do_rejected = config.get('analysis_rejected', False)
        to_csv = config.get('analysis_to_csv', False)
        csv_path = Path(config.get('analysis_csv_path', config['exportfilename']))
        if to_csv and not csv_path.is_dir():
            raise OperationalException(f"Specified directory {csv_path} does not exist.")

        timerange = TimeRange.parse_timerange(None if config.get(
            'timerange') is None else str(config.get('timerange')))

        backtest_stats = load_backtest_stats(config['exportfilename'])

        for strategy_name, results in backtest_stats['strategy'].items():
            trades = load_backtest_data(config['exportfilename'], strategy_name)

            if trades is not None and not trades.empty:
                signal_candles = _load_signal_candles(config['exportfilename'])

                rej_df = None
                if do_rejected:
                    rejected_signals_dict = _load_rejected_signals(config['exportfilename'])
                    rej_df = prepare_results(rejected_signals_dict, strategy_name,
                                             enter_reason_list, exit_reason_list,
                                             timerange=timerange)

                analysed_trades_dict = _process_candles_and_indicators(
                                        config['exchange']['pair_whitelist'], strategy_name,
                                        trades, signal_candles)

                res_df = prepare_results(analysed_trades_dict, strategy_name,
                                         enter_reason_list, exit_reason_list,
                                         timerange=timerange)

                print_results(res_df,
                              analysis_groups,
                              indicator_list,
                              rejected_signals=rej_df,
                              to_csv=to_csv,
                              csv_path=csv_path)

    except ValueError as e:
        raise OperationalException(e) from e
