# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pandas

from freqtrade.configuration import TimeRange
from freqtrade.data.history import get_timerange
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.optimize.backtesting import Backtesting


class backtest_lookahead_bias_checker:
    class varHolder:
        timerange: TimeRange
        data: pandas.DataFrame
        indicators: pandas.DataFrame
        result: pandas.DataFrame
        compared: pandas.DataFrame
        from_dt: datetime
        to_dt: datetime
        compared_dt: datetime

    class analysis:
        def __init__(self):
            self.total_signals = 0
            self.false_entry_signals = 0
            self.false_exit_signals = 0
            self.false_indicators = []
            self.has_bias = False

        total_signals: int
        false_entry_signals: int
        false_exit_signals: int

        false_indicators: list
        has_bias: bool

    def __init__(self):
        self.strategy_obj
        self.current_analysis
        self.config
        self.full_varHolder
        self.entry_varholder
        self.exit_varholder
        self.backtesting
        self.signals_to_check: int = 20
        self.current_analysis
        self.full_varHolder.from_dt
        self.full_varHolder.to_dt

    @staticmethod
    def dt_to_timestamp(dt):
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    def get_result(self, backtesting, processed):
        min_date, max_date = get_timerange(processed)

        result = backtesting.backtest(
            processed=deepcopy(processed),
            start_date=min_date,
            end_date=max_date
        )
        return result

    # analyzes two data frames with processed indicators and shows differences between them.
    def analyze_indicators(self, full_vars, cut_vars, current_pair):
        # extract dataframes
        cut_df = cut_vars.indicators[current_pair]
        full_df = full_vars.indicators[current_pair]

        # cut longer dataframe to length of the shorter
        full_df_cut = full_df[
            (full_df.date == cut_vars.compared_dt)
        ].reset_index(drop=True)
        cut_df_cut = cut_df[
            (cut_df.date == cut_vars.compared_dt)
        ].reset_index(drop=True)

        # compare dataframes
        if full_df_cut.shape[0] != 0:
            if cut_df_cut.shape[0] != 0:
                compare_df = full_df_cut.compare(cut_df_cut)

                # skippedColumns = ["date", "open", "high", "low", "close", "volume"]
                for col_name, values in compare_df.items():
                    col_idx = compare_df.columns.get_loc(col_name)
                    compare_df_row = compare_df.iloc[0]
                    # compare_df now is comprised of tuples with [1] having either 'self' or 'other'
                    if 'other' in col_name[1]:
                        continue
                    self_value = compare_df_row[col_idx]
                    other_value = compare_df_row[col_idx + 1]
                    other_value = compare_df_row[col_idx + 1]

                    # output differences
                    if self_value != other_value:

                        if not self.current_analysis.false_indicators.__contains__(col_name[0]):
                            self.current_analysis.false_indicators.append(col_name[0])
                            print(f"=> found look ahead bias in indicator {col_name[0]}. " +
                                  f"{str(self_value)} != {str(other_value)}")
                # return compare_df

    def report_signal(self, result, column_name, checked_timestamp):
        df = result['results']
        row_count = df[column_name].shape[0]

        if row_count == 0:
            return False
        else:

            df_cut = df[(df[column_name] == checked_timestamp)]
            if df_cut[column_name].shape[0] == 0:
                # print("did NOT find the same signal in column " + column_name +
                #       " at timestamp " + str(checked_timestamp))
                return False
            else:
                return True
        return False

    def prepare_data(self, varholder, var_pairs):
        self.config['timerange'] = \
            str(int(self.dt_to_timestamp(varholder.from_dt))) + "-" + \
            str(int(self.dt_to_timestamp(varholder.to_dt)))
        self.backtesting = Backtesting(self.config)
        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        varholder.data, varholder.timerange = self.backtesting.load_bt_data()
        varholder.indicators = self.backtesting.strategy.advise_all_indicators(varholder.data)
        varholder.result = self.get_result(self.backtesting, varholder.indicators)

    def start(self, config, strategy_obj: dict) -> None:
        self.strategy_obj = strategy_obj
        self.config = config
        self.current_analysis = backtest_lookahead_bias_checker.analysis()

        max_try_signals: int = 3
        found_signals: int = 0
        continue_with_strategy = True

        # first we need to get the necessary entry/exit signals
        # so we start by 14 days and increase in 1 month steps
        # until we have the desired trade amount.
        for try_buysignals in range(max_try_signals):  # range(3) = 0..2
            # re-initialize backtesting-variable
            self.full_varHolder = backtest_lookahead_bias_checker.varHolder()

            # define datetimes in human readable format
            self.full_varHolder.from_dt = datetime(2022, 9, 1)
            self.full_varHolder.to_dt = datetime(2022, 9, 15) + timedelta(days=30 * try_buysignals)

            self.prepare_data(self.full_varHolder, self.config['pairs'])

            found_signals = self.full_varHolder.result['results'].shape[0] + 1
            if try_buysignals == max_try_signals - 1:
                if found_signals < self.signals_to_check / 2:
                    print(f"... only found {str(int(found_signals / 2))} "
                          f"buy signals for {self.strategy_obj['name']}. "
                          f"Cancelling...")
                    continue_with_strategy = False
                else:
                    print(
                        f"Found {str(found_signals)} buy signals. "
                        f"Going with max {str(self.signals_to_check)} "
                        f" buy signals in the full timerange from "
                        f"{str(self.full_varHolder.from_dt)} to {str(self.full_varHolder.to_dt)}")
                    break
            elif found_signals < self.signals_to_check:
                print(
                    f"Only found {str(found_signals)} buy signals in the full timerange from "
                    f"{str(self.full_varHolder.from_dt)} to "
                    f"{str(self.full_varHolder.to_dt)}. "
                    f"will increase timerange trying to get at least "
                    f"{str(self.signals_to_check)} signals.")
            else:
                print(
                    f"Found {str(found_signals)} buy signals, more than necessary. "
                    f"Reducing to {str(self.signals_to_check)} "
                    f"checked buy signals in the full timerange from "
                    f"{str(self.full_varHolder.from_dt)} to {str(self.full_varHolder.to_dt)}")
                break
        if not continue_with_strategy:
            return

        for idx, result_row in self.full_varHolder.result['results'].iterrows():
            if self.current_analysis.total_signals == self.signals_to_check:
                break

            # if force-sold, ignore this signal since here it will unconditionally exit.
            if result_row.close_date == self.dt_to_timestamp(self.full_varHolder.to_dt):
                continue

            self.current_analysis.total_signals += 1

            self.entry_varholder = backtest_lookahead_bias_checker.varHolder()
            self.exit_varholder = backtest_lookahead_bias_checker.varHolder()

            self.entry_varholder.from_dt = self.full_varHolder.from_dt  # result_row['open_date']
            self.entry_varholder.compared_dt = result_row['open_date']

            # to_dt needs +1 candle since it won't buy on the last candle
            self.entry_varholder.to_dt = result_row['open_date'] + \
                timedelta(minutes=timeframe_to_minutes(self.config['timeframe']) * 2)

            self.prepare_data(self.entry_varholder, [result_row['pair']])

            # ---
            # print("analyzing the sell signal")
            # to_dt needs +1 candle since it will always sell all trades on the last candle
            self.exit_varholder.from_dt = self.full_varHolder.from_dt  # result_row['open_date']
            self.exit_varholder.to_dt = \
                result_row['close_date'] + \
                timedelta(minutes=timeframe_to_minutes(self.config['timeframe']))
            self.exit_varholder.compared_dt = result_row['close_date']

            self.prepare_data(self.exit_varholder, [result_row['pair']])

            # register if buy signal is broken
            if not self.report_signal(
                    self.entry_varholder.result,
                    "open_date", self.entry_varholder.compared_dt):
                self.current_analysis.false_entry_signals += 1

            # register if buy or sell signal is broken
            if not self.report_signal(self.entry_varholder.result,
                                      "open_date", self.entry_varholder.compared_dt) \
                    or not self.report_signal(self.exit_varholder.result,
                                              "close_date", self.exit_varholder.compared_dt):
                self.current_analysis.false_exit_signals += 1

            self.analyze_indicators(self.full_varHolder, self.entry_varholder, result_row['pair'])
            self.analyze_indicators(self.full_varHolder, self.exit_varholder, result_row['pair'])

        if self.current_analysis.false_entry_signals > 0 or \
                self.current_analysis.false_exit_signals > 0 or \
                len(self.current_analysis.false_indicators) > 0:
            print(" => " + self.strategy_obj['name'] + ": bias detected!")
            self.current_analysis.has_bias = True
        else:
            print(self.strategy_obj['name'] + ": no bias detected")
