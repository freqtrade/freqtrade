import copy
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.history import get_timerange
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.optimize.backtesting import Backtesting


class VarHolder:
    timerange: TimeRange
    data: DataFrame
    indicators: DataFrame
    result: DataFrame
    compared: DataFrame
    from_dt: datetime
    to_dt: datetime
    compared_dt: datetime


class Analysis:
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


class BacktestLookaheadBiasChecker:

    def __init__(self):
        self.exportfilename = None
        self.strategy_obj = None
        self.current_analysis = None
        self.local_config = None
        self.full_varHolder = None
        self.entry_varHolder = None
        self.exit_varHolder = None
        self.backtesting = None
        self.minimum_trade_amount = None
        self.targeted_trade_amount = None
        self.failed_bias_check = True

    @staticmethod
    def dt_to_timestamp(dt):
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    @staticmethod
    def get_result(backtesting, processed):
        min_date, max_date = get_timerange(processed)

        result = backtesting.backtest(
            processed=deepcopy(processed),
            start_date=min_date,
            end_date=max_date
        )
        return result

    @staticmethod
    def report_signal(result, column_name, checked_timestamp):
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
                    # compare_df now comprises tuples with [1] having either 'self' or 'other'
                    if 'other' in col_name[1]:
                        continue
                    self_value = compare_df_row[col_idx]
                    other_value = compare_df_row[col_idx + 1]

                    # output differences
                    if self_value != other_value:

                        if not self.current_analysis.false_indicators.__contains__(col_name[0]):
                            self.current_analysis.false_indicators.append(col_name[0])
                            print(f"=> found look ahead bias in indicator {col_name[0]}. " +
                                  f"{str(self_value)} != {str(other_value)}")

    def prepare_data(self, varHolder, pairs_to_load):
        prepare_data_config = copy.deepcopy(self.local_config)
        prepare_data_config['timerange'] = (str(self.dt_to_timestamp(varHolder.from_dt)) + "-" +
                                            str(self.dt_to_timestamp(varHolder.to_dt)))
        prepare_data_config['pairs'] = pairs_to_load

        self.backtesting = Backtesting(prepare_data_config)
        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        varHolder.data, varHolder.timerange = self.backtesting.load_bt_data()
        self.backtesting.load_bt_data_detail()

        varHolder.indicators = self.backtesting.strategy.advise_all_indicators(varHolder.data)
        varHolder.result = self.get_result(self.backtesting, varHolder.indicators)

    def update_output_file(self):
        pass

    def start(self, config, strategy_obj: dict, args) -> None:

        # deepcopy so we can change the pairs for the 2ndary runs
        # and not worry about another strategy to check after.
        self.local_config = deepcopy(config)
        self.local_config['strategy_list'] = [strategy_obj['name']]
        self.current_analysis = Analysis()
        self.minimum_trade_amount = args['minimum_trade_amount']
        self.targeted_trade_amount = args['targeted_trade_amount']
        self.exportfilename = args['exportfilename']
        self.strategy_obj = strategy_obj

        # first make a single backtest
        self.full_varHolder = VarHolder()

        # define datetime in human-readable format
        parsed_timerange = TimeRange.parse_timerange(config['timerange'])

        if parsed_timerange.startdt is None:
            self.full_varHolder.from_dt = datetime.utcfromtimestamp(0)
        else:
            self.full_varHolder.from_dt = parsed_timerange.startdt

        if parsed_timerange.stopdt is None:
            self.full_varHolder.to_dt = datetime.now()
        else:
            self.full_varHolder.to_dt = parsed_timerange.stopdt

        self.prepare_data(self.full_varHolder, self.local_config['pairs'])

        found_signals: int = self.full_varHolder.result['results'].shape[0] + 1
        if found_signals >= self.targeted_trade_amount:
            print(f"Found {found_signals} trades, calculating {self.targeted_trade_amount} trades.")
        elif self.targeted_trade_amount >= found_signals >= self.minimum_trade_amount:
            print(f"Only found {found_signals} trades. Calculating all available trades.")
        else:
            print(f"found {found_signals} trades "
                  f"which is less than minimum_trade_amount {self.minimum_trade_amount}. "
                  f"Cancelling this backtest lookahead bias test.")
            return

        # now we loop through all entry signals
        # starting from the same datetime to avoid miss-reports of bias
        for idx, result_row in self.full_varHolder.result['results'].iterrows():
            if self.current_analysis.total_signals == self.targeted_trade_amount:
                break

            # if force-sold, ignore this signal since here it will unconditionally exit.
            if result_row.close_date == self.dt_to_timestamp(self.full_varHolder.to_dt):
                continue

            self.current_analysis.total_signals += 1

            self.entry_varHolder = VarHolder()
            self.exit_varHolder = VarHolder()

            self.entry_varHolder.from_dt = self.full_varHolder.from_dt
            self.entry_varHolder.compared_dt = result_row['open_date']
            # to_dt needs +1 candle since it won't buy on the last candle
            self.entry_varHolder.to_dt = (result_row['open_date'] +
                                          timedelta(minutes=timeframe_to_minutes(
                                              self.local_config['timeframe'])))

            self.prepare_data(self.entry_varHolder, [result_row['pair']])

            # to_dt needs +1 candle since it will always exit/force-exit trades on the last candle
            self.exit_varHolder.from_dt = self.full_varHolder.from_dt
            self.exit_varHolder.to_dt = (result_row['close_date'] +
                                         timedelta(minutes=timeframe_to_minutes(
                                             self.local_config['timeframe'])))
            self.exit_varHolder.compared_dt = result_row['close_date']

            self.prepare_data(self.exit_varHolder, [result_row['pair']])

            # register if buy signal is broken
            if not self.report_signal(
                    self.entry_varHolder.result, "open_date", self.entry_varHolder.compared_dt):
                self.current_analysis.false_entry_signals += 1

            # register if buy or sell signal is broken
            if not self.report_signal(
                    self.exit_varHolder.result, "close_date", self.exit_varHolder.compared_dt):
                self.current_analysis.false_exit_signals += 1

            # check if the indicators themselves contain biased data
            self.analyze_indicators(self.full_varHolder, self.entry_varHolder, result_row['pair'])
            self.analyze_indicators(self.full_varHolder, self.exit_varHolder, result_row['pair'])

        if (self.current_analysis.false_entry_signals > 0 or
                self.current_analysis.false_exit_signals > 0 or
                len(self.current_analysis.false_indicators) > 0):
            print(" => " + self.local_config['strategy_list'][0] + ": bias detected!")
            self.current_analysis.has_bias = True
        else:
            print(self.local_config['strategy_list'][0] + ": no bias detected")

        self.failed_bias_check = False
