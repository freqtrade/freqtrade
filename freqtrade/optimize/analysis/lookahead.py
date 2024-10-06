import logging
import shutil
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pandas import DataFrame

from freqtrade.data.history import get_timerange
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.loggers.set_log_levels import (
    reduce_verbosity_for_bias_tester,
    restore_verbosity_for_bias_tester,
)
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.base_analysis import BaseAnalysis, VarHolder


logger = logging.getLogger(__name__)


class Analysis:
    def __init__(self) -> None:
        self.total_signals = 0
        self.false_entry_signals = 0
        self.false_exit_signals = 0
        self.false_indicators: list[str] = []
        self.has_bias = False


class LookaheadAnalysis(BaseAnalysis):
    def __init__(self, config: dict[str, Any], strategy_obj: dict):
        super().__init__(config, strategy_obj)

        self.entry_varHolders: list[VarHolder] = []
        self.exit_varHolders: list[VarHolder] = []

        self.current_analysis = Analysis()
        self.minimum_trade_amount = config["minimum_trade_amount"]
        self.targeted_trade_amount = config["targeted_trade_amount"]

    @staticmethod
    def get_result(backtesting: Backtesting, processed: DataFrame):
        min_date, max_date = get_timerange(processed)

        result = backtesting.backtest(
            processed=deepcopy(processed), start_date=min_date, end_date=max_date
        )
        return result

    @staticmethod
    def report_signal(result: dict, column_name: str, checked_timestamp: datetime):
        df = result["results"]
        row_count = df[column_name].shape[0]

        if row_count == 0:
            return False
        else:
            df_cut = df[(df[column_name] == checked_timestamp)]
            if df_cut[column_name].shape[0] == 0:
                return False
            else:
                return True
        return False

    # analyzes two data frames with processed indicators and shows differences between them.
    def analyze_indicators(self, full_vars: VarHolder, cut_vars: VarHolder, current_pair: str):
        # extract dataframes
        cut_df: DataFrame = cut_vars.indicators[current_pair]
        full_df: DataFrame = full_vars.indicators[current_pair]

        # cut longer dataframe to length of the shorter
        full_df_cut = full_df[(full_df.date == cut_vars.compared_dt)].reset_index(drop=True)
        cut_df_cut = cut_df[(cut_df.date == cut_vars.compared_dt)].reset_index(drop=True)

        # check if dataframes are not empty
        if full_df_cut.shape[0] != 0 and cut_df_cut.shape[0] != 0:
            # compare dataframes
            compare_df = full_df_cut.compare(cut_df_cut)

            if compare_df.shape[0] > 0:
                for col_name, values in compare_df.items():
                    col_idx = compare_df.columns.get_loc(col_name)
                    compare_df_row = compare_df.iloc[0]
                    # compare_df now comprises tuples with [1] having either 'self' or 'other'
                    if "other" in col_name[1]:
                        continue
                    self_value = compare_df_row.iloc[col_idx]
                    other_value = compare_df_row.iloc[col_idx + 1]

                    # output differences
                    if self_value != other_value:
                        if not self.current_analysis.false_indicators.__contains__(col_name[0]):
                            self.current_analysis.false_indicators.append(col_name[0])
                            logger.info(
                                f"=> found look ahead bias in indicator "
                                f"{col_name[0]}. "
                                f"{str(self_value)} != {str(other_value)}"
                            )

    def prepare_data(self, varholder: VarHolder, pairs_to_load: list[DataFrame]):
        if "freqai" in self.local_config and "identifier" in self.local_config["freqai"]:
            # purge previous data if the freqai model is defined
            # (to be sure nothing is carried over from older backtests)
            path_to_current_identifier = Path(
                f"{self.local_config['user_data_dir']}/models/"
                f"{self.local_config['freqai']['identifier']}"
            ).resolve()
            # remove folder and its contents
            if Path.exists(path_to_current_identifier):
                shutil.rmtree(path_to_current_identifier)

        prepare_data_config = deepcopy(self.local_config)
        prepare_data_config["timerange"] = (
            str(self.dt_to_timestamp(varholder.from_dt))
            + "-"
            + str(self.dt_to_timestamp(varholder.to_dt))
        )
        prepare_data_config["exchange"]["pair_whitelist"] = pairs_to_load

        if self._fee is not None:
            # Don't re-calculate fee per pair, as fee might differ per pair.
            prepare_data_config["fee"] = self._fee

        backtesting = Backtesting(prepare_data_config, self.exchange)
        self.exchange = backtesting.exchange
        self._fee = backtesting.fee
        backtesting._set_strategy(backtesting.strategylist[0])

        varholder.data, varholder.timerange = backtesting.load_bt_data()
        backtesting.load_bt_data_detail()
        varholder.timeframe = backtesting.timeframe

        varholder.indicators = backtesting.strategy.advise_all_indicators(varholder.data)
        varholder.result = self.get_result(backtesting, varholder.indicators)

    def fill_entry_and_exit_varHolders(self, result_row):
        # entry_varHolder
        entry_varHolder = VarHolder()
        self.entry_varHolders.append(entry_varHolder)
        entry_varHolder.from_dt = self.full_varHolder.from_dt
        entry_varHolder.compared_dt = result_row["open_date"]
        # to_dt needs +1 candle since it won't buy on the last candle
        entry_varHolder.to_dt = result_row["open_date"] + timedelta(
            minutes=timeframe_to_minutes(self.full_varHolder.timeframe)
        )
        self.prepare_data(entry_varHolder, [result_row["pair"]])

        # exit_varHolder
        exit_varHolder = VarHolder()
        self.exit_varHolders.append(exit_varHolder)
        # to_dt needs +1 candle since it will always exit/force-exit trades on the last candle
        exit_varHolder.from_dt = self.full_varHolder.from_dt
        exit_varHolder.to_dt = result_row["close_date"] + timedelta(
            minutes=timeframe_to_minutes(self.full_varHolder.timeframe)
        )
        exit_varHolder.compared_dt = result_row["close_date"]
        self.prepare_data(exit_varHolder, [result_row["pair"]])

    # now we analyze a full trade of full_varholder and look for analyze its bias
    def analyze_row(self, idx: int, result_row):
        # if force-sold, ignore this signal since here it will unconditionally exit.
        if result_row.close_date == self.dt_to_timestamp(self.full_varHolder.to_dt):
            return

        # keep track of how many signals are processed at total
        self.current_analysis.total_signals += 1

        # fill entry_varHolder and exit_varHolder
        self.fill_entry_and_exit_varHolders(result_row)

        # this will trigger a logger-message
        buy_or_sell_biased: bool = False

        # register if buy signal is broken
        if not self.report_signal(
            self.entry_varHolders[idx].result, "open_date", self.entry_varHolders[idx].compared_dt
        ):
            self.current_analysis.false_entry_signals += 1
            buy_or_sell_biased = True

        # register if buy or sell signal is broken
        if not self.report_signal(
            self.exit_varHolders[idx].result, "close_date", self.exit_varHolders[idx].compared_dt
        ):
            self.current_analysis.false_exit_signals += 1
            buy_or_sell_biased = True

        if buy_or_sell_biased:
            logger.info(
                f"found lookahead-bias in trade "
                f"pair: {result_row['pair']}, "
                f"timerange:{result_row['open_date']} - {result_row['close_date']}, "
                f"idx: {idx}"
            )

        # check if the indicators themselves contain biased data
        self.analyze_indicators(self.full_varHolder, self.entry_varHolders[idx], result_row["pair"])
        self.analyze_indicators(self.full_varHolder, self.exit_varHolders[idx], result_row["pair"])

    def start(self) -> None:
        super().start()

        reduce_verbosity_for_bias_tester()

        # check if requirements have been met of full_varholder
        found_signals: int = self.full_varHolder.result["results"].shape[0] + 1
        if found_signals >= self.targeted_trade_amount:
            logger.info(
                f"Found {found_signals} trades, "
                f"calculating {self.targeted_trade_amount} trades."
            )
        elif self.targeted_trade_amount >= found_signals >= self.minimum_trade_amount:
            logger.info(f"Only found {found_signals} trades. Calculating all available trades.")
        else:
            logger.info(
                f"found {found_signals} trades "
                f"which is less than minimum_trade_amount {self.minimum_trade_amount}. "
                f"Cancelling this backtest lookahead bias test."
            )
            return

        # now we loop through all signals
        # starting from the same datetime to avoid miss-reports of bias
        for idx, result_row in self.full_varHolder.result["results"].iterrows():
            if self.current_analysis.total_signals == self.targeted_trade_amount:
                logger.info(f"Found targeted trade amount = {self.targeted_trade_amount} signals.")
                break
            if found_signals < self.minimum_trade_amount:
                logger.info(
                    f"only found {found_signals} "
                    f"which is smaller than "
                    f"minimum trade amount = {self.minimum_trade_amount}. "
                    f"Exiting this lookahead-analysis"
                )
                return None
            if "force_exit" in result_row["exit_reason"]:
                logger.info(
                    "found force-exit in pair: {result_row['pair']}, "
                    f"timerange:{result_row['open_date']}-{result_row['close_date']}, "
                    f"idx: {idx}, skipping this one to avoid a false-positive."
                )

                # just to keep the IDs of both full, entry and exit varholders the same
                # to achieve a better debugging experience
                self.entry_varHolders.append(VarHolder())
                self.exit_varHolders.append(VarHolder())
                continue

            self.analyze_row(idx, result_row)

        if len(self.entry_varHolders) < self.minimum_trade_amount:
            logger.info(
                f"only found {found_signals} after skipping forced exits "
                f"which is smaller than "
                f"minimum trade amount = {self.minimum_trade_amount}. "
                f"Exiting this lookahead-analysis"
            )

        # Restore verbosity, so it's not too quiet for the next strategy
        restore_verbosity_for_bias_tester()
        # check and report signals
        if self.current_analysis.total_signals < self.local_config["minimum_trade_amount"]:
            logger.info(
                f" -> {self.local_config['strategy']} : too few trades. "
                f"We only found {self.current_analysis.total_signals} trades. "
                f"Hint: Extend the timerange "
                f"to get at least {self.local_config['minimum_trade_amount']} "
                f"or lower the value of minimum_trade_amount."
            )
            self.failed_bias_check = True
        elif (
            self.current_analysis.false_entry_signals > 0
            or self.current_analysis.false_exit_signals > 0
            or len(self.current_analysis.false_indicators) > 0
        ):
            logger.info(f" => {self.local_config['strategy']} : bias detected!")
            self.current_analysis.has_bias = True
            self.failed_bias_check = False
        else:
            logger.info(self.local_config["strategy"] + ": no bias detected")
            self.failed_bias_check = False
