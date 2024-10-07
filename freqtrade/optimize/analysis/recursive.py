import logging
import numbers
import shutil
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.loggers.set_log_levels import (
    reduce_verbosity_for_bias_tester,
    restore_verbosity_for_bias_tester,
)
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.base_analysis import BaseAnalysis, VarHolder
from freqtrade.resolvers import StrategyResolver


logger = logging.getLogger(__name__)


def is_number(variable):
    return isinstance(variable, numbers.Number) and not isinstance(variable, bool)


class RecursiveAnalysis(BaseAnalysis):
    def __init__(self, config: dict[str, Any], strategy_obj: dict):
        self._startup_candle = list(
            map(int, config.get("startup_candle", [199, 399, 499, 999, 1999]))
        )

        super().__init__(config, strategy_obj)

        strat = StrategyResolver.load_strategy(config)
        self._strat_scc = strat.startup_candle_count

        if self._strat_scc not in self._startup_candle:
            self._startup_candle.append(self._strat_scc)
        self._startup_candle.sort()

        self.partial_varHolder_array: list[VarHolder] = []
        self.partial_varHolder_lookahead_array: list[VarHolder] = []

        self.dict_recursive: dict[str, Any] = dict()

    # For recursive bias check
    # analyzes two data frames with processed indicators and shows differences between them.
    def analyze_indicators(self):
        pair_to_check = self.local_config["pairs"][0]
        logger.info("Start checking for recursive bias")

        # check and report signals
        base_last_row = self.full_varHolder.indicators[pair_to_check].iloc[-1]

        for part in self.partial_varHolder_array:
            part_last_row = part.indicators[pair_to_check].iloc[-1]

            compare_df = base_last_row.compare(part_last_row)
            if compare_df.shape[0] > 0:
                # print(compare_df)
                for col_name, values in compare_df.items():
                    # print(col_name)
                    if "other" == col_name:
                        continue
                    indicators = values.index

                    for indicator in indicators:
                        if indicator not in self.dict_recursive:
                            self.dict_recursive[indicator] = {}

                        values_diff = compare_df.loc[indicator]
                        values_diff_self = values_diff.loc["self"]
                        values_diff_other = values_diff.loc["other"]

                        if (
                            values_diff_self
                            and values_diff_other
                            and is_number(values_diff_self)
                            and is_number(values_diff_other)
                        ):
                            diff = (values_diff_other - values_diff_self) / values_diff_self * 100
                            str_diff = f"{diff:.3f}%"
                        else:
                            str_diff = "NaN"
                        self.dict_recursive[indicator][part.startup_candle] = str_diff

            else:
                logger.info("No variance on indicator(s) found due to recursive formula.")
                break

    # For lookahead bias check
    # analyzes two data frames with processed indicators and shows differences between them.
    def analyze_indicators_lookahead(self):
        pair_to_check = self.local_config["pairs"][0]
        logger.info("Start checking for lookahead bias on indicators only")

        part = self.partial_varHolder_lookahead_array[0]
        part_last_row = part.indicators[pair_to_check].iloc[-1]
        date_to_check = part_last_row["date"]
        index_to_get = self.full_varHolder.indicators[pair_to_check]["date"] == date_to_check
        base_row_check = self.full_varHolder.indicators[pair_to_check].loc[index_to_get].iloc[-1]

        check_time = part.to_dt.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(f"Check indicators at {check_time}")
        # logger.info(f"vs {part_timerange} with {part.startup_candle} startup candle")

        compare_df = base_row_check.compare(part_last_row)
        if compare_df.shape[0] > 0:
            # print(compare_df)
            for col_name, values in compare_df.items():
                # print(col_name)
                if "other" == col_name:
                    continue
                indicators = values.index

                for indicator in indicators:
                    logger.info(f"=> found lookahead in indicator {indicator}")
                    # logger.info("base value {:.5f}".format(values_diff_self))
                    # logger.info("part value {:.5f}".format(values_diff_other))

        else:
            logger.info("No lookahead bias on indicators found.")

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

        backtesting = Backtesting(prepare_data_config, self.exchange)
        self.exchange = backtesting.exchange
        backtesting._set_strategy(backtesting.strategylist[0])

        varholder.data, varholder.timerange = backtesting.load_bt_data()
        backtesting.load_bt_data_detail()
        varholder.timeframe = backtesting.timeframe

        varholder.indicators = backtesting.strategy.advise_all_indicators(varholder.data)

    def fill_partial_varholder(self, start_date, startup_candle):
        logger.info(f"Calculating indicators using startup candle of {startup_candle}.")
        partial_varHolder = VarHolder()

        partial_varHolder.from_dt = start_date
        partial_varHolder.to_dt = self.full_varHolder.to_dt
        partial_varHolder.startup_candle = startup_candle

        self.local_config["startup_candle_count"] = startup_candle

        self.prepare_data(partial_varHolder, self.local_config["pairs"])

        self.partial_varHolder_array.append(partial_varHolder)

    def fill_partial_varholder_lookahead(self, end_date):
        logger.info("Calculating indicators to test lookahead on indicators.")

        partial_varHolder = VarHolder()

        partial_varHolder.from_dt = self.full_varHolder.from_dt
        partial_varHolder.to_dt = end_date

        self.prepare_data(partial_varHolder, self.local_config["pairs"])

        self.partial_varHolder_lookahead_array.append(partial_varHolder)

    def start(self) -> None:
        super().start()

        reduce_verbosity_for_bias_tester()
        start_date_full = self.full_varHolder.from_dt
        end_date_full = self.full_varHolder.to_dt

        timeframe_minutes = timeframe_to_minutes(self.full_varHolder.timeframe)

        end_date_partial = start_date_full + timedelta(minutes=int(timeframe_minutes * 10))

        self.fill_partial_varholder_lookahead(end_date_partial)

        # restore_verbosity_for_bias_tester()

        start_date_partial = end_date_full - timedelta(minutes=int(timeframe_minutes))

        for startup_candle in self._startup_candle:
            self.fill_partial_varholder(start_date_partial, startup_candle)

        # Restore verbosity, so it's not too quiet for the next strategy
        restore_verbosity_for_bias_tester()

        self.analyze_indicators()
        self.analyze_indicators_lookahead()
