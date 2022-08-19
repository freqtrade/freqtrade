import logging
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (DecimalParameter, IntParameter, IStrategy,
                                merge_informative_pair)
from numpy.lib import math
from pandas import DataFrame
from technical import qtpylib

logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    """
    Example classifier hybrid strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    
    The underlying original supertrend strat is authored by @juankysoriano (Juan Carlos Soriano)
    * github: https://github.com/juankysoriano/
    
    This strategy is not designed to be used live
    """

    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.1
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True

    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)
    
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }
    
    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=3)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=10)
    buy_p3 = IntParameter(7, 21, default=10)

    sell_m1 = IntParameter(1, 7, default=1)
    sell_m2 = IntParameter(1, 7, default=3)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=10)
    sell_p3 = IntParameter(7, 21, default=10)

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e. user should not prepend any supporting metrics
        (e.g. bb_lowerband below) with % unless they explicitly want to pass that metric to the
        model.
        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        """

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, window=t)
            informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{coin}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{coin}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{coin}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{coin}bb_width-period_{t}"] = (
                informative[f"{coin}bb_upperband-period_{t}"]
                - informative[f"{coin}bb_lowerband-period_{t}"]
            ) / informative[f"{coin}bb_middleband-period_{t}"]
            informative[f"%-{coin}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{coin}bb_lowerband-period_{t}"]
            )

            informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{coin}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        informative[f"%-{coin}raw_volume"] = informative["volume"]
        informative[f"%-{coin}raw_price"] = informative["close"]

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

            # Classifiers are typically set up with strings as targets:
            df['&s-up_or_down'] = np.where( df["close"].shift(-50) >
                                            df["close"], 'up', 'down')

            # REGRESSOR Model: Can use single or multi traget
            # user adds targets here by prepending them with &- (see convention below)
            #df["&-s_close"] = (
            #    df["close"]
            #    .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #    .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #    .mean()
            #    / df["close"]
            #    - 1
            #)
            # If user wishes to use multiple targets, they can add more by
            # appending more columns with '&'. User should keep in mind that multi targets
            # requires a multioutput prediction model such as
            # templates/CatboostPredictionMultiModel.py,

            # df["&-s_range"] = (
            #     df["close"]
            #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .max()
            #     -
            #     df["close"]
            #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .min()
            # )

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # All indicators must be populated by populate_any_indicators() for live functionality
        # to work correctly.

        # the model will return all labels created by user in `populate_any_indicators`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `populate_any_indicators()` for each training period.
        
        for multiplier in self.buy_m1.range:
            for period in self.buy_p1.range:
                dataframe[f"supertrend_1_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.buy_m2.range:
            for period in self.buy_p2.range:
                dataframe[f"supertrend_2_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.buy_m3.range:
            for period in self.buy_p3.range:
                dataframe[f"supertrend_3_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.sell_m1.range:
            for period in self.sell_p1.range:
                dataframe[f"supertrend_1_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.sell_m2.range:
            for period in self.sell_p2.range:
                dataframe[f"supertrend_2_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.sell_m3.range:
            for period in self.sell_p3.range:
                dataframe[f"supertrend_3_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (df[f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"] == "up") &
            (df[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up") &
            (df[f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"] == "up") &
            (df["do_predict"] == 1) &
            (df['&s-up_or_down'] == 'up'), 
            "enter_long",
        ] = 1

        df.loc[
            (df[f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"] == "down") & 
            (df[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down") & 
            (df[f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"] == "down") & 
            (df["do_predict"] == 1) & 
            (df['&s-up_or_down'] == 'down'),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df.loc[
            (df[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down"),
            "exit_long",
        ] = 1

        df.loc[
            (df[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up"),
            "exit_short",
        ] = 1

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
        rate: float, time_in_force: str, current_time, entry_tag, side: str,
        **kwargs, ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:

        return 1

    """
    Supertrend Indicator; adapted for freqtrade, optimized by the math genius.
    from: Perkmeister#2394
    """

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()
        last_row = dataframe.tail(1).index.item()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        BASIC_UB = ((df['high'] + df['low']) / 2 + multiplier * df['ATR']).values
        BASIC_LB = ((df['high'] + df['low']) / 2 - multiplier * df['ATR']).values
        FINAL_UB = np.zeros(last_row + 1)
        FINAL_LB = np.zeros(last_row + 1)
        ST = np.zeros(last_row + 1)
        CLOSE = df['close'].values

        # Compute final upper and lower bands
        for i in range(period, last_row + 1):
            FINAL_UB[i] = BASIC_UB[i] if BASIC_UB[i] < FINAL_UB[i - 1] or CLOSE[i - 1] > FINAL_UB[i - 1] else FINAL_UB[i - 1]
            FINAL_LB[i] = BASIC_LB[i] if BASIC_LB[i] > FINAL_LB[i - 1] or CLOSE[i - 1] < FINAL_LB[i - 1] else FINAL_LB[i - 1]

        # Set the Supertrend value
        for i in range(period, last_row + 1):
            ST[i] = FINAL_UB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] <= FINAL_UB[i] else \
                    FINAL_LB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] >  FINAL_UB[i] else \
                    FINAL_LB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] >= FINAL_LB[i] else \
                    FINAL_UB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] <  FINAL_LB[i] else 0.00
        df_ST = pd.DataFrame(ST, columns=[st])
        df = pd.concat([df, df_ST],axis=1)

        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })
