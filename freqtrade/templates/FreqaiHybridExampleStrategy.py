import logging

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.

    Launching this strategy would be:

    freqtrade trade --strategy FreqaiExampleHyridStrategy --strategy-path freqtrade/templates
    --freqaimodel CatboostClassifier --config config_examples/config_freqai.example.json

    or the user simply adds this to their config:

    "freqai": {
        "enabled": true,
        "purge_old_models": true,
        "train_period_days": 15,
        "identifier": "uniqe-id",
        "feature_parameters": {
            "include_timeframes": [
                "3m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_max_period_candles": 20,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0.33,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    Thanks to @smarm and @jooopieeert for developing and sharing the strategy.
    """

    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025, "240": -1}

    process_only_new_candles = True
    stoploss = -0.1
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True

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

    # FreqAI required function, leave as is or add additional informatives to existing structure.
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

    # FreqAI required function, user can add or remove indicators, but general structure
    # must stay the same.
    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        User feeds these indicators to FreqAI to train a classifier to decide
        if the market will go up or down.

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
            informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)
            informative[f"%-{coin}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        # FreqAI needs the following lines in order to detect features and automatically
        # expand upon them.
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

        # User can set the "target" here (in present case it is the
        # "up" or "down")
        if set_generalized_indicators:
            # User "looks into the future" here to figure out if the future
            # will be "up" or "down". This same column name is available to
            # the user
            df['&s-up_or_down'] = np.where(df["close"].shift(-50) >
                                           df["close"], 'up', 'down')

        return df

    # flake8: noqa: C901
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

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

        # User now can use their custom strat creation in addition to their
        # future prediction "up" or "down".

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
            FINAL_UB[i] = (BASIC_UB[i] if BASIC_UB[i] < FINAL_UB[i - 1]
                           or CLOSE[i - 1] > FINAL_UB[i - 1] else FINAL_UB[i - 1])
            FINAL_LB[i] = (BASIC_LB[i] if BASIC_LB[i] > FINAL_LB[i - 1]
                           or CLOSE[i - 1] < FINAL_LB[i - 1] else FINAL_LB[i - 1])

        # Set the Supertrend value
        for i in range(period, last_row + 1):
            ST[i] = FINAL_UB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] <= FINAL_UB[i] else \
                    FINAL_LB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] > FINAL_UB[i] else \
                    FINAL_LB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] >= FINAL_LB[i] else \
                    FINAL_UB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] < FINAL_LB[i] else 0.00
        df_ST = pd.DataFrame(ST, columns=[st])
        df = pd.concat([df, df_ST], axis=1)

        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST': df[st],
            'STX': df[stx]
        })
