import logging

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

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
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    Thanks to @smarmau and @johanvulgt for developing and sharing the strategy.
    """

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    plot_config = {
        'main_plot': {
            'tema': {},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            },
            "Up_or_down": {
                '&s-up_or_down': {'color': 'green'},
            }
        }
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

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

        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA indicators to combine with the Freqai targets
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df['rsi'], self.buy_rsi.value)) &
                (df['tema'] <= df['bb_middleband']) &  # Guard: tema below BB middle
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            'enter_long'] = 1

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df['rsi'], self.short_rsi.value)) &
                (df['tema'] > df['bb_middleband']) &  # Guard: tema above BB middle
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'down')
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df['rsi'], self.sell_rsi.value)) &
                (df['tema'] > df['bb_middleband']) &  # Guard: tema above BB middle
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df['rsi'], self.exit_short_rsi.value)) &
                # Guard: tema below BB middle
                (df['tema'] <= df['bb_middleband']) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return df
