import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.freqai.strategy_bridge import CustomModel
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class FreqaiExampleStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.model = CustomModel(self.config)
    self.model.bridge.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    minimal_roi = {"0": 0.01, "240": -1}

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

    stoploss = -0.05
    use_sell_signal = True
    startup_candle_count: int = 300

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def bot_start(self):
        self.model = CustomModel(self.config)

    def populate_any_indicators(self, pair, df, tf, informative=None, coin=""):
        """
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e. user should not prepend any supporting metrics
        (e.g. bb_lowerband below) with % unless they explicitly want to pass that metric to the
        model.
        :params:
        :pair: pair to be used as informative
        :df: strategy dataframe which will receive merges from informatives
        :tf: timeframe of the dataframe which will modify the feature names
        :informative: the dataframe associated with the informative pair
        :coin: the name of the coin which will modify the feature names.
        """
        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        informative['%-' + coin + "rsi"] = ta.RSI(informative, timeperiod=14)
        informative['%-' + coin + "mfi"] = ta.MFI(informative, timeperiod=25)
        informative['%-' + coin + "adx"] = ta.ADX(informative, window=20)

        informative[coin + "20sma"] = ta.SMA(informative, timeperiod=20)
        informative[coin + "21ema"] = ta.EMA(informative, timeperiod=21)
        informative['%-' + coin + "bmsb"] = np.where(
            informative[coin + "20sma"].lt(informative[coin + "21ema"]), 1, 0
        )
        informative['%-' + coin + "close_over_20sma"] = informative["close"] / informative[
                                                                                    coin + "20sma"]

        informative['%-' + coin + "mfi"] = ta.MFI(informative, timeperiod=25)

        informative[coin + "ema21"] = ta.EMA(informative, timeperiod=21)
        informative[coin + "sma20"] = ta.SMA(informative, timeperiod=20)
        stoch = ta.STOCHRSI(informative, 15, 20, 2, 2)
        informative['%-' + coin + "srsi-fk"] = stoch["fastk"]
        informative['%-' + coin + "srsi-fd"] = stoch["fastd"]

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=14, stds=2.2)
        informative[coin + "bb_lowerband"] = bollinger["lower"]
        informative[coin + "bb_middleband"] = bollinger["mid"]
        informative[coin + "bb_upperband"] = bollinger["upper"]
        informative['%-' + coin + "bb_width"] = (
            informative[coin + "bb_upperband"] - informative[coin + "bb_lowerband"]
        ) / informative[coin + "bb_middleband"]
        informative['%-' + coin + "close-bb_lower"] = (
            informative["close"] / informative[coin + "bb_lowerband"]
        )

        informative['%-' + coin + "roc"] = ta.ROC(informative, timeperiod=3)
        informative['%-' + coin + "adx"] = ta.ADX(informative, window=14)

        macd = ta.MACD(informative)
        informative['%-' + coin + "macd"] = macd["macd"]
        informative[coin + "pct-change"] = informative["close"].pct_change()
        informative['%-' + coin + "relative_volume"] = (
            informative["volume"] / informative["volume"].rolling(10).mean()
        )

        informative[coin + "pct-change"] = informative["close"].pct_change()

        indicators = [col for col in informative if col.startswith('%')]

        for n in range(self.freqai_info["feature_parameters"]["shift"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [(s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]]
        df = df.drop(columns=skip_columns)

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # the configuration file parameters are stored here
        self.freqai_info = self.config["freqai"]
        self.pair = metadata['pair']

        # the following loops are necessary for building the features
        # indicated by the user in the configuration file.
        for tf in self.freqai_info["timeframes"]:
            dataframe = self.populate_any_indicators(self.pair, dataframe.copy(), tf,
                                                     coin=self.pair.split("/")[0] + "-")
            for pair in self.freqai_info["corr_pairlist"]:
                if metadata['pair'] in pair:
                    continue  # do not include whitelisted pair twice if it is in corr_pairlist
                dataframe = self.populate_any_indicators(
                    pair, dataframe.copy(), tf, coin=pair.split("/")[0] + "-"
                )

        # the model will return 4 values, its prediction, an indication of whether or not the
        # prediction should be accepted, the target mean/std values from the labels used during
        # each training period.
        (
            dataframe["prediction"],
            dataframe["do_predict"],
            dataframe["target_mean"],
            dataframe["target_std"],
        ) = self.model.bridge.start(dataframe, metadata, self)

        dataframe["target_roi"] = dataframe["target_mean"] + dataframe["target_std"] * 1.5
        dataframe["sell_roi"] = dataframe["target_mean"] - dataframe["target_std"] * 1
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        buy_conditions = [
            (dataframe["prediction"] > dataframe["target_roi"]) & (dataframe["do_predict"] == 1)
        ]

        if buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, buy_conditions), "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sell_conditions = [
            (dataframe["do_predict"] <= 0)
        ]
        if sell_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, sell_conditions), "sell"] = 1

        return dataframe

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])
