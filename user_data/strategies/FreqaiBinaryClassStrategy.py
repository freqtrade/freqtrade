from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import reduce
from turtle import update
from h11 import Data
from datetime import datetime, timedelta, timezone
import pandas as pd
import talib.abstract as ta
from pandas_ta.trend import adx
from pandas import DataFrame
from technical import qtpylib
import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import precision_recall_curve
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import DecimalParameter, IntParameter, merge_informative_pair
from freqtrade.strategy.interface import IStrategy
# from freqtrade.enums import DoPredictType


logger = logging.getLogger(__name__)


def find_support_levels(df: DataFrame) -> DataFrame:
    """
    cond1 = df['Low'][i] < df['Low'][i-1]   
    cond2 = df['Low'][i] < df['Low'][i+1]   
    cond3 = df['Low'][i+1] < df['Low'][i+2]   
    cond4 = df['Low'][i-1] < df['Low'][i-2]  
    """
    cond1 = df["low"] < df["low"].shift(1)
    cond2 = df["low"] < df["low"].shift(-1)
    cond3 = df["low"].shift(-1) < df["low"].shift(-2)
    cond4 = df["low"].shift(1) < df["low"].shift(2)
    return (cond1 & cond2 & cond3 & cond4)


class FreqaiBinaryClassStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.model = CustomModel(self.config)
    self.model.bridge.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {
                    "color": "brown"
                }
            },
            "DI_values": {
                "DI_values": {
                    "color": "#8115a9",
                    "type": "line"
                }
            },
            "GTs": {
                "tp_max": {
                    "color": "#69796a",
                    "type": "bar"
                },
                "tp_min": {
                    "color": "#e2517f",
                    "type": "bar"
                },
                 "max": {
                     "color": "#69796a",
                     "type": "line"
                 },
                 "min": {
                    "color": "#e2517f",
                    "type": "line"
                },
                 "neutral": {
                     "color": "#ffffff",
                    "type": "line"
                 }
            }
        }
    }

    position_adjustment_enable = False

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True

    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)

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
        :params:
        :pair: pair to be used as informative
        :df: strategy dataframe which will receive merges from informatives
        :tf: timeframe of the dataframe which will modify the feature names
        :informative: the dataframe associated with the informative pair
        :coin: the name of the coin which will modify the feature names.
        """

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            out = adx(informative["high"], informative["low"], informative["close"], window=t)
            informative[f"%-{coin}adx-period_{t}"] = out["ADX_14"]
            informative[f"%-{coin}diplus-period_{t}"] = out["DMP_14"]
            informative[f"%-{coin}diminus-period_{t}"] = out["DMN_14"]

            informative[f"{coin}20sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            #informative[f"{coin}21ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            informative[f"%-{coin}close_over_20sma-period_{t}"] = (
                informative["close"] / informative[f"{coin}20sma-period_{t}"]
            )

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
            macd = ta.MACD(informative, timeperiod=t)
            informative[f"%-{coin}macd-period_{t}"] = macd["macd"]

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

        # find support levels
        if tf == self.freqai_info["feature_parameters"]["include_timeframes"][-1]:
            informative_6h = resample_to_interval(informative, "6h")
            informative_6h["support_levels"] = find_support_levels(informative_6h)
            df = merge_informative_pair(df, informative_6h, self.config["timeframe"], "6h", ffill=True)

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

            # user adds targets here by prepending them with &- (see convention below)
            # If user wishes to use multiple targets, a multioutput prediction model
            # needs to be used such as templates/CatboostPredictionMultiModel.py
            #df["&s-minima"] = FreqaiBinaryClassStrategy.get_min_labels(df)
            #df["&s-maxima"] = FreqaiBinaryClassStrategy.get_max_labels(df)
            minmax = np.array(["neutral"] * len(df))
            minmax[FreqaiBinaryClassStrategy.get_min_labels(df) == 1] = "min"
            minmax[FreqaiBinaryClassStrategy.get_max_labels(df) == 1] = "max"
            df["&s-minmax"] = np.array([str(x) for x in minmax]).astype(np.object0)
        return df

    @staticmethod
    def get_max_labels(df: DataFrame) -> DataFrame:
        labels = np.zeros(len(df), dtype=np.int32)
        col = "close"
        max_peaks = argrelextrema(df[col].values, np.greater, order=12)[0]
        nn = 2
        out = adx(df["high"], df["low"], df["close"], window=12)
        diplus = out["DMP_14"]
        price = (df['high'] + df['low'] + df['close']) / 3
        di_thr = min(35, np.nanmax(diplus[max_peaks]))
        for mp in max_peaks:
            ref_close = price.iloc[mp]
            start = max(0, mp-nn)
            end = min(df.shape[0], mp+nn+1)
            pct = np.abs(price[start:end] / ref_close - 1)
            is_close = np.where(pct <= 0.005)[0]
            left_idx = is_close[0]
            right_idx = is_close[-1]
            # locality labeling
            if diplus[mp-nn+left_idx:mp-nn+right_idx].mean() >= di_thr:
                labels[mp-nn+left_idx:mp-nn+right_idx] = 1
        if labels.max() == 0:  # if not any positive label is found, we force it
            idx = np.nanargmax(diplus[max_peaks])
            labels[max_peaks[idx]] = 1
        return labels
    
    @staticmethod
    def get_min_labels(df: DataFrame) -> DataFrame:
        labels = np.zeros(len(df), dtype=np.int32)
        col = "close"
        min_peaks = argrelextrema(df[col].values, np.less, order=12)[0]
        nn = 2
        out = adx(df["high"], df["low"], df["close"], window=12)
        diminus = out["DMN_14"]
        price = (df['high'] + df['low'] + df['close']) / 3
        di_thr = min(35, np.nanmax(diminus[min_peaks]))
        for mp in min_peaks:
            ref_close = price.iloc[mp]
            start = max(0, mp-nn)
            end = min(df.shape[0], mp+nn+1)
            pct = np.abs(price[start:end] / ref_close - 1)
            is_close = np.where(pct <= 0.005)[0]
            left_idx = is_close[0]
            right_idx = is_close[-1]
            # locality labeling
            if diminus[mp-nn+left_idx:mp-nn+right_idx].mean() >= di_thr:
                labels[mp-nn+left_idx:mp-nn+right_idx] = 1
        # return np.array([str(x) for x in labels]).astype(np.object0)
        if labels.max() == 0:  # if not any positive label is found, we force it
            idx = np.nanargmax(diminus[min_peaks])
            labels[min_peaks[idx]] = 1
        return labels

    @staticmethod
    def expand_labels(df: DataFrame, peaks: List[int]):
        col = "close"
        nn = 2
        labels = np.zeros(len(df), dtype=np.int32)
        for p in peaks:
            ref_close = df[col][p]
            for idx in range(max(0, p - nn), min(p + nn, df.shape[0]-1)):
                if np.abs(df[col][idx] / ref_close - 1) <= 0.005:
                    labels[idx] = 1
        return np.array([str(x) for x in labels]).astype(np.object0)

    @staticmethod
    def get_labels(df: DataFrame, metadata: dict) -> DataFrame:
        col = "close"
        min_peaks = argrelextrema(df[col].values, np.less, order=24)
        min_peaks = min_peaks[0]
        max_peaks = argrelextrema(df[col].values, np.greater, order=24)
        max_peaks = max_peaks[0]
        # import epdb; epdb.set_trace()
        peaks = sorted(set(min_peaks).union(set(max_peaks)))
        updown = None
        max_peaks2 = []
        min_peaks2 = []
        for idx in peaks:
            if (idx in min_peaks and idx in max_peaks):
                continue
            if idx in min_peaks:
                if updown is None or updown == True:
                    updown = False
                    min_peaks2.append(idx)
                else:
                    if df[col][min_peaks2[-1]] < df[col][idx]:
                        continue
                    else:
                        min_peaks2[-1] = idx
                    
            elif idx in max_peaks:
                if updown is None or updown == False:
                    updown = True
                    max_peaks2.append(idx)
                else:
                    if df[col][max_peaks2[-1]] > df[col][idx]:
                        continue
                    else:
                        max_peaks2[-1] = idx
        profits = [df["close"][y] / df["close"][x] - 1 for x, y in zip(min_peaks2, max_peaks2)]
        print(metadata, np.mean(profits))
        #print(metadata, df["date"][min_peaks2])
        #print(metadata, df["date"][max_peaks2])
        min_peaks = FreqaiBinaryClassStrategy.expand_labels(df, min_peaks2)
        max_peaks = FreqaiBinaryClassStrategy.expand_labels(df, max_peaks2)
        return min_peaks, max_peaks

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        # the model will return 4 values, its prediction, an indication of whether or not the
        # prediction should be accepted, the target mean/std values from the labels used during
        # each training period.
        dataframe = self.freqai.start(dataframe, metadata, self)
        # dataframe["&s-minima"] = dataframe["&s-minima"].astype(np.float32)
        # dataframe["&s-maxima"] = dataframe["&s-maxima"].astype(np.float32)
        max_labels = FreqaiBinaryClassStrategy.get_max_labels(dataframe)
        min_labels = FreqaiBinaryClassStrategy.get_min_labels(dataframe)

        self.maxima_threhsold = 0.4
        self.minima_threhsold = 0.4

        dataframe["tp_max"] = max_labels.astype(np.float32)
        dataframe["tp_min"] = min_labels.astype(np.float32)
        dataframe["di-"] = ta.MINUS_DI(dataframe, window=12)
        dataframe["di+"] = ta.PLUS_DI(dataframe, window=12)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df["do_predict"] == 1, df["min"] >= self.minima_threhsold]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
            
        enter_missed_long_conditions = [df["do_predict"] == 1, df["tp_min"].shift(1) == 1]
        if enter_missed_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_missed_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long_missed")

        if self.can_short:
            enter_short_conditions = [df["do_predict"] == 1, df["max"] >= self.maxima_threhsold]

            if enter_short_conditions:
                df.loc[
                    reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
                ] = (1, "short")
            
            enter_missed_short_conditions = [df["do_predict"] == 1, df["tp_max"].shift(1) == 1]
            if enter_missed_short_conditions:
                df.loc[
                    reduce(lambda x, y: x & y, enter_missed_short_conditions), ["enter_short", "enter_tag"]
                ] = (1, "short_missed")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["max"] >= self.maxima_threhsold]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions),
                   ["exit_long", "exit_tag"]] = (1, "exit signal")

        exit_long_missed_conditions = [df["do_predict"] == 1, df["tp_max"].shift(1) == 1]
        if exit_long_missed_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_missed_conditions),
                   ["exit_long", "exit_tag"]] = (1, "exit_long_missed")

        if self.can_short:
            exit_short_conditions = [df["do_predict"] == 1, df["min"] >= self.minima_threhsold]
            if exit_short_conditions:
                df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

            exit_short_missed_conditions = [df["do_predict"] == 1, df["tp_min"].shift(1) == 1]
            if exit_short_missed_conditions:
                df.loc[reduce(lambda x, y: x & y, exit_short_missed_conditions),
                       ["exit_short", "exit_tag"]] = (1, "exit_short_missed")
        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])
    """
    def custom_exit(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        trade_date = timeframe_to_prev_date(self.config["timeframe"], trade.open_date_utc)
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]

        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)

        if not follow_mode:
            pair_dict = self.model.bridge.dd.pair_dict
        else:
            pair_dict = self.model.bridge.dd.follower_dict

        entry_tag = trade.enter_tag

        if (
            "prediction" + entry_tag not in pair_dict[pair]
            or pair_dict[pair]["prediction" + entry_tag] > 0
        ):
            with self.model.bridge.lock:
                if entry_tag == "long":
                    pair_dict[pair]["prediction" + entry_tag] = abs(trade_candle["&s-maxima"])
                else:
                    pair_dict[pair]["prediction" + entry_tag] = abs(trade_candle["&-s_close"])
                if not follow_mode:
                    self.model.bridge.dd.save_drawer_to_disk()
                else:
                    self.model.bridge.dd.save_follower_dict_to_disk()

        roi_price = pair_dict[pair]["prediction" + entry_tag]
        roi_time = self.max_roi_time_long.value

        roi_decay = roi_price * (
            1 - ((current_time - trade.open_date_utc).seconds) / (roi_time * 60)
        )
        if roi_decay < 0:
            roi_decay = self.linear_roi_offset.value
        else:
            roi_decay += self.linear_roi_offset.value

        if current_profit > roi_decay:
            return "roi_custom_win"

        if current_profit < -roi_decay:
            return "roi_custom_loss"
    """
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:

        entry_tag = trade.enter_tag
        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)
        if not follow_mode:
            pair_dict = self.freqai.dd.pair_dict
        else:
            pair_dict = self.freqai.dd.follower_dict

        pair_dict[pair]["prediction" + entry_tag] = 0
        if not follow_mode:
            self.freqai.dd.save_drawer_to_disk()
        else:
            self.freqai.dd.save_follower_dict_to_disk()

        return True

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """
        if not trade.is_short:
            if current_profit < -0.02:
                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                try:
                    new_local_minima = [df["&s-minima"] > self.minima_threhsold,
                                        (df["close"] / current_rate - 1) < 1e-3]
                    if df.shape[0] - df.loc[reduce(lambda x, y: x & y, new_local_minima)].index[-1] <= 10:
                        return 20
                except:
                    pass
        return None
