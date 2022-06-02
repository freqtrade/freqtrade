import logging
from functools import reduce

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.freqai.strategy_bridge import CustomModel
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, merge_informative_pair
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

    process_only_new_candles = False
    stoploss = -0.05
    use_sell_signal = True
    startup_candle_count: int = 300
    can_short = False

    linear_roi_offset = DecimalParameter(0.00, 0.02, default=0.005, space='sell',
                                         optimize=False, load=True)
    max_roi_time_long = IntParameter(0, 800, default=400, space='sell', optimize=False, load=True)

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

    def populate_any_indicators(self, metadata, pair, df, tf, informative=None, coin=""):
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

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods"]:

            t = int(t)
            informative['%-' + coin + "rsi-period_" + str(t)] = ta.RSI(informative, timeperiod=t)
            informative['%-' + coin + "mfi-period_" + str(t)] = ta.MFI(informative, timeperiod=t)
            informative['%-' + coin + "adx-period_" + str(t)] = ta.ADX(informative, window=t)
            informative[coin + "20sma-period_" + str(t)] = ta.SMA(informative, timeperiod=t)
            informative[coin + "21ema-period_" + str(t)] = ta.EMA(informative, timeperiod=t)
            informative['%-' + coin + "close_over_20sma-period_" +
                        str(t)] = (informative["close"] /
                                   informative[coin + "20sma-period_" + str(t)])

            informative['%-' + coin + "mfi-period_" + str(t)] = ta.MFI(informative, timeperiod=t)

            informative[coin + "ema21-period_" + str(t)] = ta.EMA(informative, timeperiod=t)
            informative[coin + "sma20-period_" + str(t)] = ta.SMA(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=t,
                                                stds=2.2)
            informative[coin + "bb_lowerband-period_" + str(t)] = bollinger["lower"]
            informative[coin + "bb_middleband-period_" + str(t)] = bollinger["mid"]
            informative[coin + "bb_upperband-period_" + str(t)] = bollinger["upper"]
            informative['%-' + coin + "bb_width-period_" + str(t)] = (
                informative[coin + "bb_upperband-period_" + str(t)] -
                informative[coin + "bb_lowerband-period_" + str(t)]
            ) / informative[coin + "bb_middleband-period_" + str(t)]
            informative['%-' + coin + "close-bb_lower-period_" + str(t)] = (
                informative["close"] / informative[coin + "bb_lowerband-period_" + str(t)]
            )

            informative['%-' + coin + "roc-period_" + str(t)] = ta.ROC(informative, timeperiod=t)
            informative['%-' + coin + "adx-period_" + str(t)] = ta.ADX(informative, window=t)

            macd = ta.MACD(informative, timeperiod=t)
            informative['%-' + coin + "macd-period_" + str(t)] = macd["macd"]

            informative['%-' + coin + "relative_volume-period_" + str(t)] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        informative['%-' + coin + "pct-change"] = informative["close"].pct_change()
        informative['%-' + coin + "raw_volume"] = informative["volume"]
        informative['%-' + coin + 'raw_price'] = informative['close']

        indicators = [col for col in informative if col.startswith('%')]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["shift"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [(s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this function to populate
        # indicators during training). Notice how we ensure not to add them multiple times
        if pair == metadata['pair'] and tf == self.timeframe:
            df['%-day_of_week'] = (df["date"].dt.dayofweek + 1) / 7
            df['%-hour_of_day'] = (df['date'].dt.hour + 1) / 25

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]
        self.pair = metadata['pair']

        # the following loops are necessary for building the features
        # indicated by the user in the configuration file.
        # All indicators must be populated by populate_any_indicators() for live functionality
        # to work correctly.
        for tf in self.freqai_info["timeframes"]:
            dataframe = self.populate_any_indicators(metadata, self.pair, dataframe.copy(), tf,
                                                     coin=self.pair.split("/")[0] + "-")
            for pair in self.freqai_info["corr_pairlist"]:
                if metadata['pair'] in pair:
                    continue  # do not include whitelisted pair twice if it is in corr_pairlist
                dataframe = self.populate_any_indicators(
                    metadata, pair, dataframe.copy(), tf, coin=pair.split("/")[0] + "-"
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

        dataframe["target_roi"] = dataframe["target_mean"] + dataframe["target_std"]
        dataframe["sell_roi"] = dataframe["target_mean"] - dataframe["target_std"]
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [
            df['do_predict'] == 1,
            df['prediction'] > df["target_roi"]
        ]

        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y,
                          enter_long_conditions), ["enter_long", "enter_tag"]] = (1, 'long')

        enter_short_conditions = [
            df['do_predict'] == 1,
            df['prediction'] < df["sell_roi"]
        ]

        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y,
                          enter_short_conditions), ["enter_short", "enter_tag"]] = (1, 'short')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df['do_predict'] == 1,
            df['prediction'] < df['sell_roi'] * 0.25
        ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            df['do_predict'] == 1,
            df['prediction'] > df['target_roi'] * 0.25
        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def custom_exit(self, pair: str, trade: Trade, current_time, current_rate,
                    current_profit, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        trade_date = timeframe_to_prev_date(self.config['timeframe'], trade.open_date_utc)
        trade_candle = dataframe.loc[(dataframe['date'] == trade_date)]

        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        follow_mode = self.config.get('freqai', {}).get('follow_mode', False)

        if not follow_mode:
            pair_dict = self.model.bridge.data_drawer.pair_dict
        else:
            pair_dict = self.model.bridge.data_drawer.follower_dict

        entry_tag = trade.enter_tag

        if 'prediction' + entry_tag not in pair_dict[pair]:
            with self.model.bridge.lock:
                pair_dict[pair]['prediction' + entry_tag] = abs(trade_candle['prediction'])
                if not follow_mode:
                    self.model.bridge.data_drawer.save_drawer_to_disk()
                else:
                    self.model.bridge.data_drawer.save_follower_dict_to_dist()
        else:
            if pair_dict[pair]['prediction' + entry_tag] > 0:
                roi_price = abs(trade_candle['prediction'])
            else:
                with self.model.bridge.lock:
                    pair_dict[pair]['prediction' + entry_tag] = abs(trade_candle['prediction'])
                    if not follow_mode:
                        self.model.bridge.data_drawer.save_drawer_to_disk()
                    else:
                        self.model.bridge.data_drawer.save_follower_dict_to_dist()

        roi_price = abs(trade_candle['prediction'])
        roi_time = self.max_roi_time_long.value

        roi_decay = roi_price * (1 - ((current_time - trade.open_date_utc).seconds) /
                                 (roi_time * 60))
        if roi_decay < 0:
            roi_decay = self.linear_roi_offset.value
        else:
            roi_decay += self.linear_roi_offset.value

        if current_profit > roi_decay:
            return 'roi_custom_win'

        if current_profit < -roi_decay:
            return 'roi_custom_loss'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time, **kwargs) -> bool:

        entry_tag = trade.enter_tag
        follow_mode = self.config.get('freqai', {}).get('follow_mode', False)
        if not follow_mode:
            pair_dict = self.model.bridge.data_drawer.pair_dict
        else:
            pair_dict = self.model.bridge.data_drawer.follower_dict

        with self.model.bridge.lock:
            pair_dict[pair]['prediction' + entry_tag] = 0
            if not follow_mode:
                self.model.bridge.data_drawer.save_drawer_to_disk()
            else:
                self.model.bridge.data_drawer.save_follower_dict_to_dist()

        return True

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time, entry_tag,
                            side: str, **kwargs) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == 'long':
            if rate > (last_candle['close'] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle['close'] * (1 - 0.0025)):
                return False

        return True
