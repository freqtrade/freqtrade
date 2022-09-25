import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
# for spice rack
import pandas as pd
import talib.abstract as ta
from scipy.signal import argrelextrema
from technical import qtpylib

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, timeframe_to_seconds
from freqtrade.exchange.exchange import market_is_active
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.strategy import merge_informative_pair


logger = logging.getLogger(__name__)


def download_all_data_for_training(dp: DataProvider, config: Config) -> None:
    """
    Called only once upon start of bot to download the necessary data for
    populating indicators and training the model.
    :param timerange: TimeRange = The full data timerange for populating the indicators
                                    and training the model.
    :param dp: DataProvider instance attached to the strategy
    """

    if dp._exchange is None:
        raise OperationalException('No exchange object found.')
    markets = [p for p, m in dp._exchange.markets.items() if market_is_active(m)
               or config.get('include_inactive')]

    all_pairs = dynamic_expand_pairlist(config, markets)

    timerange = get_required_data_timerange(config)

    new_pairs_days = int((timerange.stopts - timerange.startts) / 86400)

    refresh_backtest_ohlcv_data(
        dp._exchange,
        pairs=all_pairs,
        timeframes=config["freqai"]["feature_parameters"].get("include_timeframes"),
        datadir=config["datadir"],
        timerange=timerange,
        new_pairs_days=new_pairs_days,
        erase=False,
        data_format=config.get("dataformat_ohlcv", "json"),
        trading_mode=config.get("trading_mode", "spot"),
        prepend=config.get("prepend_data", False),
    )


def get_required_data_timerange(config: Config) -> TimeRange:
    """
    Used to compute the required data download time range
    for auto data-download in FreqAI
    """
    time = datetime.now(tz=timezone.utc).timestamp()

    timeframes = config["freqai"]["feature_parameters"].get("include_timeframes")

    max_tf_seconds = 0
    for tf in timeframes:
        secs = timeframe_to_seconds(tf)
        if secs > max_tf_seconds:
            max_tf_seconds = secs

    startup_candles = config.get('startup_candle_count', 0)
    indicator_periods = config["freqai"]["feature_parameters"]["indicator_periods_candles"]

    # factor the max_period as a factor of safety.
    max_period = int(max(startup_candles, max(indicator_periods)) * 1.5)
    config['startup_candle_count'] = max_period
    logger.info(f'FreqAI auto-downloader using {max_period} startup candles.')

    additional_seconds = max_period * max_tf_seconds

    startts = int(
        time
        - config["freqai"].get("train_period_days", 0) * 86400
        - additional_seconds
    )
    stopts = int(time)
    data_load_timerange = TimeRange('date', 'date', startts, stopts)

    return data_load_timerange


def auto_populate_any_indicators(
    self, pair, df, tf, informative=None, set_generalized_indicators=False
):
    """
    This is a premade `populate_any_indicators()` function which is set in
    the user strategy is they enable `freqai_spice_rack: true` in their
    configuration file.
    """

    coin = pair.split('/')[0]

    if informative is None:
        informative = self.dp.get_pair_dataframe(pair, tf)

    # first loop is automatically duplicating indicators for time periods
    for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

        t = int(t)
        informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
        informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
        informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
        informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
        informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

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
    if set_generalized_indicators:
        df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
        df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25
        df["&s-minima"] = 0
        df["&s-maxima"] = 0
        min_peaks = argrelextrema(df["close"].values, np.less, order=80)
        max_peaks = argrelextrema(df["close"].values, np.greater, order=80)
        for mp in min_peaks[0]:
            df.at[mp, "&s-minima"] = 1
        for mp in max_peaks[0]:
            df.at[mp, "&s-maxima"] = 1

    return df


def setup_freqai_spice_rack(config: dict, exchange: Optional[Exchange]) -> Dict[str, Any]:
    import difflib
    import json
    from pathlib import Path
    auto_config = config.get('freqai_config', 'lightgbm_config.json')
    with open(Path(__file__).parent / Path('spice_rack') / auto_config) as json_file:
        freqai_config = json.load(json_file)
        config['freqai'] = freqai_config['freqai']
        config['freqai']['identifier'] = config['freqai_identifier']
        corr_pairs = config['freqai']['feature_parameters']['include_corr_pairlist']
        timeframes = config['freqai']['feature_parameters']['include_timeframes']
        new_corr_pairs = []
        new_tfs = []

        if not exchange:
            logger.warning('No dataprovider available.')
            config['freqai']['enabled'] = False
            return config
        # find the closest pairs to what the default config wants
        for pair in corr_pairs:
            closest_pair = difflib.get_close_matches(
                                        pair,
                                        exchange.markets
                                        )
            if not closest_pair:
                logger.warning(f'Could not find {pair} in markets, removing from '
                               f'corr_pairlist.')
            else:
                closest_pair = closest_pair[0]

            new_corr_pairs.append(closest_pair)
            logger.info(f'Spice rack will use {closest_pair} as informative in FreqAI model.')

        # find the closest matching timeframes to what the default config wants
        if timeframe_to_seconds(config['timeframe']) > timeframe_to_seconds('15m'):
            logger.warning('Default spice rack is designed for lower base timeframes (e.g. > '
                           f'15m). But user passed {config["timeframe"]}.')
        new_tfs.append(config['timeframe'])

        list_tfs = [timeframe_to_seconds(tf) for tf
                    in exchange.timeframes]
        for tf in timeframes:
            tf_secs = timeframe_to_seconds(tf)
            closest_index = min(range(len(list_tfs)), key=lambda i: abs(list_tfs[i] - tf_secs))
            closest_tf = exchange.timeframes[closest_index]
            logger.info(f'Spice rack will use {closest_tf} as informative tf in FreqAI model.')
            new_tfs.append(closest_tf)

    config['freqai']['feature_parameters'].update({'include_timeframes': new_tfs})
    config['freqai']['feature_parameters'].update({'include_corr_pairlist': new_corr_pairs})
    config.update({"freqaimodel": 'LightGBMRegressorMultiTarget'})
    return config

# Keep below for when we wish to download heterogeneously lengthed data for FreqAI.
# def download_all_data_for_training(dp: DataProvider, config: Config) -> None:
#     """
#     Called only once upon start of bot to download the necessary data for
#     populating indicators and training a FreqAI model.
#     :param timerange: TimeRange = The full data timerange for populating the indicators
#                                     and training the model.
#     :param dp: DataProvider instance attached to the strategy
#     """

#     if dp._exchange is not None:
#         markets = [p for p, m in dp._exchange.markets.items() if market_is_active(m)
#                    or config.get('include_inactive')]
#     else:
#         # This should not occur:
#         raise OperationalException('No exchange object found.')

#     all_pairs = dynamic_expand_pairlist(config, markets)

#     if not dp._exchange:
#         # Not realistic - this is only called in live mode.
#         raise OperationalException("Dataprovider did not have an exchange attached.")

#     time = datetime.now(tz=timezone.utc).timestamp()

#     for tf in config["freqai"]["feature_parameters"].get("include_timeframes"):
#         timerange = TimeRange()
#         timerange.startts = int(time)
#         timerange.stopts = int(time)
#         startup_candles = dp.get_required_startup(str(tf))
#         tf_seconds = timeframe_to_seconds(str(tf))
#         timerange.subtract_start(tf_seconds * startup_candles)
#         new_pairs_days = int((timerange.stopts - timerange.startts) / 86400)
#         # FIXME: now that we are looping on `refresh_backtest_ohlcv_data`, the function
#         # redownloads the funding rate for each pair.
#         refresh_backtest_ohlcv_data(
#             dp._exchange,
#             pairs=all_pairs,
#             timeframes=[tf],
#             datadir=config["datadir"],
#             timerange=timerange,
#             new_pairs_days=new_pairs_days,
#             erase=False,
#             data_format=config.get("dataformat_ohlcv", "json"),
#             trading_mode=config.get("trading_mode", "spot"),
#             prepend=config.get("prepend_data", False),
#         )


def plot_feature_importance(model: Any, pair: str, dk: FreqaiDataKitchen,
                            count_max: int = 25) -> None:
    """
        Plot Best and worst features by importance for a single sub-train.
        :param model: Any = A model which was `fit` using a common library
                            such as catboost or lightgbm
        :param pair: str = pair e.g. BTC/USD
        :param dk: FreqaiDataKitchen = non-persistent data container for current coin/loop
        :param count_max: int = the amount of features to be loaded per column
    """
    from freqtrade.plot.plotting import go, make_subplots, store_plot_file

    # Extract feature importance from model
    models = {}
    if 'FreqaiMultiOutputRegressor' in str(model.__class__):
        for estimator, label in zip(model.estimators_, dk.label_list):
            models[label] = estimator
    else:
        models[dk.label_list[0]] = model

    for label in models:
        mdl = models[label]
        if "catboost.core" in str(mdl.__class__):
            feature_importance = mdl.get_feature_importance()
        elif "lightgbm.sklearn" or "xgb" in str(mdl.__class__):
            feature_importance = mdl.feature_importances_
        else:
            logger.info('Model type not support for generating feature importances.')
            return

        # Data preparation
        fi_df = pd.DataFrame({
            "feature_names": np.array(dk.data_dictionary['train_features'].columns),
            "feature_importance": np.array(feature_importance)
        })
        fi_df_top = fi_df.nlargest(count_max, "feature_importance")[::-1]
        fi_df_worst = fi_df.nsmallest(count_max, "feature_importance")[::-1]

        # Plotting
        def add_feature_trace(fig, fi_df, col):
            return fig.add_trace(
                go.Bar(
                    x=fi_df["feature_importance"],
                    y=fi_df["feature_names"],
                    orientation='h', showlegend=False
                ), row=1, col=col
            )
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.5)
        fig = add_feature_trace(fig, fi_df_top, 1)
        fig = add_feature_trace(fig, fi_df_worst, 2)
        fig.update_layout(title_text=f"Best and worst features by importance {pair}")
        label = label.replace('&', '').replace('%', '')  # escape two FreqAI specific characters
        store_plot_file(fig, f"{dk.model_filename}-{label}.html", dk.data_path)
