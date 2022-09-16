import logging
from datetime import datetime, timezone
# for plot_feature_importance
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange import market_is_active
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist


logger = logging.getLogger(__name__)


def download_all_data_for_training(dp: DataProvider, config: dict) -> None:
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


def get_required_data_timerange(
    config: dict
) -> TimeRange:
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


# Keep below for when we wish to download heterogeneously lengthed data for FreqAI.
# def download_all_data_for_training(dp: DataProvider, config: dict) -> None:
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


def plot_feature_importance(model, feature_names, pair, train_dir, count_max=25) -> None:
    """
        Plot Best and Worst Features by importance for CatBoost model.
        Called once per sub-train.

        Required: pip install kaleido

        Usage: plot_feature_importance(
            model=model,
            feature_names=dk.training_features_list,
            pair=pair,
            train_dir=dk.data_path)
    """

    # Gather feature importance from model
    if "catboost.core" in str(model.__class__):
        fi = model.get_feature_importance()

    elif "lightgbm.sklearn" in str(model.__class__):
        fi = model.feature_importances_

    else:
        raise NotImplementedError(f"Cannot extract feature importance for {model.__class__}")

    # Data preparation
    fi_df = pd.DataFrame({
        "feature_names": np.array(feature_names),
        "feature_importance": np.array(fi)
    })
    fi_df_top = fi_df.nlargest(count_max, "feature_importance")[::-1]
    fi_df_worst = fi_df.nsmallest(count_max, "feature_importance")[::-1]

    # Plotting
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.5)
    fig.add_trace(
        go.Bar(
            x=fi_df_top["feature_importance"],
            y=fi_df_top["feature_names"],
            orientation='h', showlegend=False
        ), row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=fi_df_worst["feature_importance"],
            y=fi_df_worst["feature_names"],
            orientation='h', showlegend=False
        ), row=1, col=2
    )
    fig.update_layout(
        title_text=f"Best and Worst Features {pair}",
        width=1000, height=600
    )

    # Create directory and save image
    model_dir, train_name = str(train_dir).rsplit("/", 1)
    fi_dir = Path(f"{model_dir}/feature_importance/{pair.split('/')[0]}")
    fi_dir.mkdir(parents=True, exist_ok=True)

    pio.write_image(fig, f"{fi_dir}/{train_name}.png", format="png")

    logger.info(f"Freqai saving feature importance plot {fi_dir}/{train_name}.png")
