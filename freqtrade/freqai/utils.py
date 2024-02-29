import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import rapidjson

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist


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
    markets = [
        p for p in dp._exchange.get_markets(
            tradable_only=True, active_only=not config.get('include_inactive')
            ).keys()
    ]

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
        data_format=config.get("dataformat_ohlcv", "feather"),
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
        elif "lightgbm.sklearn" in str(mdl.__class__):
            feature_importance = mdl.feature_importances_
        elif "xgb" in str(mdl.__class__):
            feature_importance = mdl.feature_importances_
        else:
            logger.info('Model type does not support generating feature importances.')
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


def record_params(config: Dict[str, Any], full_path: Path) -> None:
    """
    Records run params in the full path for reproducibility
    """
    params_record_path = full_path / "run_params.json"

    run_params = {
        "freqai": config.get('freqai', {}),
        "timeframe": config.get('timeframe'),
        "stake_amount": config.get('stake_amount'),
        "stake_currency": config.get('stake_currency'),
        "max_open_trades": config.get('max_open_trades'),
        "pairs": config.get('exchange', {}).get('pair_whitelist')
    }

    with params_record_path.open("w") as handle:
        rapidjson.dump(
            run_params,
            handle,
            indent=4,
            default=str,
            number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN
        )


def get_timerange_backtest_live_models(config: Config) -> str:
    """
    Returns a formated timerange for backtest live/ready models
    :param config: Configuration dictionary

    :return: a string timerange (format example: '20220801-20220822')
    """
    dk = FreqaiDataKitchen(config)
    models_path = dk.get_full_models_path(config)
    dd = FreqaiDataDrawer(models_path, config)
    timerange = dd.get_timerange_from_live_historic_predictions()
    return timerange.timerange_str


def get_tb_logger(model_type: str, path: Path, activate: bool) -> Any:

    if model_type == "pytorch" and activate:
        from freqtrade.freqai.tensorboard import TBLogger
        return TBLogger(path, activate)
    else:
        from freqtrade.freqai.tensorboard.base_tensorboard import BaseTensorboardLogger
        return BaseTensorboardLogger(path, activate)
