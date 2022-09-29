"""
FreqAI generic functions
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def get_full_models_path(config: Config) -> Path:
    """
    Returns default FreqAI model path
    :param config: Configuration dictionary
    """
    freqai_config: Dict[str, Any] = config["freqai"]
    return Path(
        config["user_data_dir"] / "models" / str(freqai_config.get("identifier"))
    )


def get_timerange_and_assets_end_dates_from_ready_models(
        models_path: Path) -> Tuple[TimeRange, Dict[str, Any]]:
    """
    Returns timerange information based on a FreqAI model directory
    :param models_path: FreqAI model path

    :return: a Tuple with (Timerange calculated from directory and
    a Dict with pair and model end training dates info)
    """
    all_models_end_dates = []
    assets_end_dates: Dict[str, Any] = get_assets_timestamps_training_from_ready_models(models_path)
    for key in assets_end_dates:
        for model_end_date in assets_end_dates[key]:
            if model_end_date not in all_models_end_dates:
                all_models_end_dates.append(model_end_date)

    if len(all_models_end_dates) == 0:
        raise OperationalException(
            'At least 1 saved model is required to '
            'run backtest with the freqai-backtest-live-models option'
        )

    if len(all_models_end_dates) == 1:
        logger.warning(
            "Only 1 model was found. Backtesting will run with the "
            "timerange from the end of the training date to the current date"
        )

    finish_timestamp = int(datetime.now(tz=timezone.utc).timestamp())
    if len(all_models_end_dates) > 1:
        # After last model end date, use the same period from previous model
        # to finish the backtest
        all_models_end_dates.sort(reverse=True)
        finish_timestamp = all_models_end_dates[0] + \
            (all_models_end_dates[0] - all_models_end_dates[1])

    all_models_end_dates.append(finish_timestamp)
    all_models_end_dates.sort()
    start_date = (datetime(*datetime.fromtimestamp(min(all_models_end_dates)).timetuple()[:3],
                           tzinfo=timezone.utc))
    end_date = (datetime(*datetime.fromtimestamp(max(all_models_end_dates)).timetuple()[:3],
                         tzinfo=timezone.utc))

    # add 1 day to string timerange to ensure BT module will load all dataframe data
    end_date = end_date + timedelta(days=1)
    backtesting_timerange = TimeRange(
        'date', 'date', int(start_date.timestamp()), int(end_date.timestamp())
    )
    return backtesting_timerange, assets_end_dates


def get_assets_timestamps_training_from_ready_models(models_path: Path) -> Dict[str, Any]:
    """
    Scan the models path and returns all assets end training dates (timestamp)
    :param models_path: FreqAI model path

    :return: a Dict with asset and model end training dates info
    """
    assets_end_dates: Dict[str, Any] = {}
    if not models_path.is_dir():
        raise OperationalException(
            'Model folders not found. Saved models are required '
            'to run backtest with the freqai-backtest-live-models option'
        )
    for model_dir in models_path.iterdir():
        if str(model_dir.name).startswith("sub-train"):
            model_end_date = int(model_dir.name.split("_")[1])
            asset = model_dir.name.split("_")[0].replace("sub-train-", "")
            model_file_name = (
                f"cb_{str(model_dir.name).replace('sub-train-', '').lower()}"
                "_model.joblib"
            )

            model_path_file = Path(model_dir / model_file_name)
            if model_path_file.is_file():
                if asset not in assets_end_dates:
                    assets_end_dates[asset] = []
                assets_end_dates[asset].append(model_end_date)

    return assets_end_dates


def get_timerange_backtest_live_models(config: Config):
    """
    Returns a formated timerange for backtest live/ready models
    :param config: Configuration dictionary

    :return: a string timerange (format example: '20220801-20220822')
    """
    models_path = get_full_models_path(config)
    timerange, _ = get_timerange_and_assets_end_dates_from_ready_models(models_path)
    start_date = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
    end_date = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
    tr = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    return tr
