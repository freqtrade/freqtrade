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


def get_full_model_path(config: Config) -> Path:
    """
    Returns default FreqAI model path
    :param config: Configuration dictionary
    """
    freqai_config: Dict[str, Any] = config["freqai"]
    return Path(
        config["user_data_dir"] / "models" / str(freqai_config.get("identifier"))
    )


def get_timerange_from_ready_models(models_path: Path) -> Tuple[TimeRange, str, Dict[str, Any]]:
    """
    Returns timerange information based on a FreqAI model directory
    :param models_path: FreqAI model path

    :returns: a Tuple with (backtesting_timerange: Timerange calculated from directory,
    backtesting_string_timerange: str timerange calculated from
    directory (format example '20020822-20220830'), \
    pairs_end_dates: Dict with pair and model end training dates info)
    """
    all_models_end_dates = []
    pairs_end_dates: Dict[str, Any] = get_pairs_timestamps_training_from_ready_models(models_path)
    for key in pairs_end_dates:
        for model_end_date in pairs_end_dates[key]:
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
    start = datetime.fromtimestamp(min(all_models_end_dates), tz=timezone.utc)
    stop = datetime.fromtimestamp(max(all_models_end_dates), tz=timezone.utc)
    end_date_string_timerange = stop
    if (
        finish_timestamp < int(datetime.now(tz=timezone.utc).timestamp()) and
        datetime.now(tz=timezone.utc).strftime('%Y%m%d') != stop.strftime('%Y%m%d')
    ):
        # add 1 day to string timerange to ensure BT module will load all dataframe data
        end_date_string_timerange = stop + timedelta(days=1)

    backtesting_string_timerange = (
        f"{start.strftime('%Y%m%d')}-{end_date_string_timerange.strftime('%Y%m%d')}"
    )
    backtesting_timerange = TimeRange(
        'date', 'date', min(all_models_end_dates), max(all_models_end_dates)
    )
    return backtesting_timerange, backtesting_string_timerange, pairs_end_dates


def get_pairs_timestamps_training_from_ready_models(models_path: Path) -> Dict[str, Any]:
    """
    Scan the models path and returns all pairs end training dates (timestamp)
    :param models_path: FreqAI model path

    :returns:
    :pairs_end_dates: Dict with pair and model end training dates info
    """
    pairs_end_dates: Dict[str, Any] = {}
    if not models_path.is_dir():
        raise OperationalException(
            'Model folders not found. Saved models are required '
            'to run backtest with the freqai-backtest-live-models option'
        )
    for model_dir in models_path.iterdir():
        if str(model_dir.name).startswith("sub-train"):
            model_end_date = int(model_dir.name.split("_")[1])
            pair = model_dir.name.split("_")[0].replace("sub-train-", "")
            model_file_name = (
                f"cb_{str(model_dir.name).replace('sub-train-', '').lower()}"
                "_model.joblib"
            )

            model_path_file = Path(model_dir / model_file_name)
            if model_path_file.is_file():
                if pair not in pairs_end_dates:
                    pairs_end_dates[pair] = []

                pairs_end_dates[pair].append(model_end_date)
    return pairs_end_dates
