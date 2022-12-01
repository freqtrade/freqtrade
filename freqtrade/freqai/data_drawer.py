import collections
import importlib
import logging
import re
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict

import numpy as np
import pandas as pd
import psutil
import rapidjson
from joblib import dump, load
from joblib.externals import cloudpickle
from numpy.typing import NDArray
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.history import load_pair_history
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class pair_info(TypedDict):
    model_filename: str
    trained_timestamp: int
    data_path: str
    extras: dict


class FreqaiDataDrawer:
    """
    Class aimed at holding all pair models/info in memory for better inferencing/retrainig/saving
    /loading to/from disk.
    This object remains persistent throughout live/dry.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @th0rntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(self, full_path: Path, config: Config, follow_mode: bool = False):

        self.config = config
        self.freqai_info = config.get("freqai", {})
        # dictionary holding all pair metadata necessary to load in from disk
        self.pair_dict: Dict[str, pair_info] = {}
        # dictionary holding all actively inferenced models in memory given a model filename
        self.model_dictionary: Dict[str, Any] = {}
        # all additional metadata that we want to keep in ram
        self.meta_data_dictionary: Dict[str, Dict[str, Any]] = {}
        self.model_return_values: Dict[str, DataFrame] = {}
        self.historic_data: Dict[str, Dict[str, DataFrame]] = {}
        self.historic_predictions: Dict[str, DataFrame] = {}
        self.follower_dict: Dict[str, pair_info] = {}
        self.full_path = full_path
        self.follower_name: str = self.config.get("bot_name", "follower1")
        self.follower_dict_path = Path(
            self.full_path / f"follower_dictionary-{self.follower_name}.json"
        )
        self.historic_predictions_path = Path(self.full_path / "historic_predictions.pkl")
        self.historic_predictions_bkp_path = Path(
            self.full_path / "historic_predictions.backup.pkl")
        self.pair_dictionary_path = Path(self.full_path / "pair_dictionary.json")
        self.global_metadata_path = Path(self.full_path / "global_metadata.json")
        self.metric_tracker_path = Path(self.full_path / "metric_tracker.json")
        self.follow_mode = follow_mode
        if follow_mode:
            self.create_follower_dict()
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()
        self.metric_tracker: Dict[str, Dict[str, Dict[str, list]]] = {}
        self.load_metric_tracker_from_disk()
        self.training_queue: Dict[str, int] = {}
        self.history_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.pair_dict_lock = threading.Lock()
        self.metric_tracker_lock = threading.Lock()
        self.old_DBSCAN_eps: Dict[str, float] = {}
        self.empty_pair_dict: pair_info = {
                "model_filename": "", "trained_timestamp": 0,
                "data_path": "", "extras": {}}
        self.model_type = self.freqai_info.get('model_save_type', 'joblib')

    def update_metric_tracker(self, metric: str, value: float, pair: str) -> None:
        """
        General utility for adding and updating custom metrics. Typically used
        for adding training performance, train timings, inferenc timings, cpu loads etc.
        """
        with self.metric_tracker_lock:
            if pair not in self.metric_tracker:
                self.metric_tracker[pair] = {}
            if metric not in self.metric_tracker[pair]:
                self.metric_tracker[pair][metric] = {'timestamp': [], 'value': []}

            timestamp = int(datetime.now(timezone.utc).timestamp())
            self.metric_tracker[pair][metric]['value'].append(value)
            self.metric_tracker[pair][metric]['timestamp'].append(timestamp)

    def collect_metrics(self, time_spent: float, pair: str):
        """
        Add metrics to the metric tracker dictionary
        """
        load1, load5, load15 = psutil.getloadavg()
        cpus = psutil.cpu_count()
        self.update_metric_tracker('train_time', time_spent, pair)
        self.update_metric_tracker('cpu_load1min', load1 / cpus, pair)
        self.update_metric_tracker('cpu_load5min', load5 / cpus, pair)
        self.update_metric_tracker('cpu_load15min', load15 / cpus, pair)

    def load_global_metadata_from_disk(self):
        """
        Locate and load a previously saved global metadata in present model folder.
        """
        exists = self.global_metadata_path.is_file()
        if exists:
            with open(self.global_metadata_path, "r") as fp:
                metatada_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                return metatada_dict
        return {}

    def load_drawer_from_disk(self):
        """
        Locate and load a previously saved data drawer full of all pair model metadata in
        present model folder.
        Load any existing metric tracker that may be present.
        """
        exists = self.pair_dictionary_path.is_file()
        if exists:
            with open(self.pair_dictionary_path, "r") as fp:
                self.pair_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        elif not self.follow_mode:
            logger.info("Could not find existing datadrawer, starting from scratch")
        else:
            logger.warning(
                f"Follower could not find pair_dictionary at {self.full_path} "
                "sending null values back to strategy"
            )

    def load_metric_tracker_from_disk(self):
        """
        Tries to load an existing metrics dictionary if the user
        wants to collect metrics.
        """
        if self.freqai_info.get('write_metrics_to_disk', False):
            exists = self.metric_tracker_path.is_file()
            if exists:
                with open(self.metric_tracker_path, "r") as fp:
                    self.metric_tracker = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                logger.info("Loading existing metric tracker from disk.")
            else:
                logger.info("Could not find existing metric tracker, starting from scratch")

    def load_historic_predictions_from_disk(self):
        """
        Locate and load a previously saved historic predictions.
        :return: bool - whether or not the drawer was located
        """
        exists = self.historic_predictions_path.is_file()
        if exists:
            try:
                with open(self.historic_predictions_path, "rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.info(
                    f"Found existing historic predictions at {self.full_path}, but beware "
                    "that statistics may be inaccurate if the bot has been offline for "
                    "an extended period of time."
                )
            except EOFError:
                logger.warning(
                    'Historical prediction file was corrupted. Trying to load backup file.')
                with open(self.historic_predictions_bkp_path, "rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.warning('FreqAI successfully loaded the backup historical predictions file.')

        elif not self.follow_mode:
            logger.info("Could not find existing historic_predictions, starting from scratch")
        else:
            logger.warning(
                f"Follower could not find historic predictions at {self.full_path} "
                "sending null values back to strategy"
            )

        return exists

    def save_historic_predictions_to_disk(self):
        """
        Save historic predictions pickle to disk
        """
        with open(self.historic_predictions_path, "wb") as fp:
            cloudpickle.dump(self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL)

        # create a backup
        shutil.copy(self.historic_predictions_path, self.historic_predictions_bkp_path)

    def save_metric_tracker_to_disk(self):
        """
        Save metric tracker of all pair metrics collected.
        """
        with self.save_lock:
            with open(self.metric_tracker_path, 'w') as fp:
                rapidjson.dump(self.metric_tracker, fp, default=self.np_encoder,
                               number_mode=rapidjson.NM_NATIVE)

    def save_drawer_to_disk(self):
        """
        Save data drawer full of all pair model metadata in present model folder.
        """
        with self.save_lock:
            with open(self.pair_dictionary_path, 'w') as fp:
                rapidjson.dump(self.pair_dict, fp, default=self.np_encoder,
                               number_mode=rapidjson.NM_NATIVE)

    def save_follower_dict_to_disk(self):
        """
        Save follower dictionary to disk (used by strategy for persistent prediction targets)
        """
        with open(self.follower_dict_path, "w") as fp:
            rapidjson.dump(self.follower_dict, fp, default=self.np_encoder,
                           number_mode=rapidjson.NM_NATIVE)

    def save_global_metadata_to_disk(self, metadata: Dict[str, Any]):
        """
        Save global metadata json to disk
        """
        with self.save_lock:
            with open(self.global_metadata_path, 'w') as fp:
                rapidjson.dump(metadata, fp, default=self.np_encoder,
                               number_mode=rapidjson.NM_NATIVE)

    def create_follower_dict(self):
        """
        Create or dictionary for each follower to maintain unique persistent prediction targets
        """

        whitelist_pairs = self.config.get("exchange", {}).get("pair_whitelist")

        exists = self.follower_dict_path.is_file()

        if exists:
            logger.info("Found an existing follower dictionary")

        for pair in whitelist_pairs:
            self.follower_dict[pair] = {}

        self.save_follower_dict_to_disk()

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    def get_pair_dict_info(self, pair: str) -> Tuple[str, int, bool]:
        """
        Locate and load existing model metadata from persistent storage. If not located,
        create a new one and append the current pair to it and prepare it for its first
        training
        :param pair: str: pair to lookup
        :return:
            model_filename: str = unique filename used for loading persistent objects from disk
            trained_timestamp: int = the last time the coin was trained
            return_null_array: bool = Follower could not find pair metadata
        """

        pair_dict = self.pair_dict.get(pair)
        data_path_set = self.pair_dict.get(pair, self.empty_pair_dict).get("data_path", "")
        return_null_array = False

        if pair_dict:
            model_filename = pair_dict["model_filename"]
            trained_timestamp = pair_dict["trained_timestamp"]
        elif not self.follow_mode:
            self.pair_dict[pair] = self.empty_pair_dict.copy()
            model_filename = ""
            trained_timestamp = 0

        if not data_path_set and self.follow_mode:
            logger.warning(
                f"Follower could not find current pair {pair} in "
                f"pair_dictionary at path {self.full_path}, sending null values "
                "back to strategy."
            )
            trained_timestamp = 0
            model_filename = ''
            return_null_array = True

        return model_filename, trained_timestamp, return_null_array

    def set_pair_dict_info(self, metadata: dict) -> None:
        pair_in_dict = self.pair_dict.get(metadata["pair"])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata["pair"]] = self.empty_pair_dict.copy()

            return

    def set_initial_return_values(self, pair: str, pred_df: DataFrame) -> None:
        """
        Set the initial return values to the historical predictions dataframe. This avoids needing
        to repredict on historical candles, and also stores historical predictions despite
        retrainings (so stored predictions are true predictions, not just inferencing on trained
        data)
        """

        hist_df = self.historic_predictions
        len_diff = len(hist_df[pair].index) - len(pred_df.index)
        if len_diff < 0:
            df_concat = pd.concat([pred_df.iloc[:abs(len_diff)], hist_df[pair]],
                                  ignore_index=True, keys=hist_df[pair].keys())
        else:
            df_concat = hist_df[pair].tail(len(pred_df.index)).reset_index(drop=True)
        df_concat = df_concat.fillna(0)
        self.model_return_values[pair] = df_concat

    def append_model_predictions(self, pair: str, predictions: DataFrame,
                                 do_preds: NDArray[np.int_],
                                 dk: FreqaiDataKitchen, strat_df: DataFrame) -> None:
        """
        Append model predictions to historic predictions dataframe, then set the
        strategy return dataframe to the tail of the historic predictions. The length of
        the tail is equivalent to the length of the dataframe that entered FreqAI from
        the strategy originally. Doing this allows FreqUI to always display the correct
        historic predictions.
        """

        len_df = len(strat_df)
        index = self.historic_predictions[pair].index[-1:]
        columns = self.historic_predictions[pair].columns

        nan_df = pd.DataFrame(np.nan, index=index, columns=columns)
        self.historic_predictions[pair] = pd.concat(
            [self.historic_predictions[pair], nan_df], ignore_index=True, axis=0)
        df = self.historic_predictions[pair]

        # model outputs and associated statistics
        for label in predictions.columns:
            df[label].iloc[-1] = predictions[label].iloc[-1]
            if df[label].dtype == object:
                continue
            df[f"{label}_mean"].iloc[-1] = dk.data["labels_mean"][label]
            df[f"{label}_std"].iloc[-1] = dk.data["labels_std"][label]

        # outlier indicators
        df["do_predict"].iloc[-1] = do_preds[-1]
        if self.freqai_info["feature_parameters"].get("DI_threshold", 0) > 0:
            df["DI_values"].iloc[-1] = dk.DI_values[-1]

        # extra values the user added within custom prediction model
        if dk.data['extra_returns_per_train']:
            rets = dk.data['extra_returns_per_train']
            for return_str in rets:
                df[return_str].iloc[-1] = rets[return_str]

        # this logic carries users between version without needing to
        # change their identifier
        if 'close_price' not in df.columns:
            df['close_price'] = np.nan
            df['date_pred'] = np.nan

        df['close_price'].iloc[-1] = strat_df['close'].iloc[-1]
        df['date_pred'].iloc[-1] = strat_df['date'].iloc[-1]

        self.model_return_values[pair] = df.tail(len_df).reset_index(drop=True)

    def attach_return_values_to_return_dataframe(
            self, pair: str, dataframe: DataFrame) -> DataFrame:
        """
        Attach the return values to the strat dataframe
        :param dataframe: DataFrame = strategy dataframe
        :return: DataFrame = strat dataframe with return values attached
        """
        df = self.model_return_values[pair]
        to_keep = [col for col in dataframe.columns if not col.startswith("&")]
        dataframe = pd.concat([dataframe[to_keep], df], axis=1)
        return dataframe

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        """
        Build 0 filled dataframe to return to strategy
        """

        dk.find_features(dataframe)
        dk.find_labels(dataframe)

        full_labels = dk.label_list + dk.unique_class_list

        for label in full_labels:
            dataframe[label] = 0
            dataframe[f"{label}_mean"] = 0
            dataframe[f"{label}_std"] = 0

        dataframe["do_predict"] = 0

        if self.freqai_info["feature_parameters"].get("DI_threshold", 0) > 0:
            dataframe["DI_values"] = 0

        if dk.data['extra_returns_per_train']:
            rets = dk.data['extra_returns_per_train']
            for return_str in rets:
                dataframe[return_str] = 0

        dk.return_dataframe = dataframe

    def purge_old_models(self) -> None:

        model_folders = [x for x in self.full_path.iterdir() if x.is_dir()]

        pattern = re.compile(r"sub-train-(\w+)_(\d{10})")

        delete_dict: Dict[str, Any] = {}

        for dir in model_folders:
            result = pattern.match(str(dir.name))
            if result is None:
                continue
            coin = result.group(1)
            timestamp = result.group(2)

            if coin not in delete_dict:
                delete_dict[coin] = {}
                delete_dict[coin]["num_folders"] = 1
                delete_dict[coin]["timestamps"] = {int(timestamp): dir}
            else:
                delete_dict[coin]["num_folders"] += 1
                delete_dict[coin]["timestamps"][int(timestamp)] = dir

        for coin in delete_dict:
            if delete_dict[coin]["num_folders"] > 2:
                sorted_dict = collections.OrderedDict(
                    sorted(delete_dict[coin]["timestamps"].items())
                )
                num_delete = len(sorted_dict) - 2
                deleted = 0
                for k, v in sorted_dict.items():
                    if deleted >= num_delete:
                        break
                    logger.info(f"Freqai purging old model file {v}")
                    shutil.rmtree(v)
                    deleted += 1

    def update_follower_metadata(self):
        # follower needs to load from disk to get any changes made by leader to pair_dict
        self.load_drawer_from_disk()
        if self.config.get("freqai", {}).get("purge_old_models", False):
            self.purge_old_models()

    def save_metadata(self, dk: FreqaiDataKitchen) -> None:
        """
        Saves only metadata for backtesting studies if user prefers
        not to save model data. This saves tremendous amounts of space
        for users generating huge studies.
        This is only active when `save_backtest_models`: false (not default)
        """
        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(dk.data_path)

        dk.data["data_path"] = str(dk.data_path)
        dk.data["model_filename"] = str(dk.model_filename)
        dk.data["training_features_list"] = list(dk.data_dictionary["train_features"].columns)
        dk.data["label_list"] = dk.label_list

        with open(save_path / f"{dk.model_filename}_metadata.json", "w") as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

        return

    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen) -> None:
        """
        Saves all data associated with a model for a single sub-train time range
        :param model: User trained model which can be reused for inferencing to generate
                      predictions
        """

        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(dk.data_path)

        # Save the trained model
        if self.model_type == 'joblib':
            dump(model, save_path / f"{dk.model_filename}_model.joblib")
        elif self.model_type == 'keras':
            model.save(save_path / f"{dk.model_filename}_model.h5")
        elif 'stable_baselines' in self.model_type or 'sb3_contrib' == self.model_type:
            model.save(save_path / f"{dk.model_filename}_model.zip")

        if dk.svm_model is not None:
            dump(dk.svm_model, save_path / f"{dk.model_filename}_svm_model.joblib")

        dk.data["data_path"] = str(dk.data_path)
        dk.data["model_filename"] = str(dk.model_filename)
        dk.data["training_features_list"] = dk.training_features_list
        dk.data["label_list"] = dk.label_list
        # store the metadata
        with open(save_path / f"{dk.model_filename}_metadata.json", "w") as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

        # save the train data to file so we can check preds for area of applicability later
        dk.data_dictionary["train_features"].to_pickle(
            save_path / f"{dk.model_filename}_trained_df.pkl"
        )

        dk.data_dictionary["train_dates"].to_pickle(
            save_path / f"{dk.model_filename}_trained_dates_df.pkl"
        )

        if self.freqai_info["feature_parameters"].get("principal_component_analysis"):
            cloudpickle.dump(
                dk.pca, open(dk.data_path / f"{dk.model_filename}_pca_object.pkl", "wb")
            )

        self.model_dictionary[coin] = model
        self.pair_dict[coin]["model_filename"] = dk.model_filename
        self.pair_dict[coin]["data_path"] = str(dk.data_path)

        if coin not in self.meta_data_dictionary:
            self.meta_data_dictionary[coin] = {}
        self.meta_data_dictionary[coin]["train_df"] = dk.data_dictionary["train_features"]
        self.meta_data_dictionary[coin]["meta_data"] = dk.data
        self.save_drawer_to_disk()

        return

    def load_metadata(self, dk: FreqaiDataKitchen) -> None:
        """
        Load only metadata into datakitchen to increase performance during
        presaved backtesting (prediction file loading).
        """
        with open(dk.data_path / f"{dk.model_filename}_metadata.json", "r") as fp:
            dk.data = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
            dk.training_features_list = dk.data["training_features_list"]
            dk.label_list = dk.data["label_list"]

    def load_data(self, coin: str, dk: FreqaiDataKitchen) -> Any:
        """
        loads all data required to make a prediction on a sub-train time range
        :returns:
        :model: User trained model which can be inferenced for new predictions
        """

        if not self.pair_dict[coin]["model_filename"]:
            return None

        if dk.live:
            dk.model_filename = self.pair_dict[coin]["model_filename"]
            dk.data_path = Path(self.pair_dict[coin]["data_path"])

        if coin in self.meta_data_dictionary:
            dk.data = self.meta_data_dictionary[coin]["meta_data"]
            dk.data_dictionary["train_features"] = self.meta_data_dictionary[coin]["train_df"]
        else:
            with open(dk.data_path / f"{dk.model_filename}_metadata.json", "r") as fp:
                dk.data = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)

            dk.data_dictionary["train_features"] = pd.read_pickle(
                dk.data_path / f"{dk.model_filename}_trained_df.pkl"
            )

        dk.training_features_list = dk.data["training_features_list"]
        dk.label_list = dk.data["label_list"]

        # try to access model in memory instead of loading object from disk to save time
        if dk.live and coin in self.model_dictionary:
            model = self.model_dictionary[coin]
        elif self.model_type == 'joblib':
            model = load(dk.data_path / f"{dk.model_filename}_model.joblib")
        elif self.model_type == 'keras':
            from tensorflow import keras
            model = keras.models.load_model(dk.data_path / f"{dk.model_filename}_model.h5")
        elif 'stable_baselines' in self.model_type or 'sb3_contrib' == self.model_type:
            mod = importlib.import_module(
                self.model_type, self.freqai_info['rl_config']['model_type'])
            MODELCLASS = getattr(mod, self.freqai_info['rl_config']['model_type'])
            model = MODELCLASS.load(dk.data_path / f"{dk.model_filename}_model")

        if Path(dk.data_path / f"{dk.model_filename}_svm_model.joblib").is_file():
            dk.svm_model = load(dk.data_path / f"{dk.model_filename}_svm_model.joblib")

        if not model:
            raise OperationalException(
                f"Unable to load model, ensure model exists at " f"{dk.data_path} "
            )

        # load it into ram if it was loaded from disk
        if coin not in self.model_dictionary:
            self.model_dictionary[coin] = model

        if self.config["freqai"]["feature_parameters"]["principal_component_analysis"]:
            dk.pca = cloudpickle.load(
                open(dk.data_path / f"{dk.model_filename}_pca_object.pkl", "rb")
            )

        return model

    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen) -> None:
        """
        Append new candles to our stores historic data (in memory) so that
        we do not need to load candle history from disk and we dont need to
        pinging exchange multiple times for the same candle.
        :param dataframe: DataFrame = strategy provided dataframe
        """
        feat_params = self.freqai_info["feature_parameters"]
        with self.history_lock:
            history_data = self.historic_data

            for pair in dk.all_pairs:
                for tf in feat_params.get("include_timeframes"):

                    # check if newest candle is already appended
                    df_dp = strategy.dp.get_pair_dataframe(pair, tf)
                    if len(df_dp.index) == 0:
                        continue
                    if str(history_data[pair][tf].iloc[-1]["date"]) == str(
                        df_dp.iloc[-1:]["date"].iloc[-1]
                    ):
                        continue

                    try:
                        index = (
                            df_dp.loc[
                                df_dp["date"] == history_data[pair][tf].iloc[-1]["date"]
                            ].index[0]
                            + 1
                        )
                    except IndexError:
                        logger.warning(
                            f"Unable to update pair history for {pair}. "
                            "If this does not resolve itself after 1 additional candle, "
                            "please report the error to #freqai discord channel"
                        )
                        return

                    history_data[pair][tf] = pd.concat(
                        [
                            history_data[pair][tf],
                            df_dp.iloc[index:],
                        ],
                        ignore_index=True,
                        axis=0,
                    )

            self.current_candle = history_data[dk.pair][self.config['timeframe']].iloc[-1]['date']

    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen) -> None:
        """
        Load pair histories for all whitelist and corr_pairlist pairs.
        Only called once upon startup of bot.
        :param timerange: TimeRange = full timerange required to populate all indicators
                          for training according to user defined train_period_days
        """
        history_data = self.historic_data

        for pair in dk.all_pairs:
            if pair not in history_data:
                history_data[pair] = {}
            for tf in self.freqai_info["feature_parameters"].get("include_timeframes"):
                history_data[pair][tf] = load_pair_history(
                    datadir=self.config["datadir"],
                    timeframe=tf,
                    pair=pair,
                    timerange=timerange,
                    data_format=self.config.get("dataformat_ohlcv", "json"),
                    candle_type=self.config.get("trading_mode", "spot"),
                )

    def get_base_and_corr_dataframes(
        self, timerange: TimeRange, pair: str, dk: FreqaiDataKitchen
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """
        Searches through our historic_data in memory and returns the dataframes relevant
        to the present pair.
        :param timerange: TimeRange = full timerange required to populate all indicators
                          for training according to user defined train_period_days
        :param metadata: dict = strategy furnished pair metadata
        """
        with self.history_lock:
            corr_dataframes: Dict[Any, Any] = {}
            base_dataframes: Dict[Any, Any] = {}
            historic_data = self.historic_data
            pairs = self.freqai_info["feature_parameters"].get(
                "include_corr_pairlist", []
            )

            for tf in self.freqai_info["feature_parameters"].get("include_timeframes"):
                base_dataframes[tf] = dk.slice_dataframe(
                    timerange, historic_data[pair][tf]).reset_index(drop=True)
                if pairs:
                    for p in pairs:
                        if pair in p:
                            continue  # dont repeat anything from whitelist
                        if p not in corr_dataframes:
                            corr_dataframes[p] = {}
                        corr_dataframes[p][tf] = dk.slice_dataframe(
                            timerange, historic_data[p][tf]
                        ).reset_index(drop=True)

        return corr_dataframes, base_dataframes

    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
        """
        Returns timerange information based on historic predictions file
        :return: timerange calculated from saved live data
        """
        if not self.historic_predictions_path.is_file():
            raise OperationalException(
                'Historic predictions not found. Historic predictions data is required '
                'to run backtest with the freqai-backtest-live-models option '
            )

        self.load_historic_predictions_from_disk()

        all_pairs_end_dates = []
        for pair in self.historic_predictions:
            pair_historic_data = self.historic_predictions[pair]
            all_pairs_end_dates.append(pair_historic_data.date_pred.max())

        global_metadata = self.load_global_metadata_from_disk()
        start_date = datetime.fromtimestamp(int(global_metadata["start_dry_live_date"]))
        end_date = max(all_pairs_end_dates)
        # add 1 day to string timerange to ensure BT module will load all dataframe data
        end_date = end_date + timedelta(days=1)
        backtesting_timerange = TimeRange(
            'date', 'date', int(start_date.timestamp()), int(end_date.timestamp())
        )
        return backtesting_timerange
