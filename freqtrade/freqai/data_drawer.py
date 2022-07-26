import collections
import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import dump, load
from joblib.externals import cloudpickle
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_pair_history
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class FreqaiDataDrawer:
    """
    Class aimed at holding all pair models/info in memory for better inferencing/retrainig/saving
    /loading to/from disk.
    This object remains persistent throughout live/dry, unlike FreqaiDataKitchen, which is
    reinstantiated for each coin.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @thorntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta
    """

    def __init__(self, full_path: Path, config: dict, follow_mode: bool = False):

        self.config = config
        self.freqai_info = config.get("freqai", {})
        # dictionary holding all pair metadata necessary to load in from disk
        self.pair_dict: Dict[str, Any] = {}
        # dictionary holding all actively inferenced models in memory given a model filename
        self.model_dictionary: Dict[str, Any] = {}
        self.model_return_values: Dict[str, Any] = {}
        self.pair_data_dict: Dict[str, Any] = {}
        self.historic_data: Dict[str, Any] = {}
        self.historic_predictions: Dict[str, Any] = {}
        self.follower_dict: Dict[str, Any] = {}
        self.full_path = full_path
        self.follower_name = self.config.get("bot_name", "follower1")
        self.follower_dict_path = Path(
            self.full_path / f"follower_dictionary-{self.follower_name}.json"
        )
        self.historic_predictions_path = Path(self.full_path / "historic_predictions.pkl")
        self.pair_dictionary_path = Path(self.full_path / "pair_dictionary.json")
        self.follow_mode = follow_mode
        if follow_mode:
            self.create_follower_dict()
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()
        self.training_queue: Dict[str, int] = {}
        self.history_lock = threading.Lock()

    def load_drawer_from_disk(self):
        """
        Locate and load a previously saved data drawer full of all pair model metadata in
        present model folder.
        :returns:
        exists: bool = whether or not the drawer was located
        """
        exists = self.pair_dictionary_path.is_file()
        if exists:
            with open(self.pair_dictionary_path, "r") as fp:
                self.pair_dict = json.load(fp)
        elif not self.follow_mode:
            logger.info("Could not find existing datadrawer, starting from scratch")
        else:
            logger.warning(
                f"Follower could not find pair_dictionary at {self.full_path} "
                "sending null values back to strategy"
            )

        return exists

    def load_historic_predictions_from_disk(self):
        """
        Locate and load a previously saved historic predictions.
        :returns:
        exists: bool = whether or not the drawer was located
        """
        exists = self.historic_predictions_path.is_file()
        if exists:
            with open(self.historic_predictions_path, "rb") as fp:
                self.historic_predictions = cloudpickle.load(fp)
            logger.info(
                f"Found existing historic predictions at {self.full_path}, but beware "
                "that statistics may be inaccurate if the bot has been offline for "
                "an extended period of time."
            )
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
        Save data drawer full of all pair model metadata in present model folder.
        """
        with open(self.historic_predictions_path, "wb") as fp:
            cloudpickle.dump(self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL)

    def save_drawer_to_disk(self):
        """
        Save data drawer full of all pair model metadata in present model folder.
        """
        with open(self.pair_dictionary_path, "w") as fp:
            json.dump(self.pair_dict, fp, default=self.np_encoder)

    def save_follower_dict_to_disk(self):
        """
        Save follower dictionary to disk (used by strategy for persistent prediction targets)
        """
        with open(self.follower_dict_path, "w") as fp:
            json.dump(self.follower_dict, fp, default=self.np_encoder)

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

        with open(self.follower_dict_path, "w") as fp:
            json.dump(self.follower_dict, fp, default=self.np_encoder)

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    def get_pair_dict_info(self, pair: str) -> Tuple[str, int, bool, bool]:
        """
        Locate and load existing model metadata from persistent storage. If not located,
        create a new one and append the current pair to it and prepare it for its first
        training
        :param pair: str: pair to lookup
        :return:
            model_filename: str = unique filename used for loading persistent objects from disk
            trained_timestamp: int = the last time the coin was trained
            coin_first: bool = If the coin is fresh without metadata
            return_null_array: bool = Follower could not find pair metadata
        """
        pair_in_dict = self.pair_dict.get(pair)
        data_path_set = self.pair_dict.get(pair, {}).get("data_path", None)
        return_null_array = False

        if pair_in_dict:
            model_filename = self.pair_dict[pair]["model_filename"]
            trained_timestamp = self.pair_dict[pair]["trained_timestamp"]
            coin_first = self.pair_dict[pair]["first"]
        elif not self.follow_mode:
            self.pair_dict[pair] = {}
            model_filename = self.pair_dict[pair]["model_filename"] = ""
            coin_first = self.pair_dict[pair]["first"] = True
            trained_timestamp = self.pair_dict[pair]["trained_timestamp"] = 0
            self.pair_dict[pair]["priority"] = len(self.pair_dict)

        if not data_path_set and self.follow_mode:
            logger.warning(
                f"Follower could not find current pair {pair} in "
                f"pair_dictionary at path {self.full_path}, sending null values "
                "back to strategy."
            )
            return_null_array = True

        return model_filename, trained_timestamp, coin_first, return_null_array

    def set_pair_dict_info(self, metadata: dict) -> None:
        pair_in_dict = self.pair_dict.get(metadata["pair"])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata["pair"]] = {}
            self.pair_dict[metadata["pair"]]["model_filename"] = ""
            self.pair_dict[metadata["pair"]]["first"] = True
            self.pair_dict[metadata["pair"]]["trained_timestamp"] = 0
            self.pair_dict[metadata["pair"]]["priority"] = len(self.pair_dict)
            return

    def pair_to_end_of_training_queue(self, pair: str) -> None:
        # march all pairs up in the queue
        for p in self.pair_dict:
            self.pair_dict[p]["priority"] -= 1
        # send pair to end of queue
        self.pair_dict[pair]["priority"] = len(self.pair_dict)

    def set_initial_return_values(self, pair: str, dk: FreqaiDataKitchen,
                                  pred_df: DataFrame, do_preds: npt.ArrayLike) -> None:
        """
        Set the initial return values to a persistent dataframe. This avoids needing to repredict on
        historical candles, and also stores historical predictions despite retrainings (so stored
        predictions are true predictions, not just inferencing on trained data)
        """
        # dynamic df returned to strategy and plotted in frequi
        mrv_df = self.model_return_values[pair] = pd.DataFrame()

        for label in dk.label_list:
            mrv_df[label] = pred_df[label]
            mrv_df[f"{label}_mean"] = dk.data["labels_mean"][label]
            mrv_df[f"{label}_std"] = dk.data["labels_std"][label]

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0) > 0:
            mrv_df["DI_values"] = dk.DI_values

        mrv_df["do_predict"] = do_preds

    def append_model_predictions(self, pair: str, predictions, do_preds, dk, len_df) -> None:

        # strat seems to feed us variable sized dataframes - and since we are trying to build our
        # own return array in the same shape, we need to figure out how the size has changed
        # and adapt our stored/returned info accordingly.
        length_difference = len(self.model_return_values[pair]) - len_df
        i = 0

        if length_difference == 0:
            i = 1
        elif length_difference > 0:
            i = length_difference + 1

        df = self.model_return_values[pair] = self.model_return_values[pair].shift(-i)

        if pair in self.historic_predictions:
            hp_df = self.historic_predictions[pair]
            # here are some pandas hula hoops to accommodate the possibility of a series
            # or dataframe depending number of labels requested by user
            nan_df = pd.DataFrame(np.nan, index=hp_df.index[-2:] + 2, columns=hp_df.columns)
            hp_df = pd.concat([hp_df, nan_df], ignore_index=True, axis=0)
            self.historic_predictions[pair] = hp_df[:-1]

        for label in dk.label_list:
            df[label].iloc[-1] = predictions[label].iloc[-1]
            df[f"{label}_mean"].iloc[-1] = dk.data["labels_mean"][label]
            df[f"{label}_std"].iloc[-1] = dk.data["labels_std"][label]
        # df['prediction'].iloc[-1] = predictions[-1]
        df["do_predict"].iloc[-1] = do_preds[-1]

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0) > 0:
            df["DI_values"].iloc[-1] = dk.DI_values[-1]

        # append the new predictions to persistent storage
        if pair in self.historic_predictions:
            self.historic_predictions[pair].iloc[-1] = df[label].iloc[-1]

        if length_difference < 0:
            prepend_df = pd.DataFrame(
                np.zeros((abs(length_difference) - 1, len(df.columns))), columns=df.columns
            )
            df = pd.concat([prepend_df, df], axis=0)

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

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk) -> None:
        """
        Build 0 filled dataframe to return to strategy
        """

        dk.find_features(dataframe)

        for label in dk.label_list:
            dataframe[label] = 0
            dataframe[f"{label}_mean"] = 0
            dataframe[f"{label}_std"] = 0

        # dataframe['prediction'] = 0
        dataframe["do_predict"] = 0

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0) > 0:
            dataframe["DI_value"] = 0

        dk.return_dataframe = dataframe

    def purge_old_models(self) -> None:

        model_folders = [x for x in self.full_path.iterdir() if x.is_dir()]

        pattern = re.compile(r"sub-train-(\w+)_(\d{10})")

        delete_dict: Dict[str, Any] = {}

        for dir in model_folders:
            result = pattern.match(str(dir.name))
            if result is None:
                break
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

    # Functions pulled back from FreqaiDataKitchen because they relied on DataDrawer

    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen) -> None:
        """
        Saves all data associated with a model for a single sub-train time range
        :params:
        :model: User trained model which can be reused for inferencing to generate
        predictions
        """

        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(dk.data_path)

        # Save the trained model
        if not dk.keras:
            dump(model, save_path / f"{dk.model_filename}_model.joblib")
        else:
            model.save(save_path / f"{dk.model_filename}_model.h5")

        if dk.svm_model is not None:
            dump(dk.svm_model, save_path / str(dk.model_filename + "_svm_model.joblib"))

        dk.data["data_path"] = str(dk.data_path)
        dk.data["model_filename"] = str(dk.model_filename)
        dk.data["training_features_list"] = list(dk.data_dictionary["train_features"].columns)
        dk.data["label_list"] = dk.label_list
        # store the metadata
        with open(save_path / str(dk.model_filename + "_metadata.json"), "w") as fp:
            json.dump(dk.data, fp, default=dk.np_encoder)

        # save the train data to file so we can check preds for area of applicability later
        dk.data_dictionary["train_features"].to_pickle(
            save_path / str(dk.model_filename + "_trained_df.pkl")
        )

        if self.freqai_info.get("feature_parameters", {}).get("principal_component_analysis"):
            cloudpickle.dump(
                dk.pca, open(dk.data_path / str(dk.model_filename + "_pca_object.pkl"), "wb")
            )

        # if self.live:
        self.model_dictionary[dk.model_filename] = model
        self.pair_dict[coin]["model_filename"] = dk.model_filename
        self.pair_dict[coin]["data_path"] = str(dk.data_path)
        self.save_drawer_to_disk()

        return

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
            if self.freqai_info.get("follow_mode", False):
                # follower can be on a different system which is rsynced to the leader:
                dk.data_path = Path(
                    self.config["user_data_dir"]
                    / "models"
                    / dk.data_path.parts[-2]
                    / dk.data_path.parts[-1]
                )

        with open(dk.data_path / str(dk.model_filename + "_metadata.json"), "r") as fp:
            dk.data = json.load(fp)
            dk.training_features_list = dk.data["training_features_list"]
            dk.label_list = dk.data["label_list"]

        dk.data_dictionary["train_features"] = pd.read_pickle(
            dk.data_path / str(dk.model_filename + "_trained_df.pkl")
        )

        # try to access model in memory instead of loading object from disk to save time
        if dk.live and dk.model_filename in self.model_dictionary:
            model = self.model_dictionary[dk.model_filename]
        elif not dk.keras:
            model = load(dk.data_path / str(dk.model_filename + "_model.joblib"))
        else:
            from tensorflow import keras

            model = keras.models.load_model(dk.data_path / str(dk.model_filename + "_model.h5"))

        if Path(dk.data_path / str(dk.model_filename + "_svm_model.joblib")).resolve().exists():
            dk.svm_model = load(dk.data_path / str(dk.model_filename + "_svm_model.joblib"))

        if not model:
            raise OperationalException(
                f"Unable to load model, ensure model exists at " f"{dk.data_path} "
            )

        if self.config["freqai"]["feature_parameters"]["principal_component_analysis"]:
            dk.pca = cloudpickle.load(
                open(dk.data_path / str(dk.model_filename + "_pca_object.pkl"), "rb")
            )

        return model

    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen) -> None:
        """
        Append new candles to our stores historic data (in memory) so that
        we do not need to load candle history from disk and we dont need to
        pinging exchange multiple times for the same candle.
        :params:
        dataframe: DataFrame = strategy provided dataframe
        """
        feat_params = self.freqai_info.get("feature_parameters", {})
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
                            strategy.dp.get_pair_dataframe(pair, tf).iloc[index:],
                        ],
                        ignore_index=True,
                        axis=0,
                    )

    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen) -> None:
        """
        Load pair histories for all whitelist and corr_pairlist pairs.
        Only called once upon startup of bot.
        :params:
        timerange: TimeRange = full timerange required to populate all indicators
        for training according to user defined train_period_days
        """
        history_data = self.historic_data

        for pair in dk.all_pairs:
            if pair not in history_data:
                history_data[pair] = {}
            for tf in self.freqai_info.get("feature_parameters", {}).get("include_timeframes"):
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
        :params:
        timerange: TimeRange = full timerange required to populate all indicators
        for training according to user defined train_period_days
        metadata: dict = strategy furnished pair metadata
        """

        with self.history_lock:
            corr_dataframes: Dict[Any, Any] = {}
            base_dataframes: Dict[Any, Any] = {}
            historic_data = self.historic_data
            pairs = self.freqai_info.get("feature_parameters", {}).get(
                "include_corr_pairlist", []
            )

            for tf in self.freqai_info.get("feature_parameters", {}).get("include_timeframes"):
                base_dataframes[tf] = dk.slice_dataframe(timerange, historic_data[pair][tf])
                if pairs:
                    for p in pairs:
                        if pair in p:
                            continue  # dont repeat anything from whitelist
                        if p not in corr_dataframes:
                            corr_dataframes[p] = {}
                        corr_dataframes[p][tf] = dk.slice_dataframe(
                            timerange, historic_data[p][tf]
                        )

        return corr_dataframes, base_dataframes

    # to be used if we want to send predictions directly to the follower instead of forcing
    # follower to load models and inference
    # def save_model_return_values_to_disk(self) -> None:
    #     with open(self.full_path / str('model_return_values.json'), "w") as fp:
    #         json.dump(self.model_return_values, fp, default=self.np_encoder)

    # def load_model_return_values_from_disk(self, dk: FreqaiDataKitchen) -> FreqaiDataKitchen:
    #     exists = Path(self.full_path / str('model_return_values.json')).resolve().exists()
    #     if exists:
    #         with open(self.full_path / str('model_return_values.json'), "r") as fp:
    #             self.model_return_values = json.load(fp)
    #     elif not self.follow_mode:
    #         logger.info("Could not find existing datadrawer, starting from scratch")
    #     else:
    #         logger.warning(f'Follower could not find pair_dictionary at {self.full_path} '
    #                        'sending null values back to strategy')

    #     return exists, dk
