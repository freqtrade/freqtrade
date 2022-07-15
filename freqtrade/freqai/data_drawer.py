import collections
import json
import logging
import pickle
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame


# from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class FreqaiDataDrawer:
    """
    Class aimed at holding all pair models/info in memory for better inferencing/retrainig/saving
    /loading to/from disk.
    This object remains persistent throughout live/dry, unlike FreqaiDataKitchen, which is
    reinstantiated for each coin.
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
        exists = Path(self.full_path / str("pair_dictionary.json")).resolve().exists()
        if exists:
            with open(self.full_path / str("pair_dictionary.json"), "r") as fp:
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
        exists = Path(self.full_path / str("historic_predictions.pkl")).resolve().exists()
        if exists:
            with open(self.full_path / str("historic_predictions.pkl"), "rb") as fp:
                self.historic_predictions = pickle.load(fp)
            logger.info(f"Found existing historic predictions at {self.full_path}, but beware of "
                        "that statistics may be inaccurate if the bot has been offline for "
                        "an extended period of time.")
        elif not self.follow_mode:
            logger.info("Could not find existing historic_predictions, starting from scratch")
        else:
            logger.warning(
                f"Follower could not find historic predictions at {self.full_path} "
                "sending null values back to strategy"
            )

        return exists

    def save_drawer_to_disk(self):
        """
        Save data drawer full of all pair model metadata in present model folder.
        """
        with open(self.full_path / str("historic_predictions.pkl"), "wb") as fp:
            pickle.dump(self.historic_predictions, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save_historic_predictions_to_disk(self):
        """
        Save data drawer full of all pair model metadata in present model folder.
        """
        with open(self.full_path / str("pair_dictionary.json"), "w") as fp:
            json.dump(self.pair_dict, fp, default=self.np_encoder)

    def save_follower_dict_to_disk(self):
        """
        Save follower dictionary to disk (used by strategy for persistent prediction targets)
        """
        follower_name = self.config.get("bot_name", "follower1")
        with open(
            self.full_path / str("follower_dictionary-" + follower_name + ".json"), "w"
        ) as fp:
            json.dump(self.follower_dict, fp, default=self.np_encoder)

    def create_follower_dict(self):
        """
        Create or dictionary for each follower to maintain unique persistent prediction targets
        """
        follower_name = self.config.get("bot_name", "follower1")
        whitelist_pairs = self.config.get("exchange", {}).get("pair_whitelist")

        exists = (
            Path(self.full_path / str("follower_dictionary-" + follower_name + ".json"))
            .resolve()
            .exists()
        )

        if exists:
            logger.info("Found an existing follower dictionary")

        for pair in whitelist_pairs:
            self.follower_dict[pair] = {}

        with open(
            self.full_path / str("follower_dictionary-" + follower_name + ".json"), "w"
        ) as fp:
            json.dump(self.follower_dict, fp, default=self.np_encoder)

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    def get_pair_dict_info(self, pair: str) -> Tuple[str, int, bool, bool]:
        """
        Locate and load existing model metadata from persistent storage. If not located,
        create a new one and append the current pair to it and prepare it for its first
        training
        :params:
        metadata: dict = strategy furnished pair metadata
        :returns:
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

    def set_initial_return_values(self, pair: str, dk, pred_df, do_preds) -> None:
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
        hp_df = self.historic_predictions[pair]

        # here are some pandas hula hoops to accommodate the possibility of a series
        # or dataframe depending number of labels requested by user
        nan_df = pd.DataFrame(np.nan, index=hp_df.index[-2:] + 2, columns=hp_df.columns)
        hp_df = pd.concat([hp_df, nan_df], ignore_index=True, axis=0)
        hp_df = pd.concat([hp_df, nan_df[-2:-1]], axis=0)

        for label in dk.label_list:
            df[label].iloc[-1] = predictions[label].iloc[-1]
            df[f"{label}_mean"].iloc[-1] = dk.data["labels_mean"][label]
            df[f"{label}_std"].iloc[-1] = dk.data["labels_std"][label]
        # df['prediction'].iloc[-1] = predictions[-1]
        df["do_predict"].iloc[-1] = do_preds[-1]

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0) > 0:
            df["DI_values"].iloc[-1] = dk.DI_values[-1]

        # append the new predictions to persistent storage
        hp_df.iloc[-1] = df[label].iloc[-1]

        if length_difference < 0:
            prepend_df = pd.DataFrame(
                np.zeros((abs(length_difference) - 1, len(df.columns))), columns=df.columns
            )
            df = pd.concat([prepend_df, df], axis=0)

    def attach_return_values_to_return_dataframe(self, pair: str, dataframe) -> DataFrame:
        """
        Attach the return values to the strat dataframe
        :params:
        dataframe: DataFrame = strat dataframe
        :returns:
        dataframe: DataFrame = strat dataframe with return values attached
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

        pattern = re.compile(r"sub-train-(\w+)(\d{10})")

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
