
import collections
import copy
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

# import pickle as pk
import numpy as np
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
    def __init__(self, full_path: Path, pair_whitelist, follow_mode: bool = False):

        # dictionary holding all pair metadata necessary to load in from disk
        self.pair_dict: Dict[str, Any] = {}
        # dictionary holding all actively inferenced models in memory given a model filename
        self.model_dictionary: Dict[str, Any] = {}
        self.model_return_values: Dict[str, Any] = {}
        self.pair_data_dict: Dict[str, Any] = {}
        self.full_path = full_path
        self.follow_mode = follow_mode
        self.load_drawer_from_disk()
        self.training_queue: Dict[str, int] = {}
        # self.create_training_queue(pair_whitelist)

    def load_drawer_from_disk(self):
        exists = Path(self.full_path / str('pair_dictionary.json')).resolve().exists()
        if exists:
            with open(self.full_path / str('pair_dictionary.json'), "r") as fp:
                self.pair_dict = json.load(fp)
        elif not self.follow_mode:
            logger.info("Could not find existing datadrawer, starting from scratch")
        else:
            logger.warning(f'Follower could not find pair_dictionary at {self.full_path} '
                           'sending null values back to strategy')

        return exists

    def save_drawer_to_disk(self):
        with open(self.full_path / str('pair_dictionary.json'), "w") as fp:
            json.dump(self.pair_dict, fp, default=self.np_encoder)

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    def get_pair_dict_info(self, metadata: dict) -> Tuple[str, int, bool, bool]:
        pair_in_dict = self.pair_dict.get(metadata['pair'])
        data_path_set = self.pair_dict.get(metadata['pair'], {}).get('data_path', None)
        return_null_array = False

        if pair_in_dict:
            model_filename = self.pair_dict[metadata['pair']]['model_filename']
            trained_timestamp = self.pair_dict[metadata['pair']]['trained_timestamp']
            coin_first = self.pair_dict[metadata['pair']]['first']
        elif not self.follow_mode:
            self.pair_dict[metadata['pair']] = {}
            model_filename = self.pair_dict[metadata['pair']]['model_filename'] = ''
            coin_first = self.pair_dict[metadata['pair']]['first'] = True
            trained_timestamp = self.pair_dict[metadata['pair']]['trained_timestamp'] = 0

        if not data_path_set and self.follow_mode:
            logger.warning(f'Follower could not find current pair {metadata["pair"]} in '
                           f'pair_dictionary at path {self.full_path}, sending null values '
                           'back to strategy.')
            return_null_array = True

        return model_filename, trained_timestamp, coin_first, return_null_array

    def set_pair_dict_info(self, metadata: dict) -> None:
        pair_in_dict = self.pair_dict.get(metadata['pair'])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata['pair']] = {}
            self.pair_dict[metadata['pair']]['model_filename'] = ''
            self.pair_dict[metadata['pair']]['first'] = True
            self.pair_dict[metadata['pair']]['trained_timestamp'] = 0
            self.pair_dict[metadata['pair']]['priority'] = len(self.pair_dict)
            return

    # def create_training_queue(self, pairs: list) -> None:
    #     for i, pair in enumerate(pairs):
    #         self.training_queue[pair] = i + 1

    def pair_to_end_of_training_queue(self, pair: str) -> None:
        # march all pairs up in the queue
        for p in self.pair_dict:
            self.pair_dict[p]['priority'] -= 1
        # send pair to end of queue
        self.pair_dict[pair]['priority'] = len(self.pair_dict)

    def set_initial_return_values(self, pair: str, dh):

        self.model_return_values[pair] = {}
        self.model_return_values[pair]['predictions'] = dh.full_predictions
        self.model_return_values[pair]['do_preds'] = dh.full_do_predict
        self.model_return_values[pair]['target_mean'] = dh.full_target_mean
        self.model_return_values[pair]['target_std'] = dh.full_target_std

        # if not self.follow_mode:
        #     self.save_model_return_values_to_disk()

    def append_model_predictions(self, pair: str, predictions, do_preds,
                                 target_mean, target_std, dh, len_df) -> None:

        # strat seems to feed us variable sized dataframes - and since we are trying to build our
        # own return array in the same shape, we need to figure out how the size has changed
        # and adapt our stored/returned info accordingly.
        length_difference = len(self.model_return_values[pair]['predictions']) - len_df
        i = 0

        if length_difference == 0:
            i = 1
        elif length_difference > 0:
            i = length_difference + 1

        self.model_return_values[pair]['predictions'] = np.append(
            self.model_return_values[pair]['predictions'][i:], predictions[-1])
        self.model_return_values[pair]['do_preds'] = np.append(
            self.model_return_values[pair]['do_preds'][i:], do_preds[-1])
        self.model_return_values[pair]['target_mean'] = np.append(
            self.model_return_values[pair]['target_mean'][i:], target_mean)
        self.model_return_values[pair]['target_std'] = np.append(
            self.model_return_values[pair]['target_std'][i:], target_std)

        if length_difference < 0:
            prepend = np.zeros(abs(length_difference) - 1)
            self.model_return_values[pair]['predictions'] = np.insert(
                self.model_return_values[pair]['predictions'], 0, prepend)
            self.model_return_values[pair]['do_preds'] = np.insert(
                self.model_return_values[pair]['do_preds'], 0, prepend)
            self.model_return_values[pair]['target_mean'] = np.insert(
                self.model_return_values[pair]['target_mean'], 0, prepend)
            self.model_return_values[pair]['target_std'] = np.insert(
                self.model_return_values[pair]['target_std'], 0, prepend)

        dh.full_predictions = copy.deepcopy(self.model_return_values[pair]['predictions'])
        dh.full_do_predict = copy.deepcopy(self.model_return_values[pair]['do_preds'])
        dh.full_target_mean = copy.deepcopy(self.model_return_values[pair]['target_mean'])
        dh.full_target_std = copy.deepcopy(self.model_return_values[pair]['target_std'])

        # if not self.follow_mode:
        #     self.save_model_return_values_to_disk()

    def return_null_values_to_strategy(self, dataframe: DataFrame, dh) -> None:

        len_df = len(dataframe)
        dh.full_predictions = np.zeros(len_df)
        dh.full_do_predict = np.zeros(len_df)
        dh.full_target_mean = np.zeros(len_df)
        dh.full_target_std = np.zeros(len_df)

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
                delete_dict[coin]['num_folders'] = 1
                delete_dict[coin]['timestamps'] = {int(timestamp): dir}
            else:
                delete_dict[coin]['num_folders'] += 1
                delete_dict[coin]['timestamps'][int(timestamp)] = dir

        for coin in delete_dict:
            if delete_dict[coin]['num_folders'] > 2:
                sorted_dict = collections.OrderedDict(
                    sorted(delete_dict[coin]['timestamps'].items()))
                num_delete = len(sorted_dict) - 2
                deleted = 0
                for k, v in sorted_dict.items():
                    if deleted >= num_delete:
                        break
                    logger.info(f'Freqai purging old model file {v}')
                    shutil.rmtree(v)
                    deleted += 1

    # to be used if we want to send predictions directly to the follower instead of forcing
    # follower to load models and inference
    # def save_model_return_values_to_disk(self) -> None:
    #     with open(self.full_path / str('model_return_values.json'), "w") as fp:
    #         json.dump(self.model_return_values, fp, default=self.np_encoder)

    # def load_model_return_values_from_disk(self, dh: FreqaiDataKitchen) -> FreqaiDataKitchen:
    #     exists = Path(self.full_path / str('model_return_values.json')).resolve().exists()
    #     if exists:
    #         with open(self.full_path / str('model_return_values.json'), "r") as fp:
    #             self.model_return_values = json.load(fp)
    #     elif not self.follow_mode:
    #         logger.info("Could not find existing datadrawer, starting from scratch")
    #     else:
    #         logger.warning(f'Follower could not find pair_dictionary at {self.full_path} '
    #                        'sending null values back to strategy')

    #     return exists, dh
