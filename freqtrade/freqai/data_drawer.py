
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

# import pickle as pk
import numpy as np


logger = logging.getLogger(__name__)


class FreqaiDataDrawer:
    """
    Class aimed at holding all pair models/info in memory for better inferencing/retrainig/saving
    /loading to/from disk.
    This object remains persistent throughout live/dry, unlike FreqaiDataKitchen, which is
    reinstantiated for each coin.
    """
    def __init__(self, full_path: Path, pair_whitelist):

        # dictionary holding all pair metadata necessary to load in from disk
        self.pair_dict: Dict[str, Any] = {}
        # dictionary holding all actively inferenced models in memory given a model filename
        self.model_dictionary: Dict[str, Any] = {}
        self.model_return_values: Dict[str, Any] = {}
        self.pair_data_dict: Dict[str, Any] = {}
        self.full_path = full_path
        self.load_drawer_from_disk()
        self.training_queue: Dict[str, int] = {}
        # self.create_training_queue(pair_whitelist)

    def load_drawer_from_disk(self):
        exists = Path(self.full_path / str('pair_dictionary.json')).resolve().exists()
        if exists:
            with open(self.full_path / str('pair_dictionary.json'), "r") as fp:
                self.pair_dict = json.load(fp)
        else:
            logger.info("Could not find existing datadrawer, starting from scratch")
        return exists

    def save_drawer_to_disk(self):
        with open(self.full_path / str('pair_dictionary.json'), "w") as fp:
            json.dump(self.pair_dict, fp, default=self.np_encoder)

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    def get_pair_dict_info(self, metadata: dict) -> Tuple[str, int, bool]:
        pair_in_dict = self.pair_dict.get(metadata['pair'])
        if pair_in_dict:
            model_filename = self.pair_dict[metadata['pair']]['model_filename']
            trained_timestamp = self.pair_dict[metadata['pair']]['trained_timestamp']
            coin_first = self.pair_dict[metadata['pair']]['first']
        else:
            self.pair_dict[metadata['pair']] = {}
            model_filename = self.pair_dict[metadata['pair']]['model_filename'] = ''
            coin_first = self.pair_dict[metadata['pair']]['first'] = True
            trained_timestamp = self.pair_dict[metadata['pair']]['trained_timestamp'] = 0

        return model_filename, trained_timestamp, coin_first

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

    def set_initial_return_values(self, pair, dh):
        self.model_return_values[pair] = {}
        self.model_return_values[pair]['predictions'] = dh.full_predictions
        self.model_return_values[pair]['do_preds'] = dh.full_do_predict
        self.model_return_values[pair]['target_mean'] = dh.full_target_mean
        self.model_return_values[pair]['target_std'] = dh.full_target_std

    def append_model_predictions(self, pair, predictions, do_preds,
                                 target_mean, target_std, dh) -> None:

        pred_store = self.model_return_values[pair]['predictions']
        do_pred_store = self.model_return_values[pair]['do_preds']
        tm_store = self.model_return_values[pair]['target_mean']
        ts_store = self.model_return_values[pair]['target_std']
        pred_store = np.append(pred_store[1:], predictions[-1])
        do_pred_store = np.append(do_pred_store[1:], do_preds[-1])
        tm_store = np.append(tm_store[1:], target_mean)
        ts_store = np.append(ts_store[1:], target_std)

        dh.full_predictions = copy.deepcopy(pred_store)
        dh.full_do_predict = copy.deepcopy(do_pred_store)
        dh.full_target_mean = copy.deepcopy(tm_store)
        dh.full_target_std = copy.deepcopy(ts_store)
