import logging
from abc import ABC, abstractmethod
from time import time
from typing import Any

import torch
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor


logger = logging.getLogger(__name__)


class BasePyTorchModel(IFreqaiModel, ABC):
    """
    Base class for PyTorch type models.
    User *must* inherit from this class and set fit() and predict() and
    data_convertor property.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        test_size = self.freqai_info.get('data_split_parameters', {}).get('test_size')
        self.splits = ["train", "test"] if test_size != 0 else ["train"]

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        # split data into train/test data.
        data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions", 0) or not self.live:
            dk.fit_labels()
        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(data_dictionary['train_features'])} data points")

        model = self.fit(data_dictionary, dk)
        end_time = time()

        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")

        return model

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        a class responsible for converting `*_features` & `*_labels` pandas dataframes
        to pytorch tensors.
        """
        raise NotImplementedError("Abstract property")
