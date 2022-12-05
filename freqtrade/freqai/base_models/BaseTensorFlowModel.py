import logging
from time import time
from typing import Any

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class BaseTensorFlowModel(IFreqaiModel):
    """
    Base class for TensorFlow type models.
    User *must* inherit from this class and set fit() and predict().
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs['config'])
        self.keras = True
        if self.ft_params.get("DI_threshold", 0):
            self.ft_params["DI_threshold"] = 0
            logger.warning("DI threshold is not configured for Keras models yet. Deactivating.")
        self.dd.model_type = 'keras'

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
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


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=None,
        val_df=None,
        test_df=None,
        train_labels=None,
        val_labels=None,
        test_labels=None,
        batch_size=None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    def make_dataset(self, data, labels=None):
        data = np.array(data, dtype=np.float32)
        if labels is not None:
            labels = np.array(labels, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=labels,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False,
            batch_size=self.batch_size,
        )

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, self.train_labels)

    @property
    def val(self):
        return self.make_dataset(self.val_df, self.val_labels)

    @property
    def test(self):
        return self.make_dataset(self.test_df, self.test_labels)

    @property
    def inference(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
