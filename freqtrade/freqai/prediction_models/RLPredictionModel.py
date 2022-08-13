import logging
from typing import Any, Dict, Tuple
#from matplotlib.colors import DivergingNorm

from pandas import DataFrame
import pandas as pd
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import tensorflow as tf
from freqtrade.freqai.prediction_models.BaseTensorFlowModel import BaseTensorFlowModel
from freqtrade.freqai.freqai_interface import IFreqaiModel
from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
import copy

from keras.layers import *
import random


logger = logging.getLogger(__name__)

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAX_EPOCHS = 10
LOOKBACK = 8


class RLPredictionModel_v2(IFreqaiModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), fit().
    """

    def fit(self, data_dictionary: Dict, pair) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        train_df = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        test_df = data_dictionary["test_features"]
        test_labels = data_dictionary["test_labels"]
        n_labels = len(train_labels.columns)
        if n_labels > 1:
            raise OperationalException(
                "Neural Net not yet configured for multi-targets. Please "
                " reduce number of targets to 1 in strategy."
            )

        n_features = len(data_dictionary["train_features"].columns)
        BATCH_SIZE = self.freqai_info.get("batch_size", 64)
        input_dims = [BATCH_SIZE, self.CONV_WIDTH, n_features]


        w1 = WindowGenerator(
            input_width=self.CONV_WIDTH,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=test_df,
            train_labels=train_labels,
            val_labels=test_labels,
            batch_size=BATCH_SIZE,
        )


        # train_agent()
        #pair = self.dd.historical_data[pair]
        #gym_env = FreqtradeEnv(data=train_df, prices=0.01, windows_size=100, pair=pair, stake_amount=100)

        # sep = '/'
        # coin = pair.split(sep, 1)[0]

        # # df1 = train_df.filter(regex='price')
        # # df2 = df1.filter(regex='raw')

        # # df3 = df2.filter(regex=f"{coin}")
        # # print(df3)

        # price = train_df[f"%-{coin}raw_price_5m"]
        # gym_env = RLPrediction_GymAnytrading(signal_features=train_df, prices=price, window_size=100)
        # sac = RLPrediction_Agent(gym_env)

        # print(sac)

        # return 0



        return model

    def predict(
        self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first=True
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_dataframe)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_dataframe, dk.training_features_list, training_filter=False
        )
        filtered_dataframe = dk.normalize_data_from_metadata(filtered_dataframe)
        dk.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk, filtered_dataframe)

        if first:
            full_df = dk.data_dictionary["prediction_features"]

            w1 = WindowGenerator(
                input_width=self.CONV_WIDTH,
                label_width=1,
                shift=1,
                test_df=full_df,
                batch_size=len(full_df),
            )

            predictions = self.model.predict(w1.inference)
            len_diff = len(dk.do_predict) - len(predictions)
            if len_diff > 0:
                dk.do_predict = dk.do_predict[len_diff:]

        else:
            data = dk.data_dictionary["prediction_features"]
            data = tf.expand_dims(data, axis=0)
            predictions = self.model(data, training=False)

        predictions = predictions[:, 0]
        pred_df = DataFrame(predictions, columns=dk.label_list)

        pred_df = dk.denormalize_labels_from_metadata(pred_df)

        return (pred_df, np.ones(len(pred_df)))

 
    def set_initial_historic_predictions(
        self, df: DataFrame, model: Any, dk: FreqaiDataKitchen, pair: str
    ) -> None:

        pass
        # w1 = WindowGenerator(
        #     input_width=self.CONV_WIDTH, label_width=1, shift=1, test_df=df, batch_size=len(df)
        # )
        
        # trained_predictions = model.predict(w1.inference)
        # #trained_predictions = trained_predictions[:, 0, 0]
        # trained_predictions = trained_predictions[:, 0]

        # n_lost_points = len(df) - len(trained_predictions)
        # pred_df = DataFrame(trained_predictions, columns=dk.label_list)
        # zeros_df = DataFrame(np.zeros((n_lost_points, len(dk.label_list))), columns=dk.label_list)
        # pred_df = pd.concat([zeros_df, pred_df], axis=0)

        # pred_df = dk.denormalize_labels_from_metadata(pred_df)

        

        # self.dd.historic_predictions[pair] = DataFrame()
        # self.dd.historic_predictions[pair] = copy.deepcopy(pred_df)


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