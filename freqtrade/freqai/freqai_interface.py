import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


class IFreqaiModel(ABC):
    """
    Class containing all tools for training and prediction in the strategy.
    User models should inherit from this class as shown in
    templates/ExamplePredictionModel.py where the user overrides
    train(), predict(), fit(), and make_labels().
    Author: Robert Caulk, rob.caulk@gmail.com
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        self.config = config
        self.freqai_info = config["freqai"]
        self.data_split_parameters = config["freqai"]["data_split_parameters"]
        self.model_training_parameters = config["freqai"]["model_training_parameters"]
        self.feature_parameters = config["freqai"]["feature_parameters"]
        self.backtest_timerange = config["timerange"]

        self.time_last_trained = None
        self.current_time = None
        self.model = None
        self.predictions = None
        self.live_trained_timerange = None

    def start(self, dataframe: DataFrame, metadata: dict, dp: DataProvider) -> DataFrame:
        """
        Entry point to the FreqaiModel, it will train a new model if
        necesssary before making the prediction.
        The backtesting and training paradigm is a sliding training window
        with a following backtest window. Both windows slide according to the
        length of the backtest window. This function is not intended to be
        overridden by children of IFreqaiModel, but technically, it can be
        if the user wishes to make deeper changes to the sliding window
        logic.
        :params:
        :dataframe: Full dataframe coming from strategy - it contains entire
        backtesting timerange + additional historical data necessary to train
        the model.
        :metadata: pair metadataa coming from strategy.
        """
        self.pair = metadata["pair"]
        self.dh = FreqaiDataKitchen(self.config, dataframe)

        if dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            logger.info('testing live')

        logger.info("going to train %s timeranges", len(self.dh.training_timeranges))

        # Loop enforcing the sliding window training/backtesting paragigm
        # tr_train is the training time range e.g. 1 historical month
        # tr_backtest is the backtesting time range e.g. the week directly
        # following tr_train. Both of these windows slide through the
        # entire backtest
        for tr_train, tr_backtest in zip(
            self.dh.training_timeranges, self.dh.backtesting_timeranges
        ):
            gc.collect()
            # self.config['timerange'] = tr_train
            self.dh.data = {}  # clean the pair specific data between models
            self.training_timerange = tr_train
            dataframe_train = self.dh.slice_dataframe(tr_train, dataframe)
            dataframe_backtest = self.dh.slice_dataframe(tr_backtest, dataframe)
            logger.info("training %s for %s", self.pair, tr_train)
            self.dh.model_path = Path(self.dh.full_path / str("sub-train" + "-" + str(tr_train)))
            if not self.model_exists(self.pair, training_timerange=tr_train):
                self.model = self.train(dataframe_train, metadata)
                self.dh.save_data(self.model)
            else:
                self.model = self.dh.load_data()

            preds, do_preds = self.predict(dataframe_backtest)

            self.dh.append_predictions(preds, do_preds, len(dataframe_backtest))
            print('predictions', len(self.dh.full_predictions),
                  'do_predict', len(self.dh.full_do_predict))

        self.dh.fill_predictions(len(dataframe))

        return (self.dh.full_predictions, self.dh.full_do_predict,
                self.dh.full_target_mean, self.dh.full_target_std)

    def make_labels(self, dataframe: DataFrame) -> DataFrame:
        """
        User defines the labels here (target values).
        :params:
        :dataframe: the full dataframe for the present training period
        """

        return

    @abstractmethod
    def train(self, unfiltered_dataframe: DataFrame, metadata: dict) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datahandler
        for storing, saving, loading, and analyzing the data.
        :params:
        :unfiltered_dataframe: Full dataframe for the current training period
        :metadata: pair metadata from strategy.
        :returns:
        :model: Trained model which can be used to inference (self.predict)
        """

    @abstractmethod
    def fit(self) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        return

    @abstractmethod
    def predict(self, dataframe: DataFrame) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

    def model_exists(self, pair: str, training_timerange: str) -> bool:
        """
        Given a pair and path, check if a model already exists
        :param pair: pair e.g. BTC/USD
        :param path: path to model
        """
        coin, _ = pair.split("/")
        self.dh.model_filename = "cb_" + coin.lower() + "_" + training_timerange
        path_to_modelfile = Path(self.dh.model_path / str(self.dh.model_filename + "_model.joblib"))
        file_exists = path_to_modelfile.is_file()
        if file_exists:
            logger.info("Found model at %s", self.dh.model_path / self.dh.model_filename)
        else:
            logger.info("Could not find model at %s", self.dh.model_path / self.dh.model_filename)
        return file_exists
