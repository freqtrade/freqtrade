# import contextlib
import gc
import logging
# import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.enums import RunMode
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

# FIXME: suppress stdout for background training?
# class DummyFile(object):
#     def write(self, x): pass


# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = DummyFile()
#     yield
#     sys.stdout = save_stdout


def threaded(fn):
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper


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
        self.assert_config(self.config)
        self.freqai_info = config["freqai"]
        self.data_split_parameters = config["freqai"]["data_split_parameters"]
        self.model_training_parameters = config["freqai"]["model_training_parameters"]
        self.feature_parameters = config["freqai"]["feature_parameters"]
        # self.backtest_timerange = config["timerange"]

        self.time_last_trained = None
        self.current_time = None
        self.model = None
        self.predictions = None
        self.training_on_separate_thread = False
        self.retrain = False
        self.first = True
        # if self.freqai_info.get('live_trained_timerange'):
        #     self.new_trained_timerange = TimeRange.parse_timerange(
        #                                            self.freqai_info['live_trained_timerange'])
        # else:
        #     self.new_trained_timerange = TimeRange()

        self.set_full_path()
        self.data_drawer = FreqaiDataDrawer(Path(self.full_path))

    def assert_config(self, config: Dict[str, Any]) -> None:

        assert config.get('freqai'), "No Freqai parameters found in config file."
        assert config.get('freqai', {}).get('data_split_parameters'), ("No Freqai"
                                                                       "data_split_parameters"
                                                                       "in config file.")
        assert config.get('freqai', {}).get('model_training_parameters'), ("No Freqai"
                                                                           "modeltrainingparameters"
                                                                           "found in config file.")
        assert config.get('freqai', {}).get('feature_parameters'), ("No Freqai"
                                                                    "feature_parameters found in"
                                                                    "config file.")

    def start(self, dataframe: DataFrame, metadata: dict, strategy: IStrategy) -> DataFrame:
        """
        Entry point to the FreqaiModel from a specific pair, it will train a new model if
        necessary before making the prediction.
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
        :metadata: pair metadata coming from strategy.
        """

        self.live = strategy.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)

        # FreqaiDataKitchen is reinstantiated for each coin
        if self.live:
            self.data_drawer.set_pair_dict_info(metadata)
            if (not self.training_on_separate_thread and
                    self.data_drawer.pair_dict[metadata['pair']]['priority'] == 1):

                self.dh = FreqaiDataKitchen(self.config, self.data_drawer,
                                            self.live, metadata["pair"])
                dh = self.start_live(dataframe, metadata, strategy, self.dh)
            else:
                # we will have at max 2 separate instances of the kitchen at once.
                self.dh_fg = FreqaiDataKitchen(self.config, self.data_drawer,
                                               self.live, metadata["pair"])
                dh = self.start_live(dataframe, metadata, strategy, self.dh_fg)

            return (dh.full_predictions, dh.full_do_predict,
                    dh.full_target_mean, dh.full_target_std)

        # Backtesting only
        self.dh = FreqaiDataKitchen(self.config, self.data_drawer, self.live, metadata["pair"])

        logger.info(f'Training {len(self.dh.training_timeranges)} timeranges')

        # Loop enforcing the sliding window training/backtesting paradigm
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
            logger.info("training %s for %s", metadata["pair"], tr_train)
            trained_timestamp = TimeRange.parse_timerange(tr_train)
            self.dh.data_path = Path(self.dh.full_path /
                                     str("sub-train" + "-" + metadata['pair'].split("/")[0] +
                                         str(int(trained_timestamp.stopts))))
            if not self.model_exists(metadata["pair"], self.dh,
                                     trained_timestamp=trained_timestamp.stopts):
                self.model = self.train(dataframe_train, metadata, self.dh)
                self.dh.save_data(self.model)
            else:
                self.model = self.dh.load_data()
                # strategy_provided_features = self.dh.find_features(dataframe_train)
                # # TOFIX doesnt work with PCA
                # if strategy_provided_features != self.dh.training_features_list:
                #     logger.info("User changed input features, retraining model.")
                #     self.model = self.train(dataframe_train, metadata)
                #     self.dh.save_data(self.model)

            preds, do_preds = self.predict(dataframe_backtest, self.dh)

            self.dh.append_predictions(preds, do_preds, len(dataframe_backtest))
            print('predictions', len(self.dh.full_predictions),
                  'do_predict', len(self.dh.full_do_predict))

        self.dh.fill_predictions(len(dataframe))

        return (self.dh.full_predictions, self.dh.full_do_predict,
                self.dh.full_target_mean, self.dh.full_target_std)

    def start_live(self, dataframe: DataFrame, metadata: dict,
                   strategy: IStrategy, dh: FreqaiDataKitchen) -> FreqaiDataKitchen:
        """
        The main broad execution for dry/live. This function will check if a retraining should be
        performed, and if so, retrain and reset the model.

        """

        (model_filename,
         trained_timestamp,
         coin_first) = self.data_drawer.get_pair_dict_info(metadata)

        if not self.training_on_separate_thread:
            file_exists = False

            if trained_timestamp != 0:
                dh.set_paths(metadata, trained_timestamp)
                # data_drawer thinks the file eixts, verify here
                file_exists = self.model_exists(metadata['pair'],
                                                dh,
                                                trained_timestamp=trained_timestamp,
                                                model_filename=model_filename)

            # if not self.training_on_separate_thread:
            # this will also prevent other pairs from trying to train simultaneously.
            (self.retrain,
             new_trained_timerange) = dh.check_if_new_training_required(trained_timestamp)
            dh.set_paths(metadata, new_trained_timerange.stopts)
            # if self.training_on_separate_thread:
            #     logger.info("FreqAI training a new model on background thread.")
            #     self.retrain = False

            if self.retrain or not file_exists:
                if coin_first:
                    self.train_model_in_series(new_trained_timerange, metadata, strategy, dh)
                else:
                    self.training_on_separate_thread = True  # acts like a lock
                    self.retrain_model_on_separate_thread(new_trained_timerange,
                                                          metadata, strategy, dh)

        else:
            logger.info("FreqAI training a new model on background thread.")
            self.data_drawer.pair_dict[metadata['pair']]['priority'] = 1

        self.model = dh.load_data(coin=metadata['pair'])

        # strategy_provided_features = dh.find_features(dataframe)
        # if strategy_provided_features != dh.training_features_list:
        #     self.train_model_in_series(new_trained_timerange, metadata, strategy)

        preds, do_preds = self.predict(dataframe, dh)
        dh.append_predictions(preds, do_preds, len(dataframe))

        return dh

    def data_cleaning_train(self, dh: FreqaiDataKitchen) -> None:
        """
        Base data cleaning method for train
        Any function inside this method should drop training data points from the filtered_dataframe
        based on user decided logic. See FreqaiDataKitchen::remove_outliers() for an example
        of how outlier data points are dropped from the dataframe used for training.
        """
        if self.freqai_info.get('feature_parameters', {}).get('principal_component_analysis'):
            dh.principal_component_analysis()

        # if self.feature_parameters["determine_statistical_distributions"]:
        #     dh.determine_statistical_distributions()
        # if self.feature_parameters["remove_outliers"]:
        #     dh.remove_outliers(predict=False)

        if self.freqai_info.get('feature_parameters', {}).get('use_SVM_to_remove_outliers'):
            dh.use_SVM_to_remove_outliers(predict=False)

        if self.freqai_info.get('feature_parameters', {}).get('DI_threshold'):
            dh.data["avg_mean_dist"] = dh.compute_distances()

    def data_cleaning_predict(self, dh: FreqaiDataKitchen) -> None:
        """
        Base data cleaning method for predict.
        These functions each modify dh.do_predict, which is a dataframe with equal length
        to the number of candles coming from and returning to the strategy. Inside do_predict,
         1 allows prediction and < 0 signals to the strategy that the model is not confident in
         the prediction.
         See FreqaiDataKitchen::remove_outliers() for an example
        of how the do_predict vector is modified. do_predict is ultimately passed back to strategy
        for buy signals.
        """
        if self.freqai_info.get('feature_parameters', {}).get('principal_component_analysis'):
            dh.pca_transform()

        # if self.feature_parameters["determine_statistical_distributions"]:
        #     dh.determine_statistical_distributions()
        # if self.feature_parameters["remove_outliers"]:
        #     dh.remove_outliers(predict=True)  # creates dropped index

        if self.freqai_info.get('feature_parameters', {}).get('use_SVM_to_remove_outliers'):
            dh.use_SVM_to_remove_outliers(predict=True)

        if self.freqai_info.get('feature_parameters', {}).get('DI_threshold'):
            dh.check_if_pred_in_training_spaces()  # sets do_predict

    def model_exists(self, pair: str, dh: FreqaiDataKitchen, trained_timestamp: int = None,
                     model_filename: str = '') -> bool:
        """
        Given a pair and path, check if a model already exists
        :param pair: pair e.g. BTC/USD
        :param path: path to model
        """
        coin, _ = pair.split("/")

        # if self.live and trained_timestamp == 0:
        #     dh.model_filename = model_filename
        if not self.live:
            dh.model_filename = model_filename = "cb_" + coin.lower() + "_" + str(trained_timestamp)

        path_to_modelfile = Path(dh.data_path / str(model_filename + "_model.joblib"))
        file_exists = path_to_modelfile.is_file()
        if file_exists:
            logger.info("Found model at %s", dh.data_path / dh.model_filename)
        else:
            logger.info("Could not find model at %s", dh.data_path / dh.model_filename)
        return file_exists

    def set_full_path(self) -> None:
        self.full_path = Path(self.config['user_data_dir'] /
                              "models" /
                              str(self.freqai_info.get('live_full_backtestrange') +
                                  self.freqai_info.get('identifier')))

    @threaded
    def retrain_model_on_separate_thread(self, new_trained_timerange: TimeRange, metadata: dict,
                                         strategy: IStrategy, dh: FreqaiDataKitchen):

        # with nostdout():
        dh.download_new_data_for_retraining(new_trained_timerange, metadata)
        corr_dataframes, base_dataframes = dh.load_pairs_histories(new_trained_timerange,
                                                                   metadata)

        unfiltered_dataframe = dh.use_strategy_to_populate_indicators(strategy,
                                                                      corr_dataframes,
                                                                      base_dataframes,
                                                                      metadata)

        self.model = self.train(unfiltered_dataframe, metadata, dh)

        self.data_drawer.pair_dict[metadata['pair']][
                                   'trained_timestamp'] = new_trained_timerange.stopts

        dh.set_new_model_names(metadata, new_trained_timerange)

        # set this coin to lower priority to allow other coins in white list to get trained
        self.data_drawer.pair_dict[metadata['pair']]['priority'] = 0

        dh.save_data(self.model, coin=metadata['pair'])

        self.training_on_separate_thread = False
        self.retrain = False

    def train_model_in_series(self, new_trained_timerange: TimeRange, metadata: dict,
                              strategy: IStrategy, dh: FreqaiDataKitchen):

        dh.download_new_data_for_retraining(new_trained_timerange, metadata)
        corr_dataframes, base_dataframes = dh.load_pairs_histories(new_trained_timerange,
                                                                   metadata)

        unfiltered_dataframe = dh.use_strategy_to_populate_indicators(strategy,
                                                                      corr_dataframes,
                                                                      base_dataframes,
                                                                      metadata)

        self.model = self.train(unfiltered_dataframe, metadata, dh)

        self.data_drawer.pair_dict[metadata['pair']][
                                   'trained_timestamp'] = new_trained_timerange.stopts

        dh.set_new_model_names(metadata, new_trained_timerange)

        self.data_drawer.pair_dict[metadata['pair']]['first'] = False

        dh.save_data(self.model, coin=metadata['pair'])
        self.retrain = False

    # Methods which are overridden by user made prediction models.
    # See freqai/prediction_models/CatboostPredictionModlel.py for an example.

    @abstractmethod
    def train(self, unfiltered_dataframe: DataFrame, metadata: dict, dh: FreqaiDataKitchen) -> Any:
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
    def predict(self, dataframe: DataFrame,
                dh: FreqaiDataKitchen) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

    @abstractmethod
    def make_labels(self, dataframe: DataFrame, dh: FreqaiDataKitchen) -> DataFrame:
        """
        User defines the labels here (target values).
        :params:
        :dataframe: the full dataframe for the present training period
        """

        return
