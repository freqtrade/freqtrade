# import contextlib
import datetime
import gc
import logging
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


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
        self.data_split_parameters = config.get("freqai", {}).get("data_split_parameters")
        self.model_training_parameters = config.get("freqai", {}).get("model_training_parameters")
        self.feature_parameters = config.get("freqai", {}).get("feature_parameters")
        self.time_last_trained = None
        self.current_time = None
        self.model = None
        self.predictions = None
        self.training_on_separate_thread = False
        self.retrain = False
        self.first = True
        self.update_historic_data = 0
        self.set_full_path()
        self.follow_mode = self.freqai_info.get("follow_mode", False)
        self.dd = FreqaiDataDrawer(Path(self.full_path), self.config, self.follow_mode)
        self.lock = threading.Lock()
        self.follow_mode = self.freqai_info.get("follow_mode", False)
        self.identifier = self.freqai_info.get("identifier", "no_id_provided")
        self.scanning = False
        self.ready_to_scan = False
        self.first = True
        self.keras = self.freqai_info.get("keras", False)
        self.CONV_WIDTH = self.freqai_info.get("conv_width", 2)

    def assert_config(self, config: Dict[str, Any]) -> None:

        if not config.get("freqai", {}):
            raise OperationalException("No freqai parameters found in configuration file.")

    def start(self, dataframe: DataFrame, metadata: dict, strategy: IStrategy) -> DataFrame:
        """
        Entry point to the FreqaiModel from a specific pair, it will train a new model if
        necessary before making the prediction.

        :params:
        :dataframe: Full dataframe coming from strategy - it contains entire
        backtesting timerange + additional historical data necessary to train
        the model.
        :metadata: pair metadata coming from strategy.
        """

        self.live = strategy.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)
        self.dd.set_pair_dict_info(metadata)

        if self.live:
            self.dk = FreqaiDataKitchen(self.config, self.dd, self.live, metadata["pair"])
            dk = self.start_live(dataframe, metadata, strategy, self.dk)

        # For backtesting, each pair enters and then gets trained for each window along the
        # sliding window defined by "train_period" (training window) and "backtest_period"
        # (backtest window, i.e. window immediately following the training window).
        # FreqAI slides the window and sequentially builds the backtesting results before returning
        # the concatenated results for the full backtesting period back to the strategy.
        elif not self.follow_mode:
            self.dk = FreqaiDataKitchen(self.config, self.dd, self.live, metadata["pair"])
            logger.info(f"Training {len(self.dk.training_timeranges)} timeranges")
            dk = self.start_backtesting(dataframe, metadata, self.dk)

        dataframe = self.remove_features_from_df(dk.return_dataframe)
        return self.return_values(dataframe, dk)

    @threaded
    def start_scanning(self, strategy: IStrategy) -> None:
        """
        Function designed to constantly scan pairs for retraining on a separate thread (intracandle)
        to improve model youth. This function is agnostic to data preparation/collection/storage,
        it simply trains on what ever data is available in the self.dd.
        :params:
        strategy: IStrategy = The user defined strategy class
        """
        while 1:
            time.sleep(1)
            for pair in self.config.get("exchange", {}).get("pair_whitelist"):

                (_, trained_timestamp, _, _) = self.dd.get_pair_dict_info(pair)

                if self.dd.pair_dict[pair]["priority"] != 1:
                    continue
                dk = FreqaiDataKitchen(self.config, self.dd, self.live, pair)
                dk.set_paths(pair, trained_timestamp)
                (
                    retrain,
                    new_trained_timerange,
                    data_load_timerange,
                ) = dk.check_if_new_training_required(trained_timestamp)
                dk.set_paths(pair, new_trained_timerange.stopts)

                if retrain:
                    self.train_model_in_series(
                        new_trained_timerange, pair, strategy, dk, data_load_timerange
                    )

    def start_backtesting(
        self, dataframe: DataFrame, metadata: dict, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        The main broad execution for backtesting. For backtesting, each pair enters and then gets
        trained for each window along the sliding window defined by "train_period" (training window)
        and "backtest_period" (backtest window, i.e. window immediately following the
        training window). FreqAI slides the window and sequentially builds the backtesting results
        before returning the concatenated results for the full backtesting period back to the
        strategy.
        :params:
        dataframe: DataFrame = strategy passed dataframe
        metadata: Dict = pair metadata
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        :returns:
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        """

        # Loop enforcing the sliding window training/backtesting paradigm
        # tr_train is the training time range e.g. 1 historical month
        # tr_backtest is the backtesting time range e.g. the week directly
        # following tr_train. Both of these windows slide through the
        # entire backtest
        for tr_train, tr_backtest in zip(dk.training_timeranges, dk.backtesting_timeranges):
            (_, _, _, _) = self.dd.get_pair_dict_info(metadata["pair"])
            gc.collect()
            dk.data = {}  # clean the pair specific data between training window sliding
            self.training_timerange = tr_train
            # self.training_timerange_timerange = tr_train
            dataframe_train = dk.slice_dataframe(tr_train, dataframe)
            dataframe_backtest = dk.slice_dataframe(tr_backtest, dataframe)

            trained_timestamp = tr_train  # TimeRange.parse_timerange(tr_train)
            tr_train_startts_str = datetime.datetime.utcfromtimestamp(tr_train.startts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            tr_train_stopts_str = datetime.datetime.utcfromtimestamp(tr_train.stopts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            logger.info("Training %s", metadata["pair"])
            logger.info(f"Training {tr_train_startts_str} to {tr_train_stopts_str}")

            dk.data_path = Path(
                dk.full_path
                / str(
                    "sub-train"
                    + "-"
                    + metadata["pair"].split("/")[0]
                    + str(int(trained_timestamp.stopts))
                )
            )
            if not self.model_exists(
                metadata["pair"], dk, trained_timestamp=trained_timestamp.stopts
            ):
                dk.find_features(dataframe_train)
                self.model = self.train(dataframe_train, metadata["pair"], dk)
                self.dd.pair_dict[metadata["pair"]]["trained_timestamp"] = trained_timestamp.stopts
                dk.set_new_model_names(metadata["pair"], trained_timestamp)
                dk.save_data(self.model, metadata["pair"], keras_model=self.keras)
            else:
                self.model = dk.load_data(metadata["pair"], keras_model=self.keras)

            self.check_if_feature_list_matches_strategy(dataframe_train, dk)

            pred_df, do_preds = self.predict(dataframe_backtest, dk)

            dk.append_predictions(pred_df, do_preds, len(dataframe_backtest))

        dk.fill_predictions(dataframe)

        return dk

    def start_live(
        self, dataframe: DataFrame, metadata: dict, strategy: IStrategy, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        The main broad execution for dry/live. This function will check if a retraining should be
        performed, and if so, retrain and reset the model.
        :params:
        dataframe: DataFrame = strategy passed dataframe
        metadata: Dict = pair metadata
        strategy: IStrategy = currently employed strategy
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        :returns:
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        """

        # update follower
        if self.follow_mode:
            self.dd.update_follower_metadata()

        # get the model metadata associated with the current pair
        (_, trained_timestamp, _, return_null_array) = self.dd.get_pair_dict_info(metadata["pair"])

        # if the metadata doesnt exist, the follower returns null arrays to strategy
        if self.follow_mode and return_null_array:
            logger.info("Returning null array from follower to strategy")
            self.dd.return_null_values_to_strategy(dataframe, dk)
            return dk

        # append the historic data once per round
        if self.dd.historic_data:
            dk.update_historic_data(strategy)
            logger.debug(f'Updating historic data on pair {metadata["pair"]}')

        # if trainable, check if model needs training, if so compute new timerange,
        # then save model and metadata.
        # if not trainable, load existing data
        if not self.follow_mode:

            (_, new_trained_timerange, data_load_timerange) = dk.check_if_new_training_required(
                trained_timestamp
            )
            dk.set_paths(metadata["pair"], new_trained_timerange.stopts)

            # download candle history if it is not already in memory
            if not self.dd.historic_data:
                logger.info(
                    "Downloading all training data for all pairs in whitelist and "
                    "corr_pairlist, this may take a while if you do not have the "
                    "data saved"
                )
                dk.download_all_data_for_training(data_load_timerange)
                dk.load_all_pair_histories(data_load_timerange)

            if not self.scanning:
                self.scanning = True
                self.start_scanning(strategy)

        elif self.follow_mode:
            dk.set_paths(metadata["pair"], trained_timestamp)
            logger.info(
                "FreqAI instance set to follow_mode, finding existing pair"
                f"using { self.identifier }"
            )

        # load the model and associated data into the data kitchen
        self.model = dk.load_data(coin=metadata["pair"], keras_model=self.keras)

        if not self.model:
            logger.warning(
                f"No model ready for {metadata['pair']}, returning null values to strategy."
            )
            self.dd.return_null_values_to_strategy(dataframe, dk)
            return dk

        # ensure user is feeding the correct indicators to the model
        self.check_if_feature_list_matches_strategy(dataframe, dk)

        self.build_strategy_return_arrays(dataframe, dk, metadata["pair"], trained_timestamp)

        return dk

    def build_strategy_return_arrays(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, pair: str, trained_timestamp: int
    ) -> None:

        # hold the historical predictions in memory so we are sending back
        # correct array to strategy

        if pair not in self.dd.model_return_values:
            pred_df, do_preds = self.predict(dataframe, dk)
            self.dd.set_initial_return_values(pair, dk, pred_df, do_preds)
            dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)
            return
        elif self.dk.check_if_model_expired(trained_timestamp):
            pred_df = DataFrame(np.zeros((2, len(dk.label_list))), columns=dk.label_list)
            do_preds, dk.DI_values = np.ones(2) * 2, np.zeros(2)
            logger.warning(
                f"Model expired for {pair}, returning null values to strategy. Strategy "
                "construction should take care to consider this event with "
                "prediction == 0 and do_predict == 2"
            )
        else:
            # Only feed in the most recent candle for prediction in live scenario
            pred_df, do_preds = self.predict(dataframe.iloc[-self.CONV_WIDTH:], dk, first=False)

        self.dd.append_model_predictions(pair, pred_df, do_preds, dk, len(dataframe))
        dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)

        return

    def check_if_feature_list_matches_strategy(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen
    ) -> None:
        """
        Ensure user is passing the proper feature set if they are reusing an `identifier` pointing
        to a folder holding existing models.
        :params:
        dataframe: DataFrame = strategy provided dataframe
        dk: FreqaiDataKitchen = non-persistent data container/analyzer for current coin/bot loop
        """
        dk.find_features(dataframe)
        if "training_features_list_raw" in dk.data:
            feature_list = dk.data["training_features_list_raw"]
        else:
            feature_list = dk.training_features_list
        if dk.training_features_list != feature_list:
            raise OperationalException(
                "Trying to access pretrained model with `identifier` "
                "but found different features furnished by current strategy."
                "Change `identifer` to train from scratch, or ensure the"
                "strategy is furnishing the same features as the pretrained"
                "model"
            )

    def data_cleaning_train(self, dk: FreqaiDataKitchen) -> None:
        """
        Base data cleaning method for train
        Any function inside this method should drop training data points from the filtered_dataframe
        based on user decided logic. See FreqaiDataKitchen::remove_outliers() for an example
        of how outlier data points are dropped from the dataframe used for training.
        """

        if self.freqai_info.get("feature_parameters", {}).get("principal_component_analysis"):
            dk.principal_component_analysis()

        if self.freqai_info.get("feature_parameters", {}).get("use_SVM_to_remove_outliers"):
            dk.use_SVM_to_remove_outliers(predict=False)

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold"):
            dk.data["avg_mean_dist"] = dk.compute_distances()

        # if self.feature_parameters["determine_statistical_distributions"]:
        #     dk.determine_statistical_distributions()
        # if self.feature_parameters["remove_outliers"]:
        #     dk.remove_outliers(predict=False)

    def data_cleaning_predict(self, dk: FreqaiDataKitchen, dataframe: DataFrame) -> None:
        """
        Base data cleaning method for predict.
        These functions each modify dk.do_predict, which is a dataframe with equal length
        to the number of candles coming from and returning to the strategy. Inside do_predict,
         1 allows prediction and < 0 signals to the strategy that the model is not confident in
         the prediction.
         See FreqaiDataKitchen::remove_outliers() for an example
        of how the do_predict vector is modified. do_predict is ultimately passed back to strategy
        for buy signals.
        """
        if self.freqai_info.get("feature_parameters", {}).get("principal_component_analysis"):
            dk.pca_transform(dataframe)

        if self.freqai_info.get("feature_parameters", {}).get("use_SVM_to_remove_outliers"):
            dk.use_SVM_to_remove_outliers(predict=True)

        if self.freqai_info.get("feature_parameters", {}).get("DI_threshold"):
            dk.check_if_pred_in_training_spaces()

        # if self.feature_parameters["determine_statistical_distributions"]:
        #     dk.determine_statistical_distributions()
        # if self.feature_parameters["remove_outliers"]:
        #     dk.remove_outliers(predict=True)  # creates dropped index

    def model_exists(
        self,
        pair: str,
        dk: FreqaiDataKitchen,
        trained_timestamp: int = None,
        model_filename: str = "",
        scanning: bool = False,
    ) -> bool:
        """
        Given a pair and path, check if a model already exists
        :param pair: pair e.g. BTC/USD
        :param path: path to model
        """
        coin, _ = pair.split("/")

        if not self.live:
            dk.model_filename = model_filename = "cb_" + coin.lower() + "_" + str(trained_timestamp)

        path_to_modelfile = Path(dk.data_path / str(model_filename + "_model.joblib"))
        file_exists = path_to_modelfile.is_file()
        if file_exists and not scanning:
            logger.info("Found model at %s", dk.data_path / dk.model_filename)
        elif not scanning:
            logger.info("Could not find model at %s", dk.data_path / dk.model_filename)
        return file_exists

    def set_full_path(self) -> None:
        self.full_path = Path(
            self.config["user_data_dir"] / "models" / str(self.freqai_info.get("identifier"))
        )
        self.full_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            self.config["config_files"][0],
            Path(self.full_path, Path(self.config["config_files"][0]).name),
        )

    def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove the features from the dataframe before returning it to strategy. This keeps it
        compact for Frequi purposes.
        """
        to_keep = [
            col for col in dataframe.columns if not col.startswith("%") or col.startswith("%%")
        ]
        return dataframe[to_keep]

    def train_model_in_series(
        self,
        new_trained_timerange: TimeRange,
        pair: str,
        strategy: IStrategy,
        dk: FreqaiDataKitchen,
        data_load_timerange: TimeRange,
    ):
        """
        Retreive data and train model in single threaded mode (only used if model directory is empty
        upon startup for dry/live )
        :params:
        new_trained_timerange: TimeRange = the timerange to train the model on
        metadata: dict = strategy provided metadata
        strategy: IStrategy = user defined strategy object
        dk: FreqaiDataKitchen = non-persistent data container for current coin/loop
        data_load_timerange: TimeRange = the amount of data to be loaded for populate_any_indicators
        (larger than new_trained_timerange so that new_trained_timerange does not contain any NaNs)
        """

        corr_dataframes, base_dataframes = dk.get_base_and_corr_dataframes(
            data_load_timerange, pair
        )

        unfiltered_dataframe = dk.use_strategy_to_populate_indicators(
            strategy, corr_dataframes, base_dataframes, pair
        )

        unfiltered_dataframe = dk.slice_dataframe(new_trained_timerange, unfiltered_dataframe)

        # find the features indicated by strategy and store in datakitchen
        dk.find_features(unfiltered_dataframe)

        model = self.train(unfiltered_dataframe, pair, dk)

        self.dd.pair_dict[pair]["trained_timestamp"] = new_trained_timerange.stopts
        dk.set_new_model_names(pair, new_trained_timerange)
        self.dd.pair_dict[pair]["first"] = False
        if self.dd.pair_dict[pair]["priority"] == 1 and self.scanning:
            with self.lock:
                self.dd.pair_to_end_of_training_queue(pair)
        dk.save_data(model, coin=pair, keras_model=self.keras)

        if self.freqai_info.get("purge_old_models", False):
            self.dd.purge_old_models()
        # self.retrain = False

    # Following methods which are overridden by user made prediction models.
    # See freqai/prediction_models/CatboostPredictionModlel.py for an example.

    @abstractmethod
    def train(self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen) -> Any:
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
        data_dictionary: Dict = the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        return

    @abstractmethod
    def predict(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, first: bool = True
    ) -> Tuple[DataFrame, npt.ArrayLike]:
        """
        Filter the prediction features data and predict with it.
        :param:
        unfiltered_dataframe: Full dataframe for the current backtest period.
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (i.e. SVM and/or DI index)
        """

    def make_labels(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> DataFrame:
        """
        User defines the labels here (target values).
        :params:
        dataframe: DataFrame = the full dataframe for the present training period
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        """

        return

    @abstractmethod
    def return_values(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> DataFrame:
        """
        User defines the dataframe to be returned to strategy here.
        :params:
        dataframe: DataFrame = the full dataframe for the current prediction (live)
        or --timerange (backtesting)
        dk: FreqaiDataKitchen = Data management/analysis tool assoicated to present pair only
        :returns:
        dataframe: DataFrame = dataframe filled with user defined data
        """

        return
