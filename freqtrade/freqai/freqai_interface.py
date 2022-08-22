# import contextlib
import datetime
import logging
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
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
    Base*PredictionModels inherit from this class.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @th0rntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        self.config = config
        self.assert_config(self.config)
        self.freqai_info: Dict[str, Any] = config["freqai"]
        self.data_split_parameters: Dict[str, Any] = config.get("freqai", {}).get(
            "data_split_parameters", {})
        self.model_training_parameters: Dict[str, Any] = config.get("freqai", {}).get(
            "model_training_parameters", {})
        self.feature_parameters = config.get("freqai", {}).get("feature_parameters")
        self.retrain = False
        self.first = True
        self.set_full_path()
        self.follow_mode: bool = self.freqai_info.get("follow_mode", False)
        self.dd = FreqaiDataDrawer(Path(self.full_path), self.config, self.follow_mode)
        self.identifier: str = self.freqai_info.get("identifier", "no_id_provided")
        self.scanning = False
        self.keras: bool = self.freqai_info.get("keras", False)
        if self.keras and self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0):
            self.freqai_info["feature_parameters"]["DI_threshold"] = 0
            logger.warning("DI threshold is not configured for Keras models yet. Deactivating.")
        self.CONV_WIDTH = self.freqai_info.get("conv_width", 2)
        self.pair_it = 0
        self.pair_it_train = 0
        self.total_pairs = len(self.config.get("exchange", {}).get("pair_whitelist"))
        self.last_trade_database_summary: DataFrame = {}
        self.current_trade_database_summary: DataFrame = {}
        self.analysis_lock = Lock()
        self.inference_time: float = 0
        self.train_time: float = 0
        self.begin_time: float = 0
        self.begin_time_train: float = 0
        self.base_tf_seconds = timeframe_to_seconds(self.config['timeframe'])

    def assert_config(self, config: Dict[str, Any]) -> None:

        if not config.get("freqai", {}):
            raise OperationalException("No freqai parameters found in configuration file.")

    def start(self, dataframe: DataFrame, metadata: dict, strategy: IStrategy) -> DataFrame:
        """
        Entry point to the FreqaiModel from a specific pair, it will train a new model if
        necessary before making the prediction.

        :param dataframe: Full dataframe coming from strategy - it contains entire
                           backtesting timerange + additional historical data necessary to train
        the model.
        :param metadata: pair metadata coming from strategy.
        :param strategy: Strategy to train on
        """

        self.live = strategy.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)
        self.dd.set_pair_dict_info(metadata)

        if self.live:
            self.inference_timer('start')
            self.dk = FreqaiDataKitchen(self.config, self.live, metadata["pair"])
            dk = self.start_live(dataframe, metadata, strategy, self.dk)

        # For backtesting, each pair enters and then gets trained for each window along the
        # sliding window defined by "train_period_days" (training window) and "live_retrain_hours"
        # (backtest window, i.e. window immediately following the training window).
        # FreqAI slides the window and sequentially builds the backtesting results before returning
        # the concatenated results for the full backtesting period back to the strategy.
        elif not self.follow_mode:
            self.dk = FreqaiDataKitchen(self.config, self.live, metadata["pair"])
            logger.info(f"Training {len(self.dk.training_timeranges)} timeranges")
            with self.analysis_lock:
                dataframe = self.dk.use_strategy_to_populate_indicators(
                    strategy, prediction_dataframe=dataframe, pair=metadata["pair"]
                )
            dk = self.start_backtesting(dataframe, metadata, self.dk)

        dataframe = dk.remove_features_from_df(dk.return_dataframe)
        self.clean_up()
        if self.live:
            self.inference_timer('stop')
        return dataframe

    def clean_up(self):
        """
        Objects that should be handled by GC already between coins, but
        are explicitly shown here to help demonstrate the non-persistence of these
        objects.
        """
        self.model = None
        self.dk = None

    @threaded
    def start_scanning(self, strategy: IStrategy) -> None:
        """
        Function designed to constantly scan pairs for retraining on a separate thread (intracandle)
        to improve model youth. This function is agnostic to data preparation/collection/storage,
        it simply trains on what ever data is available in the self.dd.
        :param strategy: IStrategy = The user defined strategy class
        """
        while 1:
            time.sleep(1)
            for pair in self.config.get("exchange", {}).get("pair_whitelist"):

                (_, trained_timestamp, _) = self.dd.get_pair_dict_info(pair)

                if self.dd.pair_dict[pair]["priority"] != 1:
                    continue
                dk = FreqaiDataKitchen(self.config, self.live, pair)
                dk.set_paths(pair, trained_timestamp)
                (
                    retrain,
                    new_trained_timerange,
                    data_load_timerange,
                ) = dk.check_if_new_training_required(trained_timestamp)
                dk.set_paths(pair, new_trained_timerange.stopts)

                if retrain:
                    self.train_timer('start')
                    self.train_model_in_series(
                        new_trained_timerange, pair, strategy, dk, data_load_timerange
                    )
                    self.train_timer('stop')

            self.dd.save_historic_predictions_to_disk()

    def start_backtesting(
        self, dataframe: DataFrame, metadata: dict, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        The main broad execution for backtesting. For backtesting, each pair enters and then gets
        trained for each window along the sliding window defined by "train_period_days"
        (training window) and "backtest_period_days" (backtest window, i.e. window immediately
        following the training window). FreqAI slides the window and sequentially builds
        the backtesting results before returning the concatenated results for the full
        backtesting period back to the strategy.
        :param dataframe: DataFrame = strategy passed dataframe
        :param metadata: Dict = pair metadata
        :param dk: FreqaiDataKitchen = Data management/analysis tool associated to present pair only
        :return:
            FreqaiDataKitchen = Data management/analysis tool associated to present pair only
        """

        self.pair_it += 1
        train_it = 0
        # Loop enforcing the sliding window training/backtesting paradigm
        # tr_train is the training time range e.g. 1 historical month
        # tr_backtest is the backtesting time range e.g. the week directly
        # following tr_train. Both of these windows slide through the
        # entire backtest
        for tr_train, tr_backtest in zip(dk.training_timeranges, dk.backtesting_timeranges):
            (_, _, _) = self.dd.get_pair_dict_info(metadata["pair"])
            train_it += 1
            total_trains = len(dk.backtesting_timeranges)
            self.training_timerange = tr_train
            dataframe_train = dk.slice_dataframe(tr_train, dataframe)
            dataframe_backtest = dk.slice_dataframe(tr_backtest, dataframe)

            trained_timestamp = tr_train
            tr_train_startts_str = datetime.datetime.utcfromtimestamp(tr_train.startts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            tr_train_stopts_str = datetime.datetime.utcfromtimestamp(tr_train.stopts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            logger.info(
                f"Training {metadata['pair']}, {self.pair_it}/{self.total_pairs} pairs"
                f" from {tr_train_startts_str} to {tr_train_stopts_str}, {train_it}/{total_trains} "
                "trains"
            )

            dk.data_path = Path(
                dk.full_path
                /
                f"sub-train-{metadata['pair'].split('/')[0]}_{int(trained_timestamp.stopts)}"
                )
            if not self.model_exists(
                metadata["pair"], dk, trained_timestamp=int(trained_timestamp.stopts)
            ):
                dk.find_features(dataframe_train)
                self.model = self.train(dataframe_train, metadata["pair"], dk)
                self.dd.pair_dict[metadata["pair"]]["trained_timestamp"] = int(
                    trained_timestamp.stopts)
                dk.set_new_model_names(metadata["pair"], trained_timestamp)
                self.dd.save_data(self.model, metadata["pair"], dk)
            else:
                self.model = self.dd.load_data(metadata["pair"], dk)

            self.check_if_feature_list_matches_strategy(dataframe_train, dk)

            pred_df, do_preds = self.predict(dataframe_backtest, dk)

            dk.append_predictions(pred_df, do_preds)

        dk.fill_predictions(dataframe)

        return dk

    def start_live(
        self, dataframe: DataFrame, metadata: dict, strategy: IStrategy, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        The main broad execution for dry/live. This function will check if a retraining should be
        performed, and if so, retrain and reset the model.
        :param dataframe: DataFrame = strategy passed dataframe
        :param metadata: Dict = pair metadata
        :param strategy: IStrategy = currently employed strategy
        dk: FreqaiDataKitchen = Data management/analysis tool associated to present pair only
        :returns:
        dk: FreqaiDataKitchen = Data management/analysis tool associated to present pair only
        """

        # update follower
        if self.follow_mode:
            self.dd.update_follower_metadata()

        # get the model metadata associated with the current pair
        (_, trained_timestamp, return_null_array) = self.dd.get_pair_dict_info(metadata["pair"])

        # if the metadata doesn't exist, the follower returns null arrays to strategy
        if self.follow_mode and return_null_array:
            logger.info("Returning null array from follower to strategy")
            self.dd.return_null_values_to_strategy(dataframe, dk)
            return dk

        # append the historic data once per round
        if self.dd.historic_data:
            self.dd.update_historic_data(strategy, dk)
            logger.debug(f'Updating historic data on pair {metadata["pair"]}')

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
                dk.download_all_data_for_training(data_load_timerange, strategy.dp)
                self.dd.load_all_pair_histories(data_load_timerange, dk)

            if not self.scanning:
                self.scanning = True
                self.start_scanning(strategy)

        elif self.follow_mode:
            dk.set_paths(metadata["pair"], trained_timestamp)
            logger.info(
                "FreqAI instance set to follow_mode, finding existing pair "
                f"using { self.identifier }"
            )

        # load the model and associated data into the data kitchen
        self.model = self.dd.load_data(metadata["pair"], dk)

        with self.analysis_lock:
            dataframe = self.dk.use_strategy_to_populate_indicators(
                strategy, prediction_dataframe=dataframe, pair=metadata["pair"]
            )

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
            # first predictions are made on entire historical candle set coming from strategy. This
            # allows FreqUI to show full return values.
            pred_df, do_preds = self.predict(dataframe, dk)
            if pair not in self.dd.historic_predictions:
                self.set_initial_historic_predictions(pred_df, dk, pair)
            self.dd.set_initial_return_values(pair, pred_df)

            dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)
            return
        elif self.dk.check_if_model_expired(trained_timestamp):
            pred_df = DataFrame(np.zeros((2, len(dk.label_list))), columns=dk.label_list)
            do_preds = np.ones(2, dtype=np.int_) * 2
            dk.DI_values = np.zeros(2)
            logger.warning(
                f"Model expired for {pair}, returning null values to strategy. Strategy "
                "construction should take care to consider this event with "
                "prediction == 0 and do_predict == 2"
            )
        else:
            # remaining predictions are made only on the most recent candles for performance and
            # historical accuracy reasons.
            pred_df, do_preds = self.predict(dataframe.iloc[-self.CONV_WIDTH:], dk, first=False)

        if self.freqai_info.get('fit_live_predictions_candles', 0) and self.live:
            self.fit_live_predictions(dk, pair)
        self.dd.append_model_predictions(pair, pred_df, do_preds, dk, len(dataframe))
        dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)

        return

    def check_if_feature_list_matches_strategy(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen
    ) -> None:
        """
        Ensure user is passing the proper feature set if they are reusing an `identifier` pointing
        to a folder holding existing models.
        :param dataframe: DataFrame = strategy provided dataframe
        :param dk: FreqaiDataKitchen = non-persistent data container/analyzer for
                   current coin/bot loop
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
                "Change `identifier` to train from scratch, or ensure the"
                "strategy is furnishing the same features as the pretrained"
                "model"
            )

    def data_cleaning_train(self, dk: FreqaiDataKitchen) -> None:
        """
        Base data cleaning method for train
        Any function inside this method should drop training data points from the filtered_dataframe
        based on user decided logic. See FreqaiDataKitchen::use_SVM_to_remove_outliers() for an
        example of how outlier data points are dropped from the dataframe used for training.
        """

        if self.freqai_info["feature_parameters"].get(
            "principal_component_analysis", False
        ):
            dk.principal_component_analysis()

        if self.freqai_info["feature_parameters"].get("use_SVM_to_remove_outliers", False):
            dk.use_SVM_to_remove_outliers(predict=False)

        if self.freqai_info["feature_parameters"].get("DI_threshold", 0):
            dk.data["avg_mean_dist"] = dk.compute_distances()

        if self.freqai_info["feature_parameters"].get("use_DBSCAN_to_remove_outliers", False):
            if dk.pair in self.dd.old_DBSCAN_eps:
                eps = self.dd.old_DBSCAN_eps[dk.pair]
            else:
                eps = None
            dk.use_DBSCAN_to_remove_outliers(predict=False, eps=eps)
            self.dd.old_DBSCAN_eps[dk.pair] = dk.data['DBSCAN_eps']

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
        if self.freqai_info["feature_parameters"].get(
            "principal_component_analysis", False
        ):
            dk.pca_transform(dataframe)

        if self.freqai_info["feature_parameters"].get("use_SVM_to_remove_outliers", False):
            dk.use_SVM_to_remove_outliers(predict=True)

        if self.freqai_info["feature_parameters"].get("DI_threshold", 0):
            dk.check_if_pred_in_training_spaces()

        if self.freqai_info["feature_parameters"].get("use_DBSCAN_to_remove_outliers", False):
            dk.use_DBSCAN_to_remove_outliers(predict=True)

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
        :return:
        :boolean: whether the model file exists or not.
        """
        coin, _ = pair.split("/")

        if not self.live:
            dk.model_filename = model_filename = f"cb_{coin.lower()}_{trained_timestamp}"

        path_to_modelfile = Path(dk.data_path / f"{model_filename}_model.joblib")
        file_exists = path_to_modelfile.is_file()
        if file_exists and not scanning:
            logger.info("Found model at %s", dk.data_path / dk.model_filename)
        elif not scanning:
            logger.info("Could not find model at %s", dk.data_path / dk.model_filename)
        return file_exists

    def set_full_path(self) -> None:
        self.full_path = Path(
            self.config["user_data_dir"] / "models" / f"{self.freqai_info['identifier']}"
        )
        self.full_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            self.config["config_files"][0],
            Path(self.full_path, Path(self.config["config_files"][0]).name),
        )

    def train_model_in_series(
        self,
        new_trained_timerange: TimeRange,
        pair: str,
        strategy: IStrategy,
        dk: FreqaiDataKitchen,
        data_load_timerange: TimeRange,
    ):
        """
        Retrieve data and train model.
        :param new_trained_timerange: TimeRange = the timerange to train the model on
        :param metadata: dict = strategy provided metadata
        :param strategy: IStrategy = user defined strategy object
        :param dk: FreqaiDataKitchen = non-persistent data container for current coin/loop
        :param data_load_timerange: TimeRange = the amount of data to be loaded
                                    for populate_any_indicators
                                    (larger than new_trained_timerange so that
                                    new_trained_timerange does not contain any NaNs)
        """

        corr_dataframes, base_dataframes = self.dd.get_base_and_corr_dataframes(
            data_load_timerange, pair, dk
        )

        with self.analysis_lock:
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
            self.dd.pair_to_end_of_training_queue(pair)
        self.dd.save_data(model, pair, dk)

        if self.freqai_info.get("purge_old_models", False):
            self.dd.purge_old_models()

    def set_initial_historic_predictions(
        self, pred_df: DataFrame, dk: FreqaiDataKitchen, pair: str
    ) -> None:
        """
        This function is called only if the datadrawer failed to load an
        existing set of historic predictions. In this case, it builds
        the structure and sets fake predictions off the first training
        data. After that, FreqAI will append new real predictions to the
        set of historic predictions.

        These values are used to generate live statistics which can be used
        in the strategy for adaptive values. E.g. &*_mean/std are quantities
        that can computed based on live predictions from the set of historical
        predictions. Those values can be used in the user strategy to better
        assess prediction rarity, and thus wait for probabilistically favorable
        entries relative to the live historical predictions.

        If the user reuses an identifier on a subsequent instance,
        this function will not be called. In that case, "real" predictions
        will be appended to the loaded set of historic predictions.
        :param: df: DataFrame = the dataframe containing the training feature data
        :param: model: Any = A model which was `fit` using a common library such as
        catboost or lightgbm
        :param: dk: FreqaiDataKitchen = object containing methods for data analysis
        :param: pair: str = current pair
        """

        self.dd.historic_predictions[pair] = pred_df
        hist_preds_df = self.dd.historic_predictions[pair]

        for label in hist_preds_df.columns:
            if hist_preds_df[label].dtype == object:
                continue
            hist_preds_df[f'{label}_mean'] = 0
            hist_preds_df[f'{label}_std'] = 0

        hist_preds_df['do_predict'] = 0

        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            hist_preds_df['DI_values'] = 0

        for return_str in dk.data['extra_returns_per_train']:
            hist_preds_df[return_str] = 0

        # # for keras type models, the conv_window needs to be prepended so
        # # viewing is correct in frequi
        if self.freqai_info.get('keras', False):
            n_lost_points = self.freqai_info.get('conv_width', 2)
            zeros_df = DataFrame(np.zeros((n_lost_points, len(hist_preds_df.columns))),
                                 columns=hist_preds_df.columns)
            self.dd.historic_predictions[pair] = pd.concat(
                [zeros_df, hist_preds_df], axis=0, ignore_index=True)

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """
        Fit the labels with a gaussian distribution
        """
        import scipy as spy

        # add classes from classifier label types if used
        full_labels = dk.label_list + dk.unique_class_list

        num_candles = self.freqai_info.get("fit_live_predictions_candles", 100)
        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label in full_labels:
            if self.dd.historic_predictions[dk.pair][label].dtype == object:
                continue
            f = spy.stats.norm.fit(self.dd.historic_predictions[dk.pair][label].tail(num_candles))
            dk.data["labels_mean"][label], dk.data["labels_std"][label] = f[0], f[1]

        return

    def inference_timer(self, do='start'):
        """
        Timer designed to track the cumulative time spent in FreqAI for one pass through
        the whitelist. This will check if the time spent is more than 1/4 the time
        of a single candle, and if so, it will warn the user of degraded performance
        """
        if do == 'start':
            self.pair_it += 1
            self.begin_time = time.time()
        elif do == 'stop':
            end = time.time()
            self.inference_time += (end - self.begin_time)
            if self.pair_it == self.total_pairs:
                logger.info(
                    f'Total time spent inferencing pairlist {self.inference_time:.2f} seconds')
                if self.inference_time > 0.25 * self.base_tf_seconds:
                    logger.warning('Inference took over 25/% of the candle time. Reduce pairlist to'
                                   ' avoid blinding open trades and degrading performance.')
                self.pair_it = 0
                self.inference_time = 0
        return

    def train_timer(self, do='start'):
        """
        Timer designed to track the cumulative time spent training the full pairlist in
        FreqAI.
        """
        if do == 'start':
            self.pair_it_train += 1
            self.begin_time_train = time.time()
        elif do == 'stop':
            end = time.time()
            self.train_time += (end - self.begin_time_train)
            if self.pair_it_train == self.total_pairs:
                logger.info(
                    f'Total time spent training pairlist {self.train_time:.2f} seconds')
                self.pair_it_train = 0
                self.train_time = 0
        return

    # Following methods which are overridden by user made prediction models.
    # See freqai/prediction_models/CatboostPredictionModel.py for an example.

    @abstractmethod
    def train(self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datahandler
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_dataframe: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return: Trained model which can be used to inference (self.predict)
        """

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, Any]) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :param data_dictionary: Dict = the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        return

    @abstractmethod
    def predict(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, first: bool = True
    ) -> Tuple[DataFrame, NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_dataframe: Full dataframe for the current backtest period.
        :param dk: FreqaiDataKitchen = Data management/analysis tool associated to present pair only
        :param first: boolean = whether this is the first prediction or not.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (i.e. SVM and/or DI index)
        """
