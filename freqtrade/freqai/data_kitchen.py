import copy
import logging
import shutil
from datetime import datetime, timezone
from math import cos, sin
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from scipy import stats
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy.interface import IStrategy


SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600

logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    """
    Class designed to analyze data for a single pair. Employed by the IFreqaiModel class.
    Functionalities include holding, saving, loading, and analyzing the data.

    This object is not persistent, it is reinstantiated for each coin, each time the coin
    model needs to be inferenced or trained.

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

    def __init__(
        self,
        config: Config,
        live: bool = False,
        pair: str = "",
    ):
        self.data: Dict[str, Any] = {}
        self.data_dictionary: Dict[str, DataFrame] = {}
        self.config = config
        self.freqai_config: Dict[str, Any] = config["freqai"]
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path = Path()
        self.label_list: List = []
        self.training_features_list: List = []
        self.model_filename: str = ""
        self.backtesting_results_path = Path()
        self.backtest_predictions_folder: str = "backtesting_predictions"
        self.live = live
        self.pair = pair

        self.svm_model: linear_model.SGDOneClassSVM = None
        self.keras: bool = self.freqai_config.get("keras", False)
        self.set_all_pairs()
        if not self.live:
            if not self.config["timerange"]:
                raise OperationalException(
                    'Please pass --timerange if you intend to use FreqAI for backtesting.')
            self.full_timerange = self.create_fulltimerange(
                self.config["timerange"], self.freqai_config.get("train_period_days", 0)
            )

            (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(
                self.full_timerange,
                config["freqai"]["train_period_days"],
                config["freqai"]["backtest_period_days"],
            )

        self.data['extra_returns_per_train'] = self.freqai_config.get('extra_returns_per_train', {})
        self.thread_count = self.freqai_config.get("data_kitchen_thread_count", -1)
        self.train_dates: DataFrame = pd.DataFrame()
        self.unique_classes: Dict[str, list] = {}
        self.unique_class_list: list = []

    def set_paths(
        self,
        pair: str,
        trained_timestamp: int = None,
    ) -> None:
        """
        Set the paths to the data for the present coin/botloop
        :param metadata: dict = strategy furnished pair metadata
        :param trained_timestamp: int = timestamp of most recent training
        """
        self.full_path = Path(
            self.config["user_data_dir"] / "models" / str(self.freqai_config.get("identifier"))
        )

        self.data_path = Path(
            self.full_path
            / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}"
        )

        return

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> Dict[Any, Any]:
        """
        Given the dataframe for the full history for training, split the data into
        training and test data according to user specified parameters in configuration
        file.
        :param filtered_dataframe: cleaned dataframe ready to be split.
        :param labels: cleaned labels ready to be split.
        """
        feat_dict = self.freqai_config["feature_parameters"]

        if 'shuffle' not in self.freqai_config['data_split_parameters']:
            self.freqai_config["data_split_parameters"].update({'shuffle': False})

        weights: npt.ArrayLike
        if feat_dict.get("weight_factor", 0) > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe[: filtered_dataframe.shape[0]],
                labels,
                weights,
                **self.config["freqai"]["data_split_parameters"],
            )
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights

        # Simplest way to reverse the order of training and test data:
        if self.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return self.build_data_dictionary(
                test_features, train_features, test_labels,
                train_labels, test_weights, train_weights
                )
        else:
            return self.build_data_dictionary(
                train_features, test_features, train_labels,
                test_labels, train_weights, test_weights
            )

    def filter_features(
        self,
        unfiltered_df: DataFrame,
        training_feature_list: List,
        label_list: List = list(),
        training_filter: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the unfiltered dataframe to extract the user requested features/labels and properly
        remove all NaNs. Any row with a NaN is removed from training dataset or replaced with
        0s in the prediction dataset. However, prediction dataset do_predict will reflect any
        row that had a NaN and will shield user from that prediction.

        :param unfiltered_df: the full dataframe for the present training period
        :param training_feature_list: list, the training feature list constructed by
                                      self.build_feature_list() according to user specified
                                      parameters in the configuration file.
        :param labels: the labels for the dataset
        :param training_filter: boolean which lets the function know if it is training data or
                                prediction data to be filtered.
        :returns:
        :filtered_df: dataframe cleaned of NaNs and only containing the user
        requested feature set.
        :labels: labels cleaned of NaNs.
        """
        filtered_df = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)

        drop_index = pd.isnull(filtered_df).any(axis=1)  # get the rows that have NaNs,
        drop_index = drop_index.replace(True, 1).replace(False, 0)  # pep8 requirement.
        if (training_filter):
            const_cols = list((filtered_df.nunique() == 1).loc[lambda x: x].index)
            if const_cols:
                filtered_df = filtered_df.filter(filtered_df.columns.difference(const_cols))
                logger.warning(f"Removed features {const_cols} with constant values.")
            # we don't care about total row number (total no. datapoints) in training, we only care
            # about removing any row with NaNs
            # if labels has multiple columns (user wants to train multiple modelEs), we detect here
            labels = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels = pd.isnull(labels).any(axis=1)
            drop_index_labels = drop_index_labels.replace(True, 1).replace(False, 0)
            dates = unfiltered_df['date']
            filtered_df = filtered_df[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # dropping values
            labels = labels[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # assuming the labels depend entirely on the dataframe here.
            self.train_dates = dates[
                (drop_index == 0) & (drop_index_labels == 0)
            ]
            logger.info(
                f"dropped {len(unfiltered_df) - len(filtered_df)} training points"
                f" due to NaNs in populated dataset {len(unfiltered_df)}."
            )
            if (1 - len(filtered_df) / len(unfiltered_df)) > 0.1 and self.live:
                worst_indicator = str(unfiltered_df.count().idxmin())
                logger.warning(
                    f" {(1 - len(filtered_df)/len(unfiltered_df)) * 100:.0f} percent "
                    " of training data dropped due to NaNs, model may perform inconsistent "
                    f"with expectations. Verify {worst_indicator}"
                )
            self.data["filter_drop_index_training"] = drop_index

        else:
            filtered_df = self.check_pred_labels(filtered_df)
            # we are backtesting so we need to preserve row number to send back to strategy,
            # so now we use do_predict to avoid any prediction based on a NaN
            drop_index = pd.isnull(filtered_df).any(axis=1)
            self.data["filter_drop_index_prediction"] = drop_index
            filtered_df.fillna(0, inplace=True)
            # replacing all NaNs with zeros to avoid issues in 'prediction', but any prediction
            # that was based on a single NaN is ultimately protected from buys with do_predict
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if (len(self.do_predict) - self.do_predict.sum()) > 0:
                logger.info(
                    "dropped %s of %s prediction data points due to NaNs.",
                    len(self.do_predict) - self.do_predict.sum(),
                    len(filtered_df),
                )
            labels = []

        return filtered_df, labels

    def build_data_dictionary(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_labels: DataFrame,
        test_labels: DataFrame,
        train_weights: Any,
        test_weights: Any,
    ) -> Dict:

        self.data_dictionary = {
            "train_features": train_df,
            "test_features": test_df,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_weights": train_weights,
            "test_weights": test_weights,
            "train_dates": self.train_dates
        }

        return self.data_dictionary

    def normalize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        """
        Normalize all data in the data_dictionary according to the training dataset
        :param data_dictionary: dictionary containing the cleaned and
                                split training/test data/labels
        :returns:
        :data_dictionary: updated dictionary with standardized values.
        """

        # standardize the data by training stats
        train_max = data_dictionary["train_features"].max()
        train_min = data_dictionary["train_features"].min()
        data_dictionary["train_features"] = (
            2 * (data_dictionary["train_features"] - train_min) / (train_max - train_min) - 1
        )
        data_dictionary["test_features"] = (
            2 * (data_dictionary["test_features"] - train_min) / (train_max - train_min) - 1
        )

        for item in train_max.keys():
            self.data[item + "_max"] = train_max[item]
            self.data[item + "_min"] = train_min[item]

        for item in data_dictionary["train_labels"].keys():
            if data_dictionary["train_labels"][item].dtype == object:
                continue
            train_labels_max = data_dictionary["train_labels"][item].max()
            train_labels_min = data_dictionary["train_labels"][item].min()
            data_dictionary["train_labels"][item] = (
                2
                * (data_dictionary["train_labels"][item] - train_labels_min)
                / (train_labels_max - train_labels_min)
                - 1
            )
            if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
                data_dictionary["test_labels"][item] = (
                    2
                    * (data_dictionary["test_labels"][item] - train_labels_min)
                    / (train_labels_max - train_labels_min)
                    - 1
                )

            self.data[f"{item}_max"] = train_labels_max
            self.data[f"{item}_min"] = train_labels_min
        return data_dictionary

    def normalize_single_dataframe(self, df: DataFrame) -> DataFrame:

        train_max = df.max()
        train_min = df.min()
        df = (
            2 * (df - train_min) / (train_max - train_min) - 1
        )

        for item in train_max.keys():
            self.data[item + "_max"] = train_max[item]
            self.data[item + "_min"] = train_min[item]

        return df

    def normalize_data_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Normalize a set of data using the mean and standard deviation from
        the associated training data.
        :param df: Dataframe to be standardized
        """

        for item in df.keys():
            df[item] = (
                2
                * (df[item] - self.data[f"{item}_min"])
                / (self.data[f"{item}_max"] - self.data[f"{item}_min"])
                - 1
            )

        return df

    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Denormalize a set of data using the mean and standard deviation from
        the associated training data.
        :param df: Dataframe of predictions to be denormalized
        """

        for label in df.columns:
            if df[label].dtype == object or label in self.unique_class_list:
                continue
            df[label] = (
                (df[label] + 1)
                * (self.data[f"{label}_max"] - self.data[f"{label}_min"])
                / 2
            ) + self.data[f"{label}_min"]

        return df

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: float = 7
    ) -> Tuple[list, list]:
        """
        Function which takes a single time range (tr) and splits it
        into sub timeranges to train and backtest on based on user input
        tr: str, full timerange to train on
        train_split: the period length for the each training (days). Specified in user
        configuration file
        bt_split: the backtesting length (days). Specified in user configuration file
        """

        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(
                f"train_period_days must be an integer greater than 0. Got {train_split}."
            )
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(
                datetime.now(tz=timezone.utc).timestamp()
            )
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        tr_training_list_timerange = []
        tr_backtesting_list_timerange = []
        first = True

        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days

            first = False
            start = datetime.fromtimestamp(timerange_train.startts, tz=timezone.utc)
            stop = datetime.fromtimestamp(timerange_train.stopts, tz=timezone.utc)
            tr_training_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))
            tr_training_list_timerange.append(copy.deepcopy(timerange_train))

            # associated backtest period

            timerange_backtest.startts = timerange_train.stopts

            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            start = datetime.fromtimestamp(timerange_backtest.startts, tz=timezone.utc)
            stop = datetime.fromtimestamp(timerange_backtest.stopts, tz=timezone.utc)
            tr_backtesting_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))
            tr_backtesting_list_timerange.append(copy.deepcopy(timerange_backtest))

            # ensure we are predicting on exactly same amount of data as requested by user defined
            #  --timerange
            if timerange_backtest.stopts == config_timerange.stopts:
                break

        # print(tr_training_list, tr_backtesting_list)
        return tr_training_list_timerange, tr_backtesting_list_timerange

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        """
        Given a full dataframe, extract the user desired window
        :param tr: timerange string that we wish to extract from df
        :param df: Dataframe containing all candles to run the entire backtest. Here
                   it is sliced down to just the present training period.
        """

        start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
        stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
        df = df.loc[df["date"] >= start, :]
        if not self.live:
            df = df.loc[df["date"] < stop, :]

        return df

    def check_pred_labels(self, df_predictions: DataFrame) -> DataFrame:
        """
        Check that prediction feature labels match training feature labels.
        :params:
        :df_predictions: incoming predictions
        """
        train_labels = self.data_dictionary["train_features"].columns
        pred_labels = df_predictions.columns
        num_diffs = len(pred_labels.difference(train_labels))
        if num_diffs != 0:
            df_predictions = df_predictions[train_labels]
            logger.warning(
                f"Removed {num_diffs} features from prediction features, "
                f"these were likely considered constant values during most recent training."
            )

        return df_predictions

    def principal_component_analysis(self) -> None:
        """
        Performs Principal Component Analysis on the data for dimensionality reduction
        and outlier detection (see self.remove_outliers())
        No parameters or returns, it acts on the data_dictionary held by the DataHandler.
        """

        from sklearn.decomposition import PCA  # avoid importing if we dont need it

        pca = PCA(0.999)
        pca = pca.fit(self.data_dictionary["train_features"])
        n_keep_components = pca.n_components_
        self.data["n_kept_components"] = n_keep_components
        n_components = self.data_dictionary["train_features"].shape[1]
        logger.info("reduced feature dimension by %s", n_components - n_keep_components)
        logger.info("explained variance %f", np.sum(pca.explained_variance_ratio_))

        train_components = pca.transform(self.data_dictionary["train_features"])
        self.data_dictionary["train_features"] = pd.DataFrame(
            data=train_components,
            columns=["PC" + str(i) for i in range(0, n_keep_components)],
            index=self.data_dictionary["train_features"].index,
        )
        # normalsing transformed training features
        self.data_dictionary["train_features"] = self.normalize_single_dataframe(
            self.data_dictionary["train_features"])

        # keeping a copy of the non-transformed features so we can check for errors during
        # model load from disk
        self.data["training_features_list_raw"] = copy.deepcopy(self.training_features_list)
        self.training_features_list = self.data_dictionary["train_features"].columns

        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            test_components = pca.transform(self.data_dictionary["test_features"])
            self.data_dictionary["test_features"] = pd.DataFrame(
                data=test_components,
                columns=["PC" + str(i) for i in range(0, n_keep_components)],
                index=self.data_dictionary["test_features"].index,
            )
            # normalise transformed test feature to transformed training features
            self.data_dictionary["test_features"] = self.normalize_data_from_metadata(
                self.data_dictionary["test_features"])

        self.data["n_kept_components"] = n_keep_components
        self.pca = pca

        logger.info(f"PCA reduced total features from  {n_components} to {n_keep_components}")

        if not self.data_path.is_dir():
            self.data_path.mkdir(parents=True, exist_ok=True)

        return None

    def pca_transform(self, filtered_dataframe: DataFrame) -> None:
        """
        Use an existing pca transform to transform data into components
        :param filtered_dataframe: DataFrame = the cleaned dataframe
        """
        pca_components = self.pca.transform(filtered_dataframe)
        self.data_dictionary["prediction_features"] = pd.DataFrame(
            data=pca_components,
            columns=["PC" + str(i) for i in range(0, self.data["n_kept_components"])],
            index=filtered_dataframe.index,
        )
        # normalise transformed predictions to transformed training features
        self.data_dictionary["prediction_features"] = self.normalize_data_from_metadata(
            self.data_dictionary["prediction_features"])

    def compute_distances(self) -> float:
        """
        Compute distances between each training point and every other training
        point. This metric defines the neighborhood of trained data and is used
        for prediction confidence in the Dissimilarity Index
        """
        # logger.info("computing average mean distance for all training points")
        pairwise = pairwise_distances(
            self.data_dictionary["train_features"], n_jobs=self.thread_count)
        # remove the diagonal distances which are itself distances ~0
        np.fill_diagonal(pairwise, np.NaN)
        pairwise = pairwise.reshape(-1, 1)
        avg_mean_dist = pairwise[~np.isnan(pairwise)].mean()

        return avg_mean_dist

    def get_outlier_percentage(self, dropped_pts: npt.NDArray) -> float:
        """
        Check if more than X% of points werer dropped during outlier detection.
        """
        outlier_protection_pct = self.freqai_config["feature_parameters"].get(
            "outlier_protection_percentage", 30)
        outlier_pct = (dropped_pts.sum() / len(dropped_pts)) * 100
        if outlier_pct >= outlier_protection_pct:
            return outlier_pct
        else:
            return 0.0

    def use_SVM_to_remove_outliers(self, predict: bool) -> None:
        """
        Build/inference a Support Vector Machine to detect outliers
        in training data and prediction
        :param predict: bool = If true, inference an existing SVM model, else construct one
        """

        if self.keras:
            logger.warning(
                "SVM outlier removal not currently supported for Keras based models. "
                "Skipping user requested function."
            )
            if predict:
                self.do_predict = np.ones(len(self.data_dictionary["prediction_features"]))
            return

        if predict:
            if not self.svm_model:
                logger.warning("No svm model available for outlier removal")
                return
            y_pred = self.svm_model.predict(self.data_dictionary["prediction_features"])
            do_predict = np.where(y_pred == -1, 0, y_pred)

            if (len(do_predict) - do_predict.sum()) > 0:
                logger.info(f"SVM tossed {len(do_predict) - do_predict.sum()} predictions.")
            self.do_predict += do_predict
            self.do_predict -= 1

        else:
            # use SGDOneClassSVM to increase speed?
            svm_params = self.freqai_config["feature_parameters"].get(
                "svm_params", {"shuffle": False, "nu": 0.1})
            self.svm_model = linear_model.SGDOneClassSVM(**svm_params).fit(
                self.data_dictionary["train_features"]
            )
            y_pred = self.svm_model.predict(self.data_dictionary["train_features"])
            kept_points = np.where(y_pred == -1, 0, y_pred)
            # keep_index = np.where(y_pred == 1)
            outlier_pct = self.get_outlier_percentage(1 - kept_points)
            if outlier_pct:
                logger.warning(
                        f"SVM detected {outlier_pct:.2f}% of the points as outliers. "
                        f"Keeping original dataset."
                )
                self.svm_model = None
                return

            self.data_dictionary["train_features"] = self.data_dictionary["train_features"][
                (y_pred == 1)
            ]
            self.data_dictionary["train_labels"] = self.data_dictionary["train_labels"][
                (y_pred == 1)
            ]
            self.data_dictionary["train_weights"] = self.data_dictionary["train_weights"][
                (y_pred == 1)
            ]

            logger.info(
                f"SVM tossed {len(y_pred) - kept_points.sum()}"
                f" train points from {len(y_pred)} total points."
            )

            # same for test data
            # TODO: This (and the part above) could be refactored into a separate function
            # to reduce code duplication
            if self.freqai_config['data_split_parameters'].get('test_size', 0.1) != 0:
                y_pred = self.svm_model.predict(self.data_dictionary["test_features"])
                kept_points = np.where(y_pred == -1, 0, y_pred)
                self.data_dictionary["test_features"] = self.data_dictionary["test_features"][
                    (y_pred == 1)
                ]
                self.data_dictionary["test_labels"] = self.data_dictionary["test_labels"][(
                    y_pred == 1)]
                self.data_dictionary["test_weights"] = self.data_dictionary["test_weights"][
                    (y_pred == 1)
                ]

            logger.info(
                f"SVM tossed {len(y_pred) - kept_points.sum()}"
                f" test points from {len(y_pred)} total points."
            )

        return

    def use_DBSCAN_to_remove_outliers(self, predict: bool, eps=None) -> None:
        """
        Use DBSCAN to cluster training data and remove "noisy" data (read outliers).
        User controls this via the config param `DBSCAN_outlier_pct` which indicates the
        pct of training data that they want to be considered outliers.
        :param predict: bool = If False (training), iterate to find the best hyper parameters
                        to match user requested outlier percent target.
                        If True (prediction), use the parameters determined from
                        the previous training to estimate if the current prediction point
                        is an outlier.
        """

        if predict:
            if not self.data['DBSCAN_eps']:
                return
            train_ft_df = self.data_dictionary['train_features']
            pred_ft_df = self.data_dictionary['prediction_features']
            num_preds = len(pred_ft_df)
            df = pd.concat([train_ft_df, pred_ft_df], axis=0, ignore_index=True)
            clustering = DBSCAN(eps=self.data['DBSCAN_eps'],
                                min_samples=self.data['DBSCAN_min_samples'],
                                n_jobs=self.thread_count
                                ).fit(df)
            do_predict = np.where(clustering.labels_[-num_preds:] == -1, 0, 1)

            if (len(do_predict) - do_predict.sum()) > 0:
                logger.info(f"DBSCAN tossed {len(do_predict) - do_predict.sum()} predictions")
            self.do_predict += do_predict
            self.do_predict -= 1

        else:

            def normalise_distances(distances):
                normalised_distances = (distances - distances.min()) / \
                                        (distances.max() - distances.min())
                return normalised_distances

            def rotate_point(origin, point, angle):
                # rotate a point counterclockwise by a given angle (in radians)
                # around a given origin
                x = origin[0] + cos(angle) * (point[0] - origin[0]) - \
                                    sin(angle) * (point[1] - origin[1])
                y = origin[1] + sin(angle) * (point[0] - origin[0]) + \
                    cos(angle) * (point[1] - origin[1])
                return (x, y)

            MinPts = int(len(self.data_dictionary['train_features'].index) * 0.25)
            # measure pairwise distances to nearest neighbours
            neighbors = NearestNeighbors(
                n_neighbors=MinPts, n_jobs=self.thread_count)
            neighbors_fit = neighbors.fit(self.data_dictionary['train_features'])
            distances, _ = neighbors_fit.kneighbors(self.data_dictionary['train_features'])
            distances = np.sort(distances, axis=0).mean(axis=1)

            normalised_distances = normalise_distances(distances)
            x_range = np.linspace(0, 1, len(distances))
            line = np.linspace(normalised_distances[0],
                               normalised_distances[-1], len(normalised_distances))
            deflection = np.abs(normalised_distances - line)
            max_deflection_loc = np.where(deflection == deflection.max())[0][0]
            origin = x_range[max_deflection_loc], line[max_deflection_loc]
            point = x_range[max_deflection_loc], normalised_distances[max_deflection_loc]
            rot_angle = np.pi / 4
            elbow_loc = rotate_point(origin, point, rot_angle)

            epsilon = elbow_loc[1] * (distances[-1] - distances[0]) + distances[0]

            clustering = DBSCAN(eps=epsilon, min_samples=MinPts,
                                n_jobs=int(self.thread_count)).fit(
                                                    self.data_dictionary['train_features']
                                                )

            logger.info(f'DBSCAN found eps of {epsilon:.2f}.')

            self.data['DBSCAN_eps'] = epsilon
            self.data['DBSCAN_min_samples'] = MinPts
            dropped_points = np.where(clustering.labels_ == -1, 1, 0)

            outlier_pct = self.get_outlier_percentage(dropped_points)
            if outlier_pct:
                logger.warning(
                        f"DBSCAN detected {outlier_pct:.2f}% of the points as outliers. "
                        f"Keeping original dataset."
                )
                self.data['DBSCAN_eps'] = 0
                return

            self.data_dictionary['train_features'] = self.data_dictionary['train_features'][
                (clustering.labels_ != -1)
            ]
            self.data_dictionary["train_labels"] = self.data_dictionary["train_labels"][
                (clustering.labels_ != -1)
            ]
            self.data_dictionary["train_weights"] = self.data_dictionary["train_weights"][
                (clustering.labels_ != -1)
            ]

            logger.info(
                f"DBSCAN tossed {dropped_points.sum()}"
                f" train points from {len(clustering.labels_)}"
            )

        return

    def compute_inlier_metric(self, set_='train') -> None:
        """
        Compute inlier metric from backwards distance distributions.
        This metric defines how well features from a timepoint fit
        into previous timepoints.
        """

        def normalise(dataframe: DataFrame, key: str) -> DataFrame:
            if set_ == 'train':
                min_value = dataframe.min()
                max_value = dataframe.max()
                self.data[f'{key}_min'] = min_value
                self.data[f'{key}_max'] = max_value
            else:
                min_value = self.data[f'{key}_min']
                max_value = self.data[f'{key}_max']
            return (dataframe - min_value) / (max_value - min_value)

        no_prev_pts = self.freqai_config["feature_parameters"]["inlier_metric_window"]

        if set_ == 'train':
            compute_df = copy.deepcopy(self.data_dictionary['train_features'])
        elif set_ == 'test':
            compute_df = copy.deepcopy(self.data_dictionary['test_features'])
        else:
            compute_df = copy.deepcopy(self.data_dictionary['prediction_features'])

        compute_df_reindexed = compute_df.reindex(
            index=np.flip(compute_df.index)
        )

        pairwise = pd.DataFrame(
            np.triu(
                pairwise_distances(compute_df_reindexed, n_jobs=self.thread_count)
            ),
            columns=compute_df_reindexed.index,
            index=compute_df_reindexed.index
        )
        pairwise = pairwise.round(5)

        column_labels = [
            '{}{}'.format('d', i) for i in range(1, no_prev_pts + 1)
        ]
        distances = pd.DataFrame(
            columns=column_labels, index=compute_df.index
        )

        for index in compute_df.index[no_prev_pts:]:
            current_row = pairwise.loc[[index]]
            current_row_no_zeros = current_row.loc[
                :, (current_row != 0).any(axis=0)
            ]
            distances.loc[[index]] = current_row_no_zeros.iloc[
                :, :no_prev_pts
            ]
        distances = distances.replace([np.inf, -np.inf], np.nan)
        drop_index = pd.isnull(distances).any(axis=1)
        distances = distances[drop_index == 0]

        inliers = pd.DataFrame(index=distances.index)
        for key in distances.keys():
            current_distances = distances[key].dropna()
            current_distances = normalise(current_distances, key)
            if set_ == 'train':
                fit_params = stats.weibull_min.fit(current_distances)
                self.data[f'{key}_fit_params'] = fit_params
            else:
                fit_params = self.data[f'{key}_fit_params']
            quantiles = stats.weibull_min.cdf(current_distances, *fit_params)

            df_inlier = pd.DataFrame(
                {key: quantiles}, index=distances.index
            )
            inliers = pd.concat(
                [inliers, df_inlier], axis=1
            )

        inlier_metric = pd.DataFrame(
            data=inliers.sum(axis=1) / no_prev_pts,
            columns=['%-inlier_metric'],
            index=compute_df.index
        )

        inlier_metric = (2 * (inlier_metric - inlier_metric.min()) /
                         (inlier_metric.max() - inlier_metric.min()) - 1)

        if set_ in ('train', 'test'):
            inlier_metric = inlier_metric.iloc[no_prev_pts:]
            compute_df = compute_df.iloc[no_prev_pts:]
            self.remove_beginning_points_from_data_dict(set_, no_prev_pts)
            self.data_dictionary[f'{set_}_features'] = pd.concat(
                [compute_df, inlier_metric], axis=1)
        else:
            self.data_dictionary['prediction_features'] = pd.concat(
                [compute_df, inlier_metric], axis=1)
            self.data_dictionary['prediction_features'].fillna(0, inplace=True)

        logger.info('Inlier metric computed and added to features.')

        return None

    def remove_beginning_points_from_data_dict(self, set_='train', no_prev_pts: int = 10):
        features = self.data_dictionary[f'{set_}_features']
        weights = self.data_dictionary[f'{set_}_weights']
        labels = self.data_dictionary[f'{set_}_labels']
        self.data_dictionary[f'{set_}_weights'] = weights[no_prev_pts:]
        self.data_dictionary[f'{set_}_features'] = features.iloc[no_prev_pts:]
        self.data_dictionary[f'{set_}_labels'] = labels.iloc[no_prev_pts:]

    def add_noise_to_training_features(self) -> None:
        """
        Add noise to train features to reduce the risk of overfitting.
        """
        mu = 0  # no shift
        sigma = self.freqai_config["feature_parameters"]["noise_standard_deviation"]
        compute_df = self.data_dictionary['train_features']
        noise = np.random.normal(mu, sigma, [compute_df.shape[0], compute_df.shape[1]])
        self.data_dictionary['train_features'] += noise
        return

    def find_features(self, dataframe: DataFrame) -> None:
        """
        Find features in the strategy provided dataframe
        :param dataframe: DataFrame = strategy provided dataframe
        :return:
        features: list = the features to be used for training/prediction
        """
        column_names = dataframe.columns
        features = [c for c in column_names if "%" in c]

        if not features:
            raise OperationalException("Could not find any features!")

        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        column_names = dataframe.columns
        labels = [c for c in column_names if "&" in c]
        self.label_list = labels

    def check_if_pred_in_training_spaces(self) -> None:
        """
        Compares the distance from each prediction point to each training data
        point. It uses this information to estimate a Dissimilarity Index (DI)
        and avoid making predictions on any points that are too far away
        from the training data set.
        """

        distance = pairwise_distances(
            self.data_dictionary["train_features"],
            self.data_dictionary["prediction_features"],
            n_jobs=self.thread_count,
        )

        self.DI_values = distance.min(axis=0) / self.data["avg_mean_dist"]

        do_predict = np.where(
            self.DI_values < self.freqai_config["feature_parameters"]["DI_threshold"],
            1,
            0,
        )

        if (len(do_predict) - do_predict.sum()) > 0:
            logger.info(
                f"DI tossed {len(do_predict) - do_predict.sum()} predictions for "
                "being too far from training data."
            )

        self.do_predict += do_predict
        self.do_predict -= 1

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        """
        Set weights so that recent data is more heavily weighted during
        training than older data.
        """
        wfactor = self.config["freqai"]["feature_parameters"]["weight_factor"]
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def get_predictions_to_append(self, predictions: DataFrame,
                                  do_predict: npt.ArrayLike) -> DataFrame:
        """
        Get backtest prediction from current backtest period
        """

        append_df = DataFrame()
        for label in predictions.columns:
            append_df[label] = predictions[label]
            if append_df[label].dtype == object:
                continue
            append_df[f"{label}_mean"] = self.data["labels_mean"][label]
            append_df[f"{label}_std"] = self.data["labels_std"][label]

        for extra_col in self.data["extra_returns_per_train"]:
            append_df["{extra_col}"] = self.data["extra_returns_per_train"][extra_col]

        append_df["do_predict"] = do_predict
        if self.freqai_config["feature_parameters"].get("DI_threshold", 0) > 0:
            append_df["DI_values"] = self.DI_values

        return append_df

    def append_predictions(self, append_df: DataFrame) -> None:
        """
        Append backtest prediction from current backtest period to all previous periods
        """

        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0)

    def fill_predictions(self, dataframe):
        """
        Back fill values to before the backtesting range so that the dataframe matches size
        when it goes back to the strategy. These rows are not included in the backtest.
        """

        len_filler = len(dataframe) - len(self.full_df.index)  # startup_candle_count
        filler_df = pd.DataFrame(
            np.zeros((len_filler, len(self.full_df.columns))), columns=self.full_df.columns
        )

        self.full_df = pd.concat([filler_df, self.full_df], axis=0, ignore_index=True)

        to_keep = [col for col in dataframe.columns if not col.startswith("&")]
        self.return_dataframe = pd.concat([dataframe[to_keep], self.full_df], axis=1)
        self.full_df = DataFrame()

        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period_days: int) -> str:

        if not isinstance(backtest_period_days, int):
            raise OperationalException("backtest_period_days must be an integer")

        if backtest_period_days < 0:
            raise OperationalException("backtest_period_days must be positive")

        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        if backtest_timerange.stopts == 0:
            # typically open ended time ranges do work, however, there are some edge cases where
            # it does not. accommodating these kinds of edge cases just to allow open-ended
            # timerange is not high enough priority to warrant the effort. It is safer for now
            # to simply ask user to add their end date
            raise OperationalException("FreqAI backtesting does not allow open ended timeranges. "
                                       "Please indicate the end date of your desired backtesting. "
                                       "timerange.")
            # backtest_timerange.stopts = int(
            #     datetime.now(tz=timezone.utc).timestamp()
            # )

        backtest_timerange.startts = (
            backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        )
        start = datetime.fromtimestamp(backtest_timerange.startts, tz=timezone.utc)
        stop = datetime.fromtimestamp(backtest_timerange.stopts, tz=timezone.utc)
        full_timerange = start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d")

        self.full_path = Path(
            self.config["user_data_dir"] / "models" / f"{self.freqai_config['identifier']}"
        )

        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                config_path.resolve(),
                Path(self.full_path / config_path.parts[-1]),
            )

        return full_timerange

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        """
        A model age checker to determine if the model is trustworthy based on user defined
        `expiration_hours` in the configuration file.
        :param trained_timestamp: int = The time of training for the most recent model.
        :return:
            bool = If the model is expired or not.
        """
        time = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time = (time - trained_timestamp) / 3600  # hours
        max_time = self.freqai_config.get("expiration_hours", 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def check_if_new_training_required(
        self, trained_timestamp: int
    ) -> Tuple[bool, TimeRange, TimeRange]:

        time = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()

        timeframes = self.freqai_config["feature_parameters"].get("include_timeframes")

        max_tf_seconds = 0
        for tf in timeframes:
            secs = timeframe_to_seconds(tf)
            if secs > max_tf_seconds:
                max_tf_seconds = secs

        # We notice that users like to use exotic indicators where
        # they do not know the required timeperiod. Here we include a factor
        # of safety by multiplying the user considered "max" by 2.
        max_period = self.config.get('startup_candle_count', 20) * 2
        additional_seconds = max_period * max_tf_seconds

        if trained_timestamp != 0:
            elapsed_time = (time - trained_timestamp) / SECONDS_IN_HOUR
            retrain = elapsed_time > self.freqai_config.get("live_retrain_hours", 0)
            if retrain:
                trained_timerange.startts = int(
                    time - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                )
                trained_timerange.stopts = int(time)
                # we want to load/populate indicators on more data than we plan to train on so
                # because most of the indicators have a rolling timeperiod, and are thus NaNs
                # unless they have data further back in time before the start of the train period
                data_load_timerange.startts = int(
                    time
                    - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                    - additional_seconds
                )
                data_load_timerange.stopts = int(time)
        else:  # user passed no live_trained_timerange in config
            trained_timerange.startts = int(
                time - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
            )
            trained_timerange.stopts = int(time)

            data_load_timerange.startts = int(
                time
                - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                - additional_seconds
            )
            data_load_timerange.stopts = int(time)
            retrain = True

        return retrain, trained_timerange, data_load_timerange

    def set_new_model_names(self, pair: str, trained_timerange: TimeRange):

        coin, _ = pair.split("/")
        self.data_path = Path(
            self.full_path
            / f"sub-train-{pair.split('/')[0]}_{int(trained_timerange.stopts)}"
        )

        self.model_filename = f"cb_{coin.lower()}_{int(trained_timerange.stopts)}"

    def set_all_pairs(self) -> None:

        self.all_pairs = copy.deepcopy(
            self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        )
        for pair in self.config.get("exchange", "").get("pair_whitelist"):
            if pair not in self.all_pairs:
                self.all_pairs.append(pair)

    def use_strategy_to_populate_indicators(
        self,
        strategy: IStrategy,
        corr_dataframes: dict = {},
        base_dataframes: dict = {},
        pair: str = "",
        prediction_dataframe: DataFrame = pd.DataFrame(),
    ) -> DataFrame:
        """
        Use the user defined strategy for populating indicators during retrain
        :param strategy: IStrategy = user defined strategy object
        :param corr_dataframes: dict = dict containing the informative pair dataframes
                                (for user defined timeframes)
        :param base_dataframes: dict = dict containing the current pair dataframes
                                (for user defined timeframes)
        :param metadata: dict = strategy furnished pair metadata
        :returns:
        dataframe: DataFrame = dataframe containing populated indicators
        """

        # for prediction dataframe creation, we let dataprovider handle everything in the strategy
        # so we create empty dictionaries, which allows us to pass None to
        # `populate_any_indicators()`. Signaling we want the dp to give us the live dataframe.
        tfs = self.freqai_config["feature_parameters"].get("include_timeframes")
        pairs = self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        if not prediction_dataframe.empty:
            dataframe = prediction_dataframe.copy()
            for tf in tfs:
                base_dataframes[tf] = None
                for p in pairs:
                    if p not in corr_dataframes:
                        corr_dataframes[p] = {}
                    corr_dataframes[p][tf] = None
        else:
            dataframe = base_dataframes[self.config["timeframe"]].copy()

        sgi = False
        for tf in tfs:
            if tf == tfs[-1]:
                sgi = True  # doing this last allows user to use all tf raw prices in labels
            dataframe = strategy.populate_any_indicators(
                pair,
                dataframe.copy(),
                tf,
                informative=base_dataframes[tf],
                set_generalized_indicators=sgi
            )
            if pairs:
                for i in pairs:
                    if pair in i:
                        continue  # dont repeat anything from whitelist
                    dataframe = strategy.populate_any_indicators(
                        i,
                        dataframe.copy(),
                        tf,
                        informative=corr_dataframes[i][tf]
                    )

        self.get_unique_classes_from_labels(dataframe)

        return dataframe

    def fit_labels(self) -> None:
        """
        Fit the labels with a gaussian distribution
        """
        import scipy as spy

        self.data["labels_mean"], self.data["labels_std"] = {}, {}
        for label in self.data_dictionary["train_labels"].columns:
            if self.data_dictionary["train_labels"][label].dtype == object:
                continue
            f = spy.stats.norm.fit(self.data_dictionary["train_labels"][label])
            self.data["labels_mean"][label], self.data["labels_std"][label] = f[0], f[1]

        # incase targets are classifications
        for label in self.unique_class_list:
            self.data["labels_mean"][label], self.data["labels_std"][label] = 0, 0

        return

    def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove the features from the dataframe before returning it to strategy. This keeps it
        compact for Frequi purposes.
        """
        to_keep = [
            col for col in dataframe.columns if not col.startswith("%") or col.startswith("%%")
        ]
        return dataframe[to_keep]

    def get_unique_classes_from_labels(self, dataframe: DataFrame) -> None:

        # self.find_features(dataframe)
        self.find_labels(dataframe)

        for key in self.label_list:
            if dataframe[key].dtype == object:
                self.unique_classes[key] = dataframe[key].dropna().unique()

        if self.unique_classes:
            for label in self.unique_classes:
                self.unique_class_list += list(self.unique_classes[label])

    def save_backtesting_prediction(
        self, append_df: DataFrame
    ) -> None:
        """
        Save prediction dataframe from backtesting to h5 file format
        :param append_df: dataframe for backtesting period
        """
        full_predictions_folder = Path(self.full_path / self.backtest_predictions_folder)
        if not full_predictions_folder.is_dir():
            full_predictions_folder.mkdir(parents=True, exist_ok=True)

        append_df.to_hdf(self.backtesting_results_path, key='append_df', mode='w')

    def get_backtesting_prediction(
        self
    ) -> DataFrame:
        """
        Get prediction dataframe from h5 file format
        """
        append_df = pd.read_hdf(self.backtesting_results_path)
        return append_df

    def check_if_backtest_prediction_exists(
        self
    ) -> bool:
        """
        Check if a backtesting prediction already exists
        :param dk: FreqaiDataKitchen
        :return:
        :boolean: whether the prediction file exists or not.
        """
        path_to_predictionfile = Path(self.full_path /
                                      self.backtest_predictions_folder /
                                      f"{self.model_filename}_prediction.h5")
        self.backtesting_results_path = path_to_predictionfile

        file_exists = path_to_predictionfile.is_file()
        if file_exists:
            logger.info(f"Found backtesting prediction file at {path_to_predictionfile}")
        else:
            logger.info(
                f"Could not find backtesting prediction file at {path_to_predictionfile}"
            )
        return file_exists
