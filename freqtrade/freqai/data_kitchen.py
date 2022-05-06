import copy
import datetime
import json
import logging
import pickle as pk
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

from freqtrade.configuration import TimeRange


SECONDS_IN_DAY = 86400

logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    """
    Class designed to handle all the data for the IFreqaiModel class model.
    Functionalities include holding, saving, loading, and analyzing the data.
    author: Robert Caulk, rob.caulk@gmail.com
    """

    def __init__(self, config: Dict[str, Any], dataframe: DataFrame):
        self.full_dataframe = dataframe
        self.data: Dict[Any, Any] = {}
        self.data_dictionary: Dict[Any, Any] = {}
        self.config = config
        self.freqai_config = config["freqai"]
        self.predictions = np.array([])
        self.do_predict = np.array([])
        self.target_mean = np.array([])
        self.target_std = np.array([])
        self.full_predictions = np.array([])
        self.full_do_predict = np.array([])
        self.full_target_mean = np.array([])
        self.full_target_std = np.array([])
        self.model_path = Path()
        self.model_filename = ""

        self.full_timerange = self.create_fulltimerange(
            self.config["timerange"], self.freqai_config["train_period"]
        )

        (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(
            self.full_timerange,
            config["freqai"]["train_period"],
            config["freqai"]["backtest_period"],
        )

    def save_data(self, model: Any) -> None:
        """
        Saves all data associated with a model for a single sub-train time range
        :params:
        :model: User trained model which can be reused for inferencing to generate
        predictions
        """

        if not self.model_path.is_dir():
            self.model_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(self.model_path)

        # Save the trained model
        dump(model, save_path / str(self.model_filename + "_model.joblib"))
        self.data["model_path"] = self.model_path
        self.data["model_filename"] = self.model_filename
        self.data["training_features_list"] = list(self.data_dictionary["train_features"].columns)
        # store the metadata
        with open(save_path / str(self.model_filename + "_metadata.json"), "w") as fp:
            json.dump(self.data, fp, default=self.np_encoder)

        # save the train data to file so we can check preds for area of applicability later
        self.data_dictionary["train_features"].to_pickle(
            save_path / str(self.model_filename + "_trained_df.pkl")
        )

        return

    def load_data(self) -> Any:
        """
        loads all data required to make a prediction on a sub-train time range
        :returns:
        :model: User trained model which can be inferenced for new predictions
        """
        model = load(self.model_path / str(self.model_filename + "_model.joblib"))

        with open(self.model_path / str(self.model_filename + "_metadata.json"), "r") as fp:
            self.data = json.load(fp)
            self.training_features_list = self.data["training_features_list"]
            # if self.data.get("training_features_list"):
            #     self.training_features_list = [*self.data.get("training_features_list")]

        self.data_dictionary["train_features"] = pd.read_pickle(
            self.model_path / str(self.model_filename + "_trained_df.pkl")
        )

        self.model_path = self.data["model_path"]
        self.model_filename = self.data["model_filename"]
        if self.config["freqai"]["feature_parameters"]["principal_component_analysis"]:
            self.pca = pk.load(
                open(self.model_path / str(self.model_filename + "_pca_object.pkl"), "rb")
            )

        return model

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> Dict[Any, Any]:
        """
        Given the dataframe for the full history for training, split the data into
        training and test data according to user specified parameters in configuration
        file.
        :filtered_dataframe: cleaned dataframe ready to be split.
        :labels: cleaned labels ready to be split.
        """

        if self.config["freqai"]["feature_parameters"]["weight_factor"] > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

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
            **self.config["freqai"]["data_split_parameters"]
        )

        return self.build_data_dictionary(
            train_features, test_features, train_labels, test_labels, train_weights, test_weights
        )

    def filter_features(
        self,
        unfiltered_dataframe: DataFrame,
        training_feature_list: List,
        labels: DataFrame = pd.DataFrame(),
        training_filter: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the unfiltered dataframe to extract the user requested features and properly
        remove all NaNs. Any row with a NaN is removed from training dataset or replaced with
        0s in the prediction dataset. However, prediction dataset do_predict will reflect any
        row that had a NaN and will shield user from that prediction.
        :params:
        :unfiltered_dataframe: the full dataframe for the present training period
        :training_feature_list: list, the training feature list constructed by
        self.build_feature_list() according to user specified parameters in the configuration file.
        :labels: the labels for the dataset
        :training_filter: boolean which lets the function know if it is training data or
        prediction data to be filtered.
        :returns:
        :filtered_dataframe: dataframe cleaned of NaNs and only containing the user
        requested feature set.
        :labels: labels cleaned of NaNs.
        """
        filtered_dataframe = unfiltered_dataframe.filter(training_feature_list, axis=1)
        drop_index = pd.isnull(filtered_dataframe).any(1)  # get the rows that have NaNs,
        drop_index = drop_index.replace(True, 1).replace(False, 0)  # pep8 requirement.
        if (
            training_filter
        ):  # we don't care about total row number (total no. datapoints) in training, we only care
            # about removing any row with NaNs
            drop_index_labels = pd.isnull(labels)
            drop_index_labels = drop_index_labels.replace(True, 1).replace(False, 0)
            filtered_dataframe = filtered_dataframe[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # dropping values
            labels = labels[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # assuming the labels depend entirely on the dataframe here.
            logger.info(
                "dropped %s training points due to NaNs, ensure all historical data downloaded",
                len(unfiltered_dataframe) - len(filtered_dataframe),
            )
            self.data["filter_drop_index_training"] = drop_index

        else:
            # we are backtesting so we need to preserve row number to send back to strategy,
            # so now we use do_predict to avoid any prediction based on a NaN
            drop_index = pd.isnull(filtered_dataframe).any(1)
            self.data["filter_drop_index_prediction"] = drop_index
            filtered_dataframe.fillna(0, inplace=True)
            # replacing all NaNs with zeros to avoid issues in 'prediction', but any prediction
            # that was based on a single NaN is ultimately protected from buys with do_predict
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            logger.info(
                "dropped %s of %s prediction data points due to NaNs.",
                len(self.do_predict) - self.do_predict.sum(),
                len(filtered_dataframe),
            )

        return filtered_dataframe, labels

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
        }

        return self.data_dictionary

    def standardize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        """
        Standardize all data in the data_dictionary according to the training dataset
        :params:
        :data_dictionary: dictionary containing the cleaned and split training/test data/labels
        :returns:
        :data_dictionary: updated dictionary with standardized values.
        """
        # standardize the data by training stats
        train_mean = data_dictionary["train_features"].mean()
        train_std = data_dictionary["train_features"].std()
        data_dictionary["train_features"] = (
            data_dictionary["train_features"] - train_mean
        ) / train_std
        data_dictionary["test_features"] = (
            data_dictionary["test_features"] - train_mean
        ) / train_std

        train_labels_std = data_dictionary["train_labels"].std()
        train_labels_mean = data_dictionary["train_labels"].mean()
        data_dictionary["train_labels"] = (
            data_dictionary["train_labels"] - train_labels_mean
        ) / train_labels_std
        data_dictionary["test_labels"] = (
            data_dictionary["test_labels"] - train_labels_mean
        ) / train_labels_std

        for item in train_std.keys():
            self.data[item + "_std"] = train_std[item]
            self.data[item + "_mean"] = train_mean[item]

        self.data["labels_std"] = train_labels_std
        self.data["labels_mean"] = train_labels_mean

        return data_dictionary

    def standardize_data_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Standardizes a set of data using the mean and standard deviation from
        the associated training data.
        :params:
        :df: Dataframe to be standardized
        """

        for item in df.keys():
            df[item] = (df[item] - self.data[item + "_mean"]) / self.data[item + "_std"]

        return df

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: int = 7
    ) -> Tuple[list, list]:
        """
        Function which takes a single time range (tr) and splits it
        into sub timeranges to train and backtest on based on user input
        tr: str, full timerange to train on
        train_split: the period length for the each training (days). Specified in user
        configuration file
        bt_split: the backtesting length (dats). Specified in user configuration file
        """

        train_period = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        first = True
        # within_config_timerange = True
        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + bt_period
            timerange_train.stopts = timerange_train.startts + train_period

            first = False
            start = datetime.datetime.utcfromtimestamp(timerange_train.startts)
            stop = datetime.datetime.utcfromtimestamp(timerange_train.stopts)
            tr_training_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))

            # associated backtest period

            timerange_backtest.startts = timerange_train.stopts

            timerange_backtest.stopts = timerange_backtest.startts + bt_period

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            start = datetime.datetime.utcfromtimestamp(timerange_backtest.startts)
            stop = datetime.datetime.utcfromtimestamp(timerange_backtest.stopts)
            tr_backtesting_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))

            # ensure we are predicting on exactly same amount of data as requested by user defined
            #  --timerange
            if timerange_backtest.stopts == config_timerange.stopts:
                break

        print(tr_training_list, tr_backtesting_list)
        return tr_training_list, tr_backtesting_list

    def slice_dataframe(self, tr: str, df: DataFrame) -> DataFrame:
        """
        Given a full dataframe, extract the user desired window
        :params:
        :tr: timerange string that we wish to extract from df
        :df: Dataframe containing all candles to run the entire backtest. Here
        it is sliced down to just the present training period.
        """
        timerange = TimeRange.parse_timerange(tr)
        start = datetime.datetime.fromtimestamp(timerange.startts, tz=datetime.timezone.utc)
        stop = datetime.datetime.fromtimestamp(timerange.stopts, tz=datetime.timezone.utc)
        df = df.loc[df["date"] >= start, :]
        df = df.loc[df["date"] <= stop, :]

        return df

    def principal_component_analysis(self) -> None:
        """
        Performs Principal Component Analysis on the data for dimensionality reduction
        and outlier detection (see self.remove_outliers())
        No parameters or returns, it acts on the data_dictionary held by the DataHandler.
        """

        from sklearn.decomposition import PCA  # avoid importing if we dont need it

        n_components = self.data_dictionary["train_features"].shape[1]
        pca = PCA(n_components=n_components)
        pca = pca.fit(self.data_dictionary["train_features"])
        n_keep_components = np.argmin(pca.explained_variance_ratio_.cumsum() < 0.999)
        pca2 = PCA(n_components=n_keep_components)
        self.data["n_kept_components"] = n_keep_components
        pca2 = pca2.fit(self.data_dictionary["train_features"])
        logger.info("reduced feature dimension by %s", n_components - n_keep_components)
        logger.info("explained variance %f", np.sum(pca2.explained_variance_ratio_))
        train_components = pca2.transform(self.data_dictionary["train_features"])
        test_components = pca2.transform(self.data_dictionary["test_features"])

        self.data_dictionary["train_features"] = pd.DataFrame(
            data=train_components,
            columns=["PC" + str(i) for i in range(0, n_keep_components)],
            index=self.data_dictionary["train_features"].index,
        )

        self.data_dictionary["test_features"] = pd.DataFrame(
            data=test_components,
            columns=["PC" + str(i) for i in range(0, n_keep_components)],
            index=self.data_dictionary["test_features"].index,
        )

        self.data["n_kept_components"] = n_keep_components
        self.pca = pca2

        if not self.model_path.is_dir():
            self.model_path.mkdir(parents=True, exist_ok=True)
        pk.dump(pca2, open(self.model_path / str(self.model_filename + "_pca_object.pkl"), "wb"))

        return None

    def compute_distances(self) -> float:
        logger.info("computing average mean distance for all training points")
        pairwise = pairwise_distances(self.data_dictionary["train_features"], n_jobs=-1)
        avg_mean_dist = pairwise.mean(axis=1).mean()
        logger.info("avg_mean_dist %s", avg_mean_dist)

        return avg_mean_dist

    def remove_outliers(self, predict: bool) -> None:
        """
        Remove data that looks like an outlier based on the distribution of each
        variable.
        :params:
        :predict: boolean which tells the function if this is prediction data or
        training data coming in.
        """

        lower_quantile = self.data_dictionary["train_features"].quantile(0.001)
        upper_quantile = self.data_dictionary["train_features"].quantile(0.999)

        if predict:

            df = self.data_dictionary["prediction_features"][
                (self.data_dictionary["prediction_features"] < upper_quantile)
                & (self.data_dictionary["prediction_features"] > lower_quantile)
            ]
            drop_index = pd.isnull(df).any(1)
            self.data_dictionary["prediction_features"].fillna(0, inplace=True)
            drop_index = ~drop_index
            do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))

            logger.info(
                "remove_outliers() tossed %s predictions",
                len(do_predict) - do_predict.sum(),
            )
            self.do_predict += do_predict
            self.do_predict -= 1

        else:

            filter_train_df = self.data_dictionary["train_features"][
                (self.data_dictionary["train_features"] < upper_quantile)
                & (self.data_dictionary["train_features"] > lower_quantile)
            ]
            drop_index = pd.isnull(filter_train_df).any(1)
            drop_index = drop_index.replace(True, 1).replace(False, 0)
            self.data_dictionary["train_features"] = self.data_dictionary["train_features"][
                (drop_index == 0)
            ]
            self.data_dictionary["train_labels"] = self.data_dictionary["train_labels"][
                (drop_index == 0)
            ]
            self.data_dictionary["train_weights"] = self.data_dictionary["train_weights"][
                (drop_index == 0)
            ]

            # do the same for the test data
            filter_test_df = self.data_dictionary["test_features"][
                (self.data_dictionary["test_features"] < upper_quantile)
                & (self.data_dictionary["test_features"] > lower_quantile)
            ]
            drop_index = pd.isnull(filter_test_df).any(1)
            drop_index = drop_index.replace(True, 1).replace(False, 0)
            self.data_dictionary["test_labels"] = self.data_dictionary["test_labels"][
                (drop_index == 0)
            ]
            self.data_dictionary["test_features"] = self.data_dictionary["test_features"][
                (drop_index == 0)
            ]
            self.data_dictionary["test_weights"] = self.data_dictionary["test_weights"][
                (drop_index == 0)
            ]

        return

    def build_feature_list(self, config: dict) -> list:
        """
        Build the list of features that will be used to filter
        the full dataframe. Feature list is construced from the
        user configuration file.
        :params:
        :config: Canonical freqtrade config file containing all
        user defined input in config['freqai] dictionary.
        """
        features = []
        for tf in config["freqai"]["timeframes"]:
            for ft in config["freqai"]["base_features"]:
                for n in range(config["freqai"]["feature_parameters"]["shift"] + 1):
                    shift = ""
                    if n > 0:
                        shift = "_shift-" + str(n)
                    features.append(ft + shift + "_" + tf)
                    for p in config["freqai"]["corr_pairlist"]:
                        features.append(p.split("/")[0] + "-" + ft + shift + "_" + tf)

        logger.info("number of features %s", len(features))
        return features

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
            n_jobs=-1,
        )

        do_predict = np.where(
            distance.min(axis=0) / self.data["avg_mean_dist"]
            < self.config["freqai"]["feature_parameters"]["DI_threshold"],
            1,
            0,
        )

        logger.info(
            "Distance checker tossed %s predictions for being too far from training data",
            len(do_predict) - do_predict.sum(),
        )

        self.do_predict += do_predict
        self.do_predict -= 1

    def set_weights_higher_recent(self, num_weights: int) -> int:
        """
        Set weights so that recent data is more heavily weighted during
        training than older data.
        """
        weights = np.zeros(num_weights)
        for i in range(1, len(weights)):
            weights[len(weights) - i] = np.exp(
                -i / (self.config["freqai"]["feature_parameters"]["weight_factor"] * num_weights)
            )
        return weights

    def append_predictions(self, predictions, do_predict, len_dataframe):
        """
        Append backtest prediction from current backtest period to all previous periods
        """

        ones = np.ones(len_dataframe)
        s_mean, s_std = ones * self.data["s_mean"], ones * self.data["s_std"]

        self.full_predictions = np.append(self.full_predictions, predictions)
        self.full_do_predict = np.append(self.full_do_predict, do_predict)
        self.full_target_mean = np.append(self.full_target_mean, s_mean)
        self.full_target_std = np.append(self.full_target_std, s_std)

        return

    def fill_predictions(self, len_dataframe):
        """
        Back fill values to before the backtesting range so that the dataframe matches size
        when it goes back to the strategy. These rows are not included in the backtest.
        """

        filler = np.zeros(len_dataframe - len(self.full_predictions))  # startup_candle_count
        self.full_predictions = np.append(filler, self.full_predictions)
        self.full_do_predict = np.append(filler, self.full_do_predict)
        self.full_target_mean = np.append(filler, self.full_target_mean)
        self.full_target_std = np.append(filler, self.full_target_std)

        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period: int) -> str:
        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        backtest_timerange.startts = backtest_timerange.startts - backtest_period * SECONDS_IN_DAY
        start = datetime.datetime.utcfromtimestamp(backtest_timerange.startts)
        stop = datetime.datetime.utcfromtimestamp(backtest_timerange.stopts)
        full_timerange = start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d")

        self.full_path = Path(
            self.config["user_data_dir"]
            / "models"
            / str(full_timerange + self.freqai_config["identifier"])
        )

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                Path(self.config["config_files"][0]).name,
                Path(self.full_path / self.config["config_files"][0]),
            )

        return full_timerange

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()
