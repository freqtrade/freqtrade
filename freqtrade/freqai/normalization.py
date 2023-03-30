from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException


TransformerType = TypeVar('TransformerType', MinMaxScaler, StandardScaler, QuantileTransformer)


def normalization_factory(
            config: Config,
            meta_data: Dict[str, Any],
            pickle_meta_data: Dict[str, Any],
            unique_class_list: list):
    freqai_config: Dict[str, Any] = config["freqai"]
    norm_config_id = freqai_config["feature_parameters"].get("data_normalization", "legacy")
    if norm_config_id.lower() == "legacy":
        return LegacyNormalization(config, meta_data, pickle_meta_data, unique_class_list)
    elif norm_config_id.lower() == "standard":
        return StandardNormalization(config, meta_data, pickle_meta_data, unique_class_list)
    elif norm_config_id.lower() == "minmax":
        return MinMaxNormalization(config, meta_data, pickle_meta_data, unique_class_list)
    elif norm_config_id.lower() == "quantile":
        return QuantileNormalization(config, meta_data, pickle_meta_data, unique_class_list)
    else:
        raise OperationalException(f"Invalid data normalization identifier '{norm_config_id}'")


class Normalization(ABC):
    def __init__(
        self,
            config: Config,
            meta_data: Dict[str, Any],
            pickle_meta_data: Dict[str, Any],
            unique_class_list: list
    ):
        self.freqai_config: Dict[str, Any] = config["freqai"]
        self.data: Dict[str, Any] = meta_data
        self.pkl_data: Dict[str, Any] = pickle_meta_data
        self.unique_class_list: list = unique_class_list

    @abstractmethod
    def normalize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        """"""

    @abstractmethod
    def normalize_single_dataframe(self, df: DataFrame) -> DataFrame:
        """"""

    @abstractmethod
    def normalize_data_from_metadata(self, df: DataFrame) -> DataFrame:
        """"""

    @abstractmethod
    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        """"""


class LegacyNormalization(Normalization):
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

        train_max = [None] * len(df.keys())
        train_min = [None] * len(df.keys())

        for i, item in enumerate(df.keys()):
            train_max[i] = self.data[f"{item}_max"]
            train_min[i] = self.data[f"{item}_min"]

        train_max_series = pd.Series(train_max, index=df.keys())
        train_min_series = pd.Series(train_min, index=df.keys())

        df = (
            2 * (df - train_min_series) / (train_max_series - train_min_series) - 1
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


class SKLearnNormalization(Normalization):
    def __init__(self,
                 config: Config,
                 meta_data: Dict[str, Any],
                 pickle_meta_data: Dict[str, Any],
                 unique_class_list: list,
                 transformer: TransformerType):
        super().__init__(config, meta_data, pickle_meta_data, unique_class_list)
        self.transformer = transformer

    def normalize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        """
        Normalize all data in the data_dictionary according to the training dataset
        :param data_dictionary: dictionary containing the cleaned and
                                split training/test data/labels
        :returns:
        :data_dictionary: updated dictionary with standardized values.
        """

        # standardize the data by training stats
        for column in data_dictionary["train_features"].columns:
            scaler = self.transformer()
            data_dictionary["train_features"][column] = \
                scaler.fit_transform(data_dictionary["train_features"][[column]])
            data_dictionary["test_features"][column] = \
                scaler.transform(data_dictionary["test_features"][[column]])
            self.pkl_data[column + "_scaler"] = scaler

        for column in data_dictionary["train_labels"].columns:
            if data_dictionary["train_labels"][column].dtype == object:
                continue
            scaler = self.transformer()
            data_dictionary["train_labels"][column] = \
                scaler.fit_transform(data_dictionary["train_labels"][[column]])

            if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
                data_dictionary["test_labels"][column] = \
                    scaler.transform(data_dictionary["test_labels"][[column]])

            self.pkl_data[column + "_scaler"] = scaler
        return data_dictionary

    def normalize_single_dataframe(self, df: DataFrame) -> DataFrame:
        for column in df.columns:
            scaler = self.transformer()
            df[column] = scaler.fit_transform(df[[column]])
            self.pkl_data[column + "_scaler"] = scaler

        return df

    def normalize_data_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Normalize a set of data using the mean and standard deviation from
        the associated training data.
        :param df: Dataframe to be standardized
        """

        for column in df.columns:
            df[column] = self.pkl_data[column + "_scaler"].transform(df[[column]])

        return df

    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Denormalize a set of data using the mean and standard deviation from
        the associated training data.
        :param df: Dataframe of predictions to be denormalized
        """

        for column in df.columns:
            if df[column].dtype == object or column in self.unique_class_list:
                continue
            df[column] = self.pkl_data[column + "_scaler"].inverse_transform(df[[column]])

        return df


class StandardNormalization(SKLearnNormalization):
    def __init__(self,
                 config: Config,
                 meta_data: Dict[str, Any],
                 pickle_meta_data: Dict[str, Any],
                 unique_class_list: list):
        super().__init__(config, meta_data, pickle_meta_data, unique_class_list, StandardScaler)


class MinMaxNormalization(SKLearnNormalization):
    def __init__(self,
                 config: Config,
                 meta_data: Dict[str, Any],
                 pickle_meta_data: Dict[str, Any],
                 unique_class_list: list):
        super().__init__(config, meta_data, pickle_meta_data, unique_class_list, MinMaxScaler)


class QuantileNormalization(SKLearnNormalization):
    def __init__(self,
                 config: Config,
                 meta_data: Dict[str, Any],
                 pickle_meta_data: Dict[str, Any],
                 unique_class_list: list):
        super().__init__(config, meta_data, pickle_meta_data, unique_class_list,
                         QuantileTransformer)
