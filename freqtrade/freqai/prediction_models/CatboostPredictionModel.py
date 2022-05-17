import logging
from typing import Any, Dict, Tuple

import pandas as pd
from catboost import CatBoostRegressor, Pool
from pandas import DataFrame

from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class CatboostPredictionModel(IFreqaiModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def make_labels(self, dataframe: DataFrame) -> DataFrame:
        """
        User defines the labels here (target values).
        :params:
        :dataframe: the full dataframe for the present training period
        """

        dataframe["s"] = (
            dataframe["close"]
            .shift(-self.feature_parameters["period"])
            .rolling(self.feature_parameters["period"])
            .max()
            / dataframe["close"]
            - 1
        )
        self.dh.data["s_mean"] = dataframe["s"].mean()
        self.dh.data["s_std"] = dataframe["s"].std()

        # logger.info("label mean", self.dh.data["s_mean"], "label std", self.dh.data["s_std"])

        return dataframe["s"]

    def train(self, unfiltered_dataframe: DataFrame, metadata: dict) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datahkitchen
        for storing, saving, loading, and analyzing the data.
        :params:
        :unfiltered_dataframe: Full dataframe for the current training period
        :metadata: pair metadata from strategy.
        :returns:
        :model: Trained model which can be used to inference (self.predict)
        """
        logger.info("--------------------Starting training--------------------")

        # create the full feature list based on user config info
        self.dh.training_features_list = self.dh.build_feature_list(self.config, metadata)
        unfiltered_labels = self.make_labels(unfiltered_dataframe)

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = self.dh.filter_features(
            unfiltered_dataframe,
            self.dh.training_features_list,
            unfiltered_labels,
            training_filter=True,
        )

        # split data into train/test data.
        data_dictionary = self.dh.make_train_test_datasets(features_filtered, labels_filtered)
        # standardize all data based on train_dataset only
        data_dictionary = self.dh.standardize_data(data_dictionary)

        # optional additional data cleaning
        if self.feature_parameters["principal_component_analysis"]:
            self.dh.principal_component_analysis()
        if self.feature_parameters["remove_outliers"]:
            self.dh.remove_outliers(predict=False)
        if self.feature_parameters["DI_threshold"]:
            self.dh.data["avg_mean_dist"] = self.dh.compute_distances()

        logger.info("length of train data %s", len(data_dictionary["train_features"]))

        model = self.fit(data_dictionary)

        logger.info(f'--------------------done training {metadata["pair"]}--------------------')

        return model

    def fit(self, data_dictionary: Dict) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        train_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        test_data = Pool(
            data=data_dictionary["test_features"],
            label=data_dictionary["test_labels"],
            weight=data_dictionary["test_weights"],
        )

        model = CatBoostRegressor(
            verbose=100, early_stopping_rounds=400, **self.model_training_parameters
        )
        model.fit(X=train_data, eval_set=test_data)

        return model

    def predict(self, unfiltered_dataframe: DataFrame, metadata: dict) -> Tuple[DataFrame,
                                                                                DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        # logger.info("--------------------Starting prediction--------------------")

        original_feature_list = self.dh.build_feature_list(self.config, metadata)
        filtered_dataframe, _ = self.dh.filter_features(
            unfiltered_dataframe, original_feature_list, training_filter=False
        )
        filtered_dataframe = self.dh.standardize_data_from_metadata(filtered_dataframe)
        self.dh.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning
        if self.feature_parameters["principal_component_analysis"]:
            pca_components = self.dh.pca.transform(filtered_dataframe)
            self.dh.data_dictionary["prediction_features"] = pd.DataFrame(
                data=pca_components,
                columns=["PC" + str(i) for i in range(0, self.dh.data["n_kept_components"])],
                index=filtered_dataframe.index,
            )

        if self.feature_parameters["remove_outliers"]:
            self.dh.remove_outliers(predict=True)  # creates dropped index

        if self.feature_parameters["DI_threshold"]:
            self.dh.check_if_pred_in_training_spaces()  # sets do_predict

        predictions = self.model.predict(self.dh.data_dictionary["prediction_features"])

        # compute the non-standardized predictions
        self.dh.predictions = predictions * self.dh.data["labels_std"] + self.dh.data["labels_mean"]

        # logger.info("--------------------Finished prediction--------------------")

        return (self.dh.predictions, self.dh.do_predict)
