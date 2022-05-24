import logging
from typing import Any, Dict, Tuple

from catboost import CatBoostRegressor, Pool
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class CatboostPredictionModel(IFreqaiModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def make_labels(self, dataframe: DataFrame, dh: FreqaiDataKitchen) -> DataFrame:
        """
        User defines the labels here (target values).
        :params:
        :dataframe: the full dataframe for the present training period
        """

        dataframe["s"] = (
            dataframe["close"]
            .shift(-self.feature_parameters["period"])
            .rolling(self.feature_parameters["period"])
            .mean()
            / dataframe["close"]
            - 1
        )
        dh.data["s_mean"] = dataframe["s"].mean()
        dh.data["s_std"] = dataframe["s"].std()

        # logger.info("label mean", dh.data["s_mean"], "label std", dh.data["s_std"])

        return dataframe["s"]

    def train(self, unfiltered_dataframe: DataFrame,
              metadata: dict, dh: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
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
        dh.training_features_list = dh.find_features(unfiltered_dataframe)
        unfiltered_labels = self.make_labels(unfiltered_dataframe, dh)
        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dh.filter_features(
            unfiltered_dataframe,
            dh.training_features_list,
            unfiltered_labels,
            training_filter=True,
        )

        # split data into train/test data.
        data_dictionary = dh.make_train_test_datasets(features_filtered, labels_filtered)
        # standardize all data based on train_dataset only
        data_dictionary = dh.standardize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dh)

        logger.info(f'Training model on {len(dh.training_features_list)} features')
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        model = self.fit(data_dictionary)

        logger.info(f'--------------------done training {metadata["pair"]}--------------------')

        return model

    def fit(self, data_dictionary: Dict) -> Any:
        """
        User sets up the training and test data to fit their desired model here
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
            allow_writing_files=False,
            verbose=100, early_stopping_rounds=400, **self.model_training_parameters
        )
        model.fit(X=train_data, eval_set=test_data)

        return model

    def predict(self, unfiltered_dataframe: DataFrame,
                dh: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        # logger.info("--------------------Starting prediction--------------------")

        original_feature_list = dh.find_features(unfiltered_dataframe)
        filtered_dataframe, _ = dh.filter_features(
            unfiltered_dataframe, original_feature_list, training_filter=False
        )
        filtered_dataframe = dh.standardize_data_from_metadata(filtered_dataframe)
        dh.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dh)

        predictions = self.model.predict(dh.data_dictionary["prediction_features"])

        # compute the non-standardized predictions
        dh.predictions = (predictions + 1) * (dh.data["labels_max"] -
                                              dh.data["labels_min"]) / 2 + dh.data["labels_min"]

        # logger.info("--------------------Finished prediction--------------------")

        return (dh.predictions, dh.do_predict)
