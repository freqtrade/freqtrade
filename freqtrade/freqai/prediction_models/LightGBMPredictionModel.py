import logging
from typing import Any, Dict, Tuple

from lightgbm import LGBMRegressor
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class LightGBMPredictionModel(IFreqaiModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def return_values(self, dataframe: DataFrame, dh: FreqaiDataKitchen) -> DataFrame:

        dataframe["prediction"] = dh.full_predictions
        dataframe["do_predict"] = dh.full_do_predict
        dataframe["target_mean"] = dh.full_target_mean
        dataframe["target_std"] = dh.full_target_std
        if self.freqai_info.get('feature_parameters', {}).get('DI_threshold', 0) > 0:
            dataframe["DI"] = dh.full_DI_values

        return dataframe

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

        return dataframe["s"]

    def train(self, unfiltered_dataframe: DataFrame,
              pair: str, dh: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datahkitchen
        for storing, saving, loading, and analyzing the data.
        :params:
        :unfiltered_dataframe: Full dataframe for the current training period
        :metadata: pair metadata from strategy.
        :returns:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info('--------------------Starting training '
                    f'{pair} --------------------')

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
        dh.fit_labels()  # fit labels to a cauchy distribution so we know what to expect in strategy
        # normalize all data based on train_dataset only
        data_dictionary = dh.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dh)

        logger.info(f'Training model on {len(dh.data_dictionary["train_features"].columns)}'
                    ' features')
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        model = self.fit(data_dictionary)

        logger.info(f'--------------------done training {pair}--------------------')

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

        eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        model = LGBMRegressor(seed=42, n_estimators=2000, verbosity=1, force_col_wise=True)
        model.fit(X=X, y=y, eval_set=eval_set)

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
        filtered_dataframe = dh.normalize_data_from_metadata(filtered_dataframe)
        dh.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dh, filtered_dataframe)

        predictions = self.model.predict(dh.data_dictionary["prediction_features"])

        # compute the non-normalized predictions
        dh.predictions = (predictions + 1) * (dh.data["labels_max"] -
                                              dh.data["labels_min"]) / 2 + dh.data["labels_min"]

        # logger.info("--------------------Finished prediction--------------------")

        return (dh.predictions, dh.do_predict)
