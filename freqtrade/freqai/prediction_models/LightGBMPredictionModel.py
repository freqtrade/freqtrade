import logging
from typing import Any, Dict

from lightgbm import LGBMRegressor

from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel


logger = logging.getLogger(__name__)


class LightGBMPredictionModel(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        model = LGBMRegressor(**self.model_training_parameters)
        model.fit(X=X, y=y, eval_set=eval_set)

        return model
