import logging
from typing import Any, Dict

import xgboost as xgb

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel


logger = logging.getLogger(__name__)


class XGBoostRegressor(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        xgb.set_config(verbosity=2)
        xgb.config_context(verbosity=2)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]

        sample_weight = data_dictionary["train_weights"]

        xgb_model = self.get_init_model(dk.pair)

        model = xgb.XGBRegressor(**self.model_training_parameters)

        model.fit(X=X, y=y, sample_weight=sample_weight, eval_set=eval_set, xgb_model=xgb_model)

        return model
