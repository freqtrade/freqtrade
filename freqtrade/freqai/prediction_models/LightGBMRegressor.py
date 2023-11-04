import logging
from typing import Any, Dict

from lightgbm import LGBMRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class LightGBMRegressor(BaseRegressionModel):
    """
    User created prediction model. The class inherits IFreqaiModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = data_dictionary["test_weights"]
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        init_model = self.get_init_model(dk.pair)

        model = LGBMRegressor(**self.model_training_parameters)

        model.fit(X=X, y=y, eval_set=eval_set, sample_weight=train_weights,
                  eval_sample_weight=[eval_weights], init_model=init_model)

        return model
