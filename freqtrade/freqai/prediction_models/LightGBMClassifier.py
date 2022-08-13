import logging
from typing import Any, Dict

from lightgbm import LGBMClassifier

from freqtrade.freqai.prediction_models.BaseClassifierModel import BaseClassifierModel


logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifierModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
        else:
            eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        model = LGBMClassifier(**self.model_training_parameters)

        model.fit(X=X, y=y, eval_set=eval_set)

        return model
