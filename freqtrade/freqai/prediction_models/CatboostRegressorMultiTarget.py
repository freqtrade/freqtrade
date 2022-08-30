import logging
from typing import Any, Dict

from catboost import CatBoostRegressor  # , Pool
from sklearn.multioutput import MultiOutputRegressor

from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel


logger = logging.getLogger(__name__)


class CatboostRegressorMultiTarget(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        cbr = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters,
        )

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
        sample_weight = data_dictionary["train_weights"]

        model = MultiOutputRegressor(estimator=cbr)
        model.fit(X=X, y=y, sample_weight=sample_weight)  # , eval_set=eval_set)

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            train_score = model.score(X, y)
            test_score = model.score(*eval_set)
            logger.info(f"Train score {train_score}, Test score {test_score}")
        return model
