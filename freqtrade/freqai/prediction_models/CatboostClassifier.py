import logging
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.prediction_models.BaseClassifierModel import BaseClassifierModel


logger = logging.getLogger(__name__)


class CatboostClassifier(BaseClassifierModel):
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

        train_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        cbr = CatBoostClassifier(
            allow_writing_files=False,
            loss_function='MultiClass',
            **self.model_training_parameters,
        )

        cbr.fit(train_data)

        return cbr
