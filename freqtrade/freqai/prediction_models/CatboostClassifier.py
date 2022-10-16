import logging
import sys
from pathlib import Path
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class CatboostClassifier(BaseClassifierModel):
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

        train_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            test_data = None
        else:
            test_data = Pool(
                data=data_dictionary["test_features"],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
            )

        cbr = CatBoostClassifier(
            allow_writing_files=True,
            loss_function='MultiClass',
            train_dir=Path(dk.data_path),
            **self.model_training_parameters,
        )

        init_model = self.get_init_model(dk.pair)

        cbr.fit(X=train_data, eval_set=test_data, init_model=init_model,
                log_cout=sys.stdout, log_cerr=sys.stderr)

        return cbr
