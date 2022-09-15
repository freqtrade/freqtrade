import logging
from typing import Any, Dict, Tuple

from catboost import CatBoostClassifier  # , Pool
from sklearn.multioutput import MultiOutputClassifier
from pandas import DataFrame
import numpy as np

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel


logger = logging.getLogger(__name__)


class CatboostPredictionBinaryMultiModel(BaseClassifierModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """
    
    def fit_augmented(self, data_dictionary: Dict, dk: FreqaiDataKitchen,) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        cbr = CatBoostClassifier(
            allow_writing_files=False,
            gpu_ram_part=0.5,
            verbose=100,
            early_stopping_rounds=400,
            **self.model_training_parameters,
        )

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        if data_dictionary["test_features"].size:
            eval_set = (data_dictionary["test_features"],
                        data_dictionary["test_labels"])
        sample_weight = data_dictionary["train_weights"]

        if True :
            # mu = 0
            # sigma = 0.01
            # noise = np.random.normal(mu, sigma, [X.shape[0], X.shape[1]])
            # Xaugmented = X + noise

            Xaugmented = X + np.random.randn(*X.shape) / 100 * X.std(0)[None, :]
            X = np.vstack((X, Xaugmented))
            y = y.append(y)
            sample_weight = np.tile(sample_weight, 2)

        from collections import Counter
        weights = y.copy()
        for col_name in y:
            cnt = Counter(y[col_name])
            for k, v in cnt.items():
                weights[col_name][y[col_name] == k] = len(y) / v

        # model = MultiOutputClassifier(estimator=cbr)
        model = cbr

        init_model = self.get_init_model(dk.pair)
        
        model.fit(X=X, y=y,
                  sample_weight=sample_weight * weights.sum(1))  # , eval_set=eval_set)
        train_score = model.score(X, y)
        test_score = "Empty"
        if data_dictionary["test_features"].size:
            test_score = model.score(*eval_set)
        logger.info(f"Augmented Train score {train_score}, Augmented Test score {test_score}")
        return model

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen,) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        # cbr = CatBoostClassifier(
        #     allow_writing_files=False,
        #     gpu_ram_part=0.5,
        #     verbose=100,
        #     early_stopping_rounds=400,
        #     **self.model_training_parameters,
        # )

        # X = data_dictionary["train_features"]
        # y = data_dictionary["train_labels"]
        # if data_dictionary["test_features"].size:
        #     eval_set = (data_dictionary["test_features"],
        #                 data_dictionary["test_labels"].values.astype(np.float32))
        # sample_weight = data_dictionary["train_weights"]

        # from collections import Counter
        # weights = y.copy()
        # for col_name in y:
        #     cnt = Counter(y[col_name])
        #     for k, v in cnt.items():
        #         weights[col_name][y[col_name] == k] = len(y) / v

        # model = MultiOutputClassifier(estimator=cbr)
        # model.fit(X=X, Y=y.values.astype(np.float32),
        #           sample_weight=sample_weight * weights.sum(1))  # , eval_set=eval_set)
        # train_score = model.score(X, y.values.astype(np.float32))
        # test_score = "Empty"
        # if data_dictionary["test_features"].size:
        #     test_score = model.score(*eval_set)
        # logger.info(f"Train score {train_score}, Test score {test_score}")

        model = self.fit_augmented(data_dictionary, dk)
        return model
