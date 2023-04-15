import logging
from typing import Any, Dict

from lightgbm import LGBMClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class LightGBMClassifierMultiTarget(BaseClassifierModel):
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

        lgb = LGBMClassifier(**self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            eval_sets = [(None, None)] * data_dictionary['test_labels'].shape[1]  # type: ignore
            for i in range(data_dictionary['test_labels'].shape[1]):
                eval_sets[i] = (  # type: ignore
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i]
                )

        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {'eval_set': eval_sets[i], 'eval_sample_weight': eval_weights,
                 'init_model': init_models[i]})

        model = FreqaiMultiOutputClassifier(estimator=lgb)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model
