import logging

from typing import Dict
from typing import Any, Dict, Tuple
import numpy.typing as npt

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

from torch.nn import functional as F

from freqtrade.freqai.base_models.BasePytorchModel import BasePytorchModel
from freqtrade.freqai.base_models.PytorchModelTrainer import PytorchModelTrainer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.PytorchMLPModel import MLP

logger = logging.getLogger(__name__)


class PytorchClassifierMultiTarget(BasePytorchModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # todo move to config
        self.n_hidden = 1024
        self.labels = ['0.0', '1.0', '2.0']
        self.max_iters = 100
        self.batch_size = 64
        self.learning_rate = 3e-4

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param tensor_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """
        n_features = data_dictionary['train_features'].shape[-1]
        tensor_dictionary = self.convert_data_to_tensors(data_dictionary)
        model = MLP(
            input_dim=n_features,
            hidden_dim=self.n_hidden,
            output_dim=len(self.labels)
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        init_model = self.get_init_model(dk.pair)
        trainer = PytorchModelTrainer(model, optimizer, init_model=init_model)
        trainer.fit(tensor_dictionary, self.max_iters, self.batch_size)
        return trainer

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.normalize_data_from_metadata(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        self.data_cleaning_predict(dk)
        dk.data_dictionary["prediction_features"] = torch.tensor(
            dk.data_dictionary["prediction_features"].values
        ).to(self.device)

        logits, _ = self.model.model(dk.data_dictionary["prediction_features"])
        probs = F.softmax(logits, dim=-1)
        label_ints = torch.argmax(probs, dim=-1)

        pred_df_prob = DataFrame(probs.detach().numpy(), columns=self.labels)
        pred_df = DataFrame(label_ints, columns=dk.label_list).astype(float).astype(str)
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)
        return (pred_df, dk.do_predict)

    def convert_data_to_tensors(self, data_dictionary: Dict) -> Dict:
        tensor_dictionary = {}
        for split in ['train', 'test']:
            tensor_dictionary[f'{split}_features'] = torch.tensor(
                data_dictionary[f'{split}_features'].values
            ).to(self.device)
            tensor_dictionary[f'{split}_labels'] = torch.tensor(
                data_dictionary[f'{split}_labels'].astype(float).values
            ).long().to(self.device)

        return tensor_dictionary
