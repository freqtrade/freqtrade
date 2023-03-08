import logging
from typing import Any, Dict, Tuple, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import functional as F

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.base_models.BasePyTorchModel import BasePyTorchModel
from freqtrade.freqai.base_models.PyTorchModelTrainer import PyTorchModelTrainer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.PyTorchMLPModel import PyTorchMLPModel


logger = logging.getLogger(__name__)


class PyTorchClassifierMultiTarget(BasePyTorchModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiclass_names = self.freqai_info.get("multiclass_target_names", None)
        logger.info(f"setting multiclass_names: {self.multiclass_names}")
        if not self.multiclass_names:
            raise OperationalException(
                "Missing 'multiclass_names' in freqai_info, "
                "multi class pytorch classifier model requires predefined list of "
                "class names matching the strategy being used."
            )

        self.class_name_to_index = {s: i for i, s in enumerate(self.multiclass_names)}
        self.index_to_class_name = {i: s for i, s in enumerate(self.multiclass_names)}
        logger.info(f"class_name_to_index: {self.class_name_to_index}")

        model_training_parameters = self.freqai_info["model_training_parameters"]
        self.n_hidden = model_training_parameters.get("n_hidden", 1024)
        self.max_iters = model_training_parameters.get("max_iters", 100)
        self.batch_size = model_training_parameters.get("batch_size", 64)
        self.learning_rate = model_training_parameters.get("learning_rate", 3e-4)
        self.eval_iters = model_training_parameters.get("eval_iters", 10)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param tensor_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """
        self.encode_classes_name(data_dictionary, dk)
        n_features = data_dictionary['train_features'].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features,
            hidden_dim=self.n_hidden,
            output_dim=len(self.multiclass_names)
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        init_model = self.get_init_model(dk.pair)
        trainer = PyTorchModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            batch_size=self.batch_size,
            max_iters=self.max_iters,
            eval_iters=self.eval_iters,
            init_model=init_model
        )
        trainer.fit(data_dictionary)
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
        ).float().to(self.device)

        logits = self.model.model(dk.data_dictionary["prediction_features"])
        probs = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        predicted_classes_str = self.decode_classes_name(predicted_classes)

        pred_df_prob = DataFrame(probs.detach().numpy(), columns=self.multiclass_names)
        pred_df = DataFrame(predicted_classes_str, columns=[dk.label_list[0]])
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)
        return (pred_df, dk.do_predict)

    def encode_classes_name(self, data_dictionary: Dict[str, pd.DataFrame], dk: FreqaiDataKitchen):
        """
        encode class name str -> int
        assuming first column of *_labels data frame to contain class names
        """
        target_column_name = dk.label_list[0]
        for split in ["train", "test"]:
            label_df = data_dictionary[f"{split}_labels"]
            self.assert_valid_class_names(label_df[target_column_name])
            label_df[target_column_name] = list(
                map(lambda x: self.class_name_to_index[x], label_df[target_column_name])
            )

    def assert_valid_class_names(self, labels: pd.Series):
        non_defined_labels = set(labels) - set(self.multiclass_names)
        if len(non_defined_labels) != 0:
            raise OperationalException(
                f"Found non defined labels: {non_defined_labels}, ",
                f"expecting labels: {self.multiclass_names}"
            )

    def decode_classes_name(self, classes: torch.Tensor[int]) -> List[str]:
        """
        decode class name int -> str
        """
        return list(map(lambda x: self.index_to_class_name[x.item()], classes))
