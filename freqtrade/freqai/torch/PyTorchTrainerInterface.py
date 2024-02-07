from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch import nn


class PyTorchTrainerInterface(ABC):

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> None:
        """
        :param data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        :param splits: splits to use in training, splits must contain "train",
        optional "test" could be added by setting freqai.data_split_parameters.test_size > 0
        in the config file.

         - Calculates the predicted output for the batch using the PyTorch model.
         - Calculates the loss between the predicted and actual output using a loss function.
         - Computes the gradients of the loss with respect to the model's parameters using
           backpropagation.
         - Updates the model's parameters using an optimizer.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        - Saving any nn.Module state_dict
        - Saving model_meta_data, this dict should contain any additional data that the
          user needs to store. e.g class_names for classification models.
        """

    def load(self, path: Path) -> nn.Module:
        """
        :param path: path to zip file.
        :returns: pytorch model.
        """
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    @abstractmethod
    def load_from_checkpoint(self, checkpoint: Dict) -> nn.Module:
        """
        when using continual_learning, DataDrawer will load the dictionary
        (containing state dicts and model_meta_data) by calling torch.load(path).
        you can access this dict from any class that inherits IFreqaiModel by calling
        get_init_model method.
        :checkpoint checkpoint: dict containing the model & optimizer state dicts,
        model_meta_data, etc..
        """
