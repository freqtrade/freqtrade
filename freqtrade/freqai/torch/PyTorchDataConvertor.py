from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import torch


class PyTorchDataConvertor(ABC):
    """
    This class is responsible for converting `*_features` & `*_labels` pandas dataframes
    to pytorch tensors.
    """

    @abstractmethod
    def convert_x(self, df: pd.DataFrame, device: Optional[str] = None) -> torch.Tensor:
        """
        :param df: "*_features" dataframe.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        """

    @abstractmethod
    def convert_y(self, df: pd.DataFrame, device: Optional[str] = None) -> torch.Tensor:
        """
        :param df: "*_labels" dataframe.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        """


class DefaultPyTorchDataConvertor(PyTorchDataConvertor):
    """
    A default conversion that keeps features dataframe shapes.
    """

    def __init__(
            self,
            target_tensor_type: Optional[torch.dtype] = None,
            squeeze_target_tensor: bool = False
    ):
        """
        :param target_tensor_type: type of target tensor, for classification use
            torch.long, for regressor use torch.float or torch.double.
        :param squeeze_target_tensor: controls the target shape, used for loss functions
            that requires 0D or 1D.
        """
        self._target_tensor_type = target_tensor_type
        self._squeeze_target_tensor = squeeze_target_tensor

    def convert_x(self, df: pd.DataFrame, device: Optional[str] = None) -> torch.Tensor:
        x = torch.from_numpy(df.values).float()
        if device:
            x = x.to(device)

        return x

    def convert_y(self, df: pd.DataFrame, device: Optional[str] = None) -> torch.Tensor:
        y = torch.from_numpy(df.values)

        if self._target_tensor_type:
            y = y.to(self._target_tensor_type)

        if self._squeeze_target_tensor:
            y = y.squeeze()

        if device:
            y = y.to(device)

        return y
