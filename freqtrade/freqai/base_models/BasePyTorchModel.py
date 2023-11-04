import logging
from abc import ABC, abstractmethod

import torch

from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor


logger = logging.getLogger(__name__)


class BasePyTorchModel(IFreqaiModel, ABC):
    """
    Base class for PyTorch type models.
    User *must* inherit from this class and set fit() and predict() and
    data_convertor property.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        test_size = self.freqai_info.get('data_split_parameters', {}).get('test_size')
        self.splits = ["train", "test"] if test_size != 0 else ["train"]
        self.window_size = self.freqai_info.get("conv_width", 1)

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        a class responsible for converting `*_features` & `*_labels` pandas dataframes
        to pytorch tensors.
        """
        raise NotImplementedError("Abstract property")
