import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


class PyTorchMLPModel(nn.Module):
    """
    A multi-layer perceptron (MLP) model implemented using PyTorch.

    This class mainly serves as a simple example for the integration of PyTorch model's
    to freqai. It is not optimized at all and should not be used for production purposes.

    :param input_dim: The number of input features. This parameter specifies the number
        of features in the input data that the MLP will use to make predictions.
    :param output_dim: The number of output classes. This parameter specifies the number
        of classes that the MLP will predict.
    :param hidden_dim: The number of hidden units in each layer. This parameter controls
        the complexity of the MLP and determines how many nonlinear relationships the MLP
        can represent. Increasing the number of hidden units can increase the capacity of
        the MLP to model complex patterns, but it also increases the risk of overfitting
        the training data. Default: 256
    :param dropout_percent: The dropout rate for regularization. This parameter specifies
        the probability of dropping out a neuron during training to prevent overfitting.
        The dropout rate should be tuned carefully to balance between underfitting and
        overfitting. Default: 0.2
    :param n_layer: The number of layers in the MLP. This parameter specifies the number
        of layers in the MLP architecture. Adding more layers to the MLP can increase its
        capacity to model complex patterns, but it also increases the risk of overfitting
        the training data. Default: 1

    :returns: The output of the MLP, with shape (batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        hidden_dim: int = kwargs.get("hidden_dim", 256)
        dropout_percent: int = kwargs.get("dropout_percent", 0.2)
        n_layer: int = kwargs.get("n_layer", 1)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[Block(hidden_dim, dropout_percent) for _ in range(n_layer)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor = tensors[0]
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class Block(nn.Module):
    """
    A building block for a multi-layer perceptron (MLP).

    :param hidden_dim: The number of hidden units in the feedforward network.
    :param dropout_percent: The dropout rate for regularization.

    :returns: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout_percent: int):
        super().__init__()
        self.ff = FeedForward(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_percent)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(self.ln(x))
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    A simple fully-connected feedforward neural network block.

    :param hidden_dim: The number of hidden units in the block.
    :return: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
