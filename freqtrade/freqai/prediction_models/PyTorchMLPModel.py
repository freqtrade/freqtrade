import logging

import torch.nn as nn
import torch

logger = logging.getLogger(__name__)


class PyTorchMLPModel(nn.Module):
    """
    A multi-layer perceptron (MLP) model implemented using PyTorch.

    :param input_dim: The number of input features.
    :param output_dim: The number of output classes.
    :param hidden_dim: The number of hidden units in each layer. Default: 256
    :param dropout_percent: The dropout rate for regularization. Default: 0.2
    :param n_layer: The number of layers in the MLP. Default: 1

    :returns: The output of the MLP, with shape (batch_size, output_dim)


    A neural network typically consists of input, output, and hidden layers, where the
    information flows from the input layer through the hidden layers to the output layer.
    In a feedforward neural network, also known as a multilayer perceptron (MLP), the
    information flows in one direction only. Each hidden layer contains multiple units
    or nodes that take input from the previous layer and produce output that goes to the
    next layer.

    The hidden_dim parameter in the FeedForward class refers to the number of units
    (or nodes) in the hidden layer. This parameter controls the complexity of the neural
    network and determines how many nonlinear relationships the network can represent.
    A higher value of hidden_dim allows the network to represent more complex functions
    but may also make the network more prone to overfitting, where the model memorizes
    the training data instead of learning general patterns.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super(PyTorchMLPModel, self).__init__()
        hidden_dim: int = kwargs.get("hidden_dim", 256)
        dropout_percent: int = kwargs.get("dropout_percent", 0.2)
        n_layer: int = kwargs.get("n_layer", 1)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[Block(hidden_dim, dropout_percent) for _ in range(n_layer)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.output_layer(x)
        return logits


class Block(nn.Module):
    """
    A building block for a multi-layer perceptron (MLP) implemented using PyTorch.

    :param hidden_dim: The number of hidden units in the feedforward network.
    :param dropout_percent: The dropout rate for regularization.

    :returns: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout_percent: int):
        super(Block, self).__init__()
        self.ff = FeedForward(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_percent)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(self.ln(x))
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    A fully-connected feedforward neural network block.

    :param hidden_dim: The number of hidden units in the block.
    :return: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
