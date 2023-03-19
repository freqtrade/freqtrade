import logging

import torch.nn as nn
from torch import Tensor


logger = logging.getLogger(__name__)


class PyTorchMLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super(PyTorchMLPModel, self).__init__()
        hidden_dim: int = kwargs.get("hidden_dim", 1024)
        dropout_percent: int = kwargs.get("dropout_percent", 0.2)
        n_layer: int = kwargs.get("n_layer", 1)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[Block(hidden_dim, dropout_percent) for _ in range(n_layer)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.output_layer(x)
        return logits


class Block(nn.Module):
    def __init__(self, hidden_dim: int, dropout_percent: int):
        super(Block, self).__init__()
        self.ff = FeedForward(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_percent)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.ff(self.ln(x))
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
