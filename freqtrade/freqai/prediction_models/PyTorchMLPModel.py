import logging


import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PyTorchMLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PyTorchMLPModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits
