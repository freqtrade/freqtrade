import math

import torch
from torch import nn


"""
The architecture is based on the paper “Attention Is All You Need”.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017.
"""


class PyTorchTransformerModel(nn.Module):
    """
    A transformer approach to time series modeling using positional encoding.
    The architecture is based on the paper “Attention Is All You Need”.
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017.
    """

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 7,
        hidden_dim=1024,
        n_layer=2,
        dropout_percent=0.1,
        time_window=10,
        nhead=8,
    ):
        super().__init__()
        self.time_window = time_window
        # ensure the input dimension to the transformer is divisible by nhead
        self.dim_val = input_dim - (input_dim % nhead)
        self.input_net = nn.Sequential(
            nn.Dropout(dropout_percent), nn.Linear(input_dim, self.dim_val)
        )

        # Encode the timeseries with Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.dim_val, max_len=self.dim_val)

        # Define the encoder block of the Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_val, nhead=nhead, dropout=dropout_percent, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

        # the pseudo decoding FC
        self.output_net = nn.Sequential(
            nn.Linear(self.dim_val * time_window, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim), int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim / 4), output_dim),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = x.reshape(-1, 1, self.time_window * x.shape[-1])
        x = self.output_net(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding
        # for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
