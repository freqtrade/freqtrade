import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraModel(nn.Module):
    n_kernels_per_group: int
    n_groups: int
    dilations: torch.Tensor
    num_dilations: int
    paddings: torch.Tensor
    divisor: int
    h: int
    W: torch.Tensor

    def __init__(
        self,
        input_length: int,
        n_kernels_per_group: int = 8,
        n_groups: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n_kernels_per_group = n_kernels_per_group  # num kernels per group
        self.n_groups = n_groups  # num groups
        max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9
        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)
        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode="floor").int()
        self.divisor = min(2, self.n_groups)
        self.h = self.n_groups // self.divisor
        self.W = torch.randn(
            self.num_dilations, self.divisor, self.n_kernels_per_group * self.h, 1, 9
        )
        self.W = self.W - self.W.mean(-1, keepdims=True)
        self.W = self.W / self.W.abs().sum(-1, keepdims=True)

    def batch(self, X: torch.Tensor, batch_size: int = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        num_examples: int = X.shape[0]
        Z = []
        for dilation_index in range(self.num_dilations):
            n_dilations = int(self.dilations[dilation_index].item())
            n_padding = int(self.paddings[dilation_index].item())
            for diff_index in range(self.divisor):
                _Z = F.conv1d(
                    X if diff_index == 0 else torch.diff(X),
                    self.W[dilation_index, diff_index],
                    dilation=n_dilations,
                    padding=n_padding,
                ).view(num_examples, self.h, self.n_kernels_per_group, -1)
                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.n_kernels_per_group)
                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.n_kernels_per_group)
                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))
                Z.append(count_max)
                Z.append(count_min)
        Z = torch.cat(Z, 1).view(num_examples, -1)
        return Z
