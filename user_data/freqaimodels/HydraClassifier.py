from freqtrade.freqai.base_models.BasePyTorchClassifier import BasePyTorchClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.PyTorchMLPClassifier import PyTorchMLPModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer
from pandas import DataFrame
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraPyTorchDataConvertor(PyTorchDataConvertor):
    target_tensor_type: torch.dtype

    def __init__(self, target_tensor_type: torch.dtype = torch.float32) -> None:
        self.target_tensor_type = target_tensor_type

    def convert_x(self, df: DataFrame, device: str) -> torch.Tensor:
        numpy_arrays = df.values
        x = torch.tensor(numpy_arrays, device=device, dtype=torch.float32).unsqueeze(1)
        return x

    def convert_y(self, df: DataFrame, device: str) -> torch.Tensor:
        numpy_arrays = df.values
        y = torch.tensor(numpy_arrays, device=device, dtype=self.target_tensor_type).squeeze(1)
        return y


class HydraClassifier(BasePyTorchClassifier):
    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return HydraPyTorchDataConvertor(target_tensor_type=torch.long)
        # return DefaultPyTorchDataConvertor(target_tensor_type=torch.long)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: Dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: Dict[str, Any] = config.get("trainer_kwargs", {})
        self.class_names = ["up", "down", "nan"]

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)

        X = torch.from_numpy(data_dictionary["train_features"].to_numpy()).float().unsqueeze(1)
        y = torch.from_numpy(data_dictionary["train_labels"].to_numpy()).squeeze(1)

        transform = HydraModel(input_length=X.shape[0])

        X_features = transform(X)

        scaler = SparseScaler()

        X_features_scaled = scaler.fit_transform(X_features)

        data_dictionary1 = {
            "train_features": DataFrame(X_features_scaled.numpy()),
            "test_features": DataFrame(),
            "train_labels": DataFrame(y.numpy()),
            "test_labels": np.array([0, 0], dtype=np.float32),
            "train_weights": data_dictionary["train_weights"],
            "test_weights": np.array([0, 0], dtype=np.float32),
            "train_dates": data_dictionary["train_dates"],
        }

        n_features = data_dictionary1["train_features"].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features, output_dim=len(class_names), **self.model_kwargs
        )
        model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                model_meta_data={"class_names": class_names},
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary1, self.splits)
        return trainer


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
        if self.divisor > 1:
            diff_X: torch.Tensor = torch.diff(X)
        Z = []
        for dilation_index in range(self.num_dilations):
            n_dilations = int(self.dilations[dilation_index].item())
            n_padding = int(self.paddings[dilation_index].item())
            for diff_index in range(self.divisor):
                _Z = F.conv1d(
                    X if diff_index == 0 else diff_X,
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


class SparseScaler:
    mask: bool
    exponent: int
    fitted: bool
    epsilon: torch.Tensor
    sigma: torch.Tensor
    mu: torch.Tensor

    def __init__(self, mask: bool = True, exponent: int = 4) -> None:
        self.mask = mask
        self.exponent = exponent
        self.fitted = False

    def fit(self, X: torch.Tensor) -> None:
        assert not self.fitted, "Already fitted."
        X = X.clamp(0).sqrt()
        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8
        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon
        self.fitted = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Not fitted."
        X = X.clamp(0).sqrt()
        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)
