import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


class PyTorchModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            device: str,
            init_model: Dict,
            target_tensor_type: torch.dtype,
            model_meta_data: Dict[str, Any] = {},
            **kwargs
    ):
        """
        :param model: The PyTorch model to be trained.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        :param init_model: A dictionary containing the initial model/optimizer
            state_dict and model_meta_data saved by self.save() method.
        :param target_tensor_type: type of target tensor, for classification usually
            torch.long, for regressor usually torch.float.
        :param model_meta_data: Additional metadata about the model (optional).
        :param max_iters: The number of training iterations to run.
            iteration here refers to the number of times we call
            self.optimizer.step(). used to calculate n_epochs.
        :param batch_size: The size of the batches to use during training.
        :param max_n_eval_batches: The maximum number batches to use for evaluation.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.target_tensor_type = target_tensor_type
        self.max_iters: int = kwargs.get("max_iters", 100)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.max_n_eval_batches: Optional[int] = kwargs.get("max_n_eval_batches", None)
        if init_model:
            self.load_from_checkpoint(init_model)

    def fit(self, data_dictionary: Dict[str, pd.DataFrame]):
        """
         - Calculates the predicted output for the batch using the PyTorch model.
         - Calculates the loss between the predicted and actual output using a loss function.
         - Computes the gradients of the loss with respect to the model's parameters using
           backpropagation.
         - Updates the model's parameters using an optimizer.
        """
        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary)
        epochs = self.calc_n_epochs(
            n_obs=len(data_dictionary["train_features"]),
            batch_size=self.batch_size,
            n_iters=self.max_iters
        )
        for epoch in range(epochs):
            # training
            losses = []
            for i, batch_data in enumerate(data_loaders_dictionary["train"]):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            train_loss = sum(losses) / len(losses)

            # evaluation
            test_loss = self.estimate_loss(data_loaders_dictionary, self.max_n_eval_batches, "test")
            logger.info(
                f"epoch {epoch}/{epochs}:"
                f" train loss {train_loss:.4f} ; test loss {test_loss:.4f}"
            )

    @torch.no_grad()
    def estimate_loss(
            self,
            data_loader_dictionary: Dict[str, DataLoader],
            max_n_eval_batches: Optional[int],
            split: str,
    ) -> float:
        self.model.eval()
        n_batches = 0
        losses = []
        for i, batch in enumerate(data_loader_dictionary[split]):
            if max_n_eval_batches and i > max_n_eval_batches:
                n_batches += 1
                break

            xb, yb = batch
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            losses.append(loss.item())

        self.model.train()
        return sum(losses) / len(losses)

    def create_data_loaders_dictionary(
            self,
            data_dictionary: Dict[str, pd.DataFrame]
    ) -> Dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        """
        data_loader_dictionary = {}
        for split in ["train", "test"]:
            labels_shape = data_dictionary[f"{split}_labels"].shape
            labels_view = (labels_shape[0], 1) if labels_shape[1] == 1 else labels_shape
            dataset = TensorDataset(
                torch.from_numpy(data_dictionary[f"{split}_features"].values).float(),
                torch.from_numpy(data_dictionary[f"{split}_labels"].values)
                .to(self.target_tensor_type)
                .view(labels_view)
            )

            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary

    @staticmethod
    def calc_n_epochs(n_obs: int, batch_size: int, n_iters: int) -> int:
        """
        Calculates the number of epochs required to reach the maximum number
        of iterations specified in the model training parameters.

        the motivation here is that `max_iters` is easier to optimize and keep stable,
        across different n_obs - the number of data points.
        """

        n_batches = math.ceil(n_obs // batch_size)
        epochs = math.ceil(n_iters // n_batches)
        return epochs

    def save(self, path: Path):
        """
        - Saving any nn.Module state_dict
        - Saving model_meta_data, this dict should contain any additional data that the
          user needs to store. e.g class_names for classification models.
        """

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_meta_data": self.model_meta_data,
        }, path)

    def load_from_file(self, path: Path):
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict):
        """
        when using continual_learning, DataDrawer will load the dictionary
        (containing state dicts and model_meta_data) by calling torch.load(path).
        you can access this dict from any class that inherits IFreqaiModel by calling
        get_init_model method.
        """

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint["model_meta_data"]
        return self
