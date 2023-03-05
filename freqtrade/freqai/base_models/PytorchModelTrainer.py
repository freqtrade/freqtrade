import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PytorchModelTrainer:
    def __init__(self, model: nn.Module, optimizer, init_model: Dict):
        self.model = model
        self.optimizer = optimizer
        if init_model:
            self.load_from_checkpoint(init_model)

    def fit(self, tensor_dictionary, max_iters, batch_size):
        for iter in range(max_iters):

            # todo add validation evaluation here

            xb, yb = self.get_batch(tensor_dictionary, 'train', batch_size)
            logits, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_from_file(self, path: Path):
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self

    @staticmethod
    def get_batch(tensor_dictionary: Dict, split: str, batch_size: int):
        ix = torch.randint(len(tensor_dictionary[f'{split}_labels']), (batch_size,))
        x = tensor_dictionary[f'{split}_features'][ix]
        y = tensor_dictionary[f'{split}_labels'][ix]
        return x, y

