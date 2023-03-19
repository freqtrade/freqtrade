from typing import Any, Dict

from freqtrade.freqai.base_models.PyTorchModelTrainer import PyTorchModelTrainer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.PyTorchClassifierClassifier import PyTorchClassifier
from freqtrade.freqai.prediction_models.PyTorchMLPModel import PyTorchMLPModel

import torch


class MLPPyTorchClassifier(PyTorchClassifier):
    """
    This class implements the fit method of IFreqaiModel.
    int the fit method we initialize the model and trainer objects.
    the only requirement from the model is to be aligned to PyTorchClassifier
    predict method that expects the model to predict tensor of type long.
    the trainer defines the training loop.

    parameters are passed via `model_training_parameters` under the freqai
    section in the config file. e.g:
    {
        ...
        "freqai": {
            ...
            "model_training_parameters" : {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "max_iters": 5000,
                    "batch_size": 64,
                    "max_n_eval_batches": None,
                },
                "model_kwargs": {
                    "hidden_dim": 512,
                    "dropout_percent": 0.2,
                    "n_layer": 1,
                },
            }
        }
    }


    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_training_params = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = model_training_params.get("learning_rate", 3e-4)
        self.model_kwargs: Dict[str, any] = model_training_params.get("model_kwargs", {})
        self.trainer_kwargs: Dict[str, any] = model_training_params.get("trainer_kwargs", {})

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        :raises ValueError: If self.class_names is not defined in the parent class.
        """

        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)
        n_features = data_dictionary["train_features"].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features,
            output_dim=len(class_names),
            **self.model_kwargs
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        init_model = self.get_init_model(dk.pair)
        trainer = PyTorchModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            model_meta_data={"class_names": class_names},
            device=self.device,
            init_model=init_model,
            **self.trainer_kwargs,
        )
        trainer.fit(data_dictionary)
        return trainer
