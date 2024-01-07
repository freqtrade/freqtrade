from freqtrade.freqai.base_models.BasePyTorchClassifier import BasePyTorchClassifier
from freqtrade.freqai.torch.HydraTrainer import HydraTrainer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    PyTorchDataConvertor,
    DefaultPyTorchDataConvertor,
)
from pandas import DataFrame
from typing import Any, Dict, Tuple
import numpy as np
import numpy.typing as npt
import torch


class HydraClassifier(BasePyTorchClassifier):
    trainer: HydraTrainer

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.long)

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

        self.trainer = self.get_init_model(dk.pair) or HydraTrainer(
            model_meta_data={"class_names": class_names},
            device=self.device,
            tb_logger=self.tb_logger,
            **self.trainer_kwargs,
        )

        self.trainer.fit(data_dictionary, self.splits)

        return self.trainer

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        class_names = self.get_class_names()

        if not self.class_name_to_index:
            self.init_class_names_to_index_mapping(class_names)

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = filtered_df

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        X: np.ndarray = dk.data_dictionary["prediction_features"].to_numpy()
        # y: np.ndarray = dk.data_dictionary["prediction_labels"].to_numpy()

        trainer = self.trainer
        predictions = trainer.predict(
            X,
            # y,
            trainer.transform,
            trainer.model,
            trainer.f_mean,
            trainer.f_std,
        )
        df_predictions = DataFrame(predictions, columns=[dk.label_list[0]])
        dk.do_predict = outliers
        return (df_predictions, dk.do_predict)


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
