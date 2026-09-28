from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from freqtrade.freqai.base_models.BasePyTorchClassifier import BasePyTorchClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchTransformerTrainer
from freqtrade.freqai.torch.PyTorchTransformerModel import PyTorchTransformerModel


class PyTorchTransformerClassifier(BasePyTorchClassifier):

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(
            target_tensor_type=torch.long, squeeze_target_tensor=True
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)
        n_features = data_dictionary["train_features"].shape[-1]
        n_labels = len(class_names)

        model = PyTorchTransformerModel(
            input_dim=n_features,
            output_dim=n_labels,
            time_window=self.window_size,
            **self.model_kwargs,
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # check if continual_learning is activated, and retrieve the model to continue training
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchTransformerTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                window_size=self.window_size,
                tb_logger=self.tb_logger,
                model_meta_data={"class_names": class_names},
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[pd.DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        class_names = self.model.model_meta_data.get("class_names", None)
        if not class_names:
            raise ValueError(
                "Missing class names. self.model.model_meta_data['class_names'] is None."
            )

        if not self.class_name_to_index:
            self.init_class_names_to_index_mapping(class_names)

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"], device=self.device
        )
        # if user is asking for multiple predictions, slide the window
        # along the tensor
        x = x.unsqueeze(0)
        # create empty torch tensor
        self.model.model.eval()
        yb = torch.empty(0).to(self.device)
        probs = torch.empty(0, len(class_names)).to(self.device)
        if x.shape[1] > self.window_size:
            ws = self.window_size
            for i in range(0, x.shape[1] - ws):
                xb = x[:, i : i + ws, :].to(self.device)
                logits = self.model.model(xb)
                _probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = torch.cat((probs, _probs), dim=0)
                y = torch.argmax(_probs, dim=-1)
                yb = torch.cat((yb, y), dim=-1)
        else:
            logits = self.model.model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            yb = torch.argmax(probs, dim=-1)

        predicted_classes_str = self.decode_class_names(yb)
        pred_df_prob = pd.DataFrame(probs.detach().tolist(), columns=class_names)

        pred_df = pd.DataFrame(predicted_classes_str, columns=[dk.label_list[0]])
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        if x.shape[1] > len(pred_df):
            # Add zeros to the prediction dataframe to match the input dataframe length
            # Add empty str to the prediction dataframe to match the input dataframe length

            double_type_columns = pred_df.select_dtypes(include=["float64", "int64"]).columns
            str_stype_columns = pred_df.select_dtypes(include=["object"]).columns
            zeros_df = pd.DataFrame(
                np.zeros((x.shape[1] - len(pred_df), len(double_type_columns))),
                columns=double_type_columns,
            )

            empty_str_df = pd.DataFrame(
                np.full((x.shape[1] - len(pred_df), len(str_stype_columns)), ""),
                columns=str_stype_columns,
            )

            # merge zeros_df and empty_str_df by columns
            zeros_df = pd.concat([zeros_df, empty_str_df], axis=1)
            pred_df = pd.concat([zeros_df, pred_df], axis=0, ignore_index=True)

        return (pred_df, dk.do_predict)
