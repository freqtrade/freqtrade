import logging
import platform
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import AlphaDropout, BatchNormalization, Dense, Dropout, Input
from keras.optimizers import SGD
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, RobustScaler

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class TensorFlowClassifier(BaseClassifierModel):
    """
    A custom classifier model that uses TensorFlow and Keras.
    The class inherits from BaseClassifierModel to leverage Frequency AI functionality.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        config = self.freqai_info.get("model_training_parameters", {})

        # Determine current operating system
        system = platform.system()

        # Set GPU configuration based on OS
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if system == "Windows":
                logger.info(f"Windows is detected. Running on GPU. {gpus}")
            elif system == "Linux":
                logger.info(f"Linux is detected. Running on GPU. {gpus}")
            elif system == "Darwin":
                logger.info(
                    "MacOS is detected. If your model isn't big enough to "
                    "benefit from GPU acceleration, "
                    "consider uninstalling tensorflow-metal")
        else:
            logger.info("No GPU found. The model will run on CPU.")

        self.num_layers = config.get('num_layers', 3)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate: float = config.get("learning_rate", 0.0001)
        self.dropout_rate = config.get("dropout_rate", 0.3)
        self.label_encoder: Optional[LabelEncoder] = None

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Train the classifier model.
        :param data_dictionary: the dictionary holding all data for train, test, labels, weights
        :param dk: The datakitchen object for the current coin/model
        """
        n_features = data_dictionary["train_features"].shape[1]

        # Robust Scaling of the features
        scaler = RobustScaler()
        train_X = scaler.fit_transform(data_dictionary["train_features"])
        train_y = data_dictionary["train_labels"]

        # Convert labels to integers
        self.label_encoder = LabelEncoder()
        train_y = self.label_encoder.fit_transform(train_y)
        n_classes = len(self.label_encoder.classes_)

        # If a test set exists, transform it similarly
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            test_X = scaler.transform(data_dictionary["test_features"])
            test_y = data_dictionary["test_labels"]
            test_y = self.label_encoder.transform(test_y)
        else:
            test_X, test_y = None, None

        # Designing the model
        input_layer = Input(shape=(n_features,))
        x = Dense(units=100, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x_res = x  # save for residual connection
        for _ in range(self.num_layers - 1):
            x = Dense(units=100, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            x = tf.keras.layers.Add()([x, x_res])  # residual connection
            x_res = x
        x = Dense(units=36, activation='relu')(x)
        x = AlphaDropout(0.5)(x)
        output_layer = Dense(units=n_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9,
                        nesterov=True, clipvalue=0.5,
                        weight_decay=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        tensorboard_callback = TensorBoard(log_dir=dk.data_path, histogram_freq=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit network
        model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size,
                  validation_data=(test_X, test_y), verbose=2, shuffle=False,
                  callbacks=[early_stopping, lr_scheduler, tensorboard_callback])

        return model

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs) \
            -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Predict using the trained classifier model.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :param dk: The datakitchen object for the current coin/model
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = filtered_df

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        predictions_prob = self.model.predict(dk.data_dictionary["prediction_features"])

        if self.label_encoder is None:
            raise ValueError("Label encoder is not initialized."
                             " Make sure to call `fit` before `predict`.")

        predictions = self.label_encoder.inverse_transform(predictions_prob.argmax(axis=1))
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))
            predictions_prob = np.reshape(predictions_prob, (-1, len(self.label_encoder.classes_)))

        pred_df = DataFrame(predictions, columns=dk.label_list)

        if self.label_encoder is None:
            raise ValueError("Label encoder is not initialized. "
                             "Make sure to call `fit` before `predict`.")
        pred_df_prob = DataFrame(predictions_prob, columns=self.label_encoder.classes_)
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return (pred_df, dk.do_predict)
