import logging
import platform
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import LSTM, Add, AlphaDropout, BatchNormalization, Dense, Dropout, Input
from keras.metrics import RootMeanSquaredError as rmse
from keras.optimizers import SGD
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class TensorFlowLSTMRegressor(BaseRegressionModel):
    """
    The following is an example of a custom LSTM model that uses TensorFlow and Keras.
    The class inherits from BaseRegressionModel, which means it has full access to all Frequency AI
    functionality. Typically, users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change various aspects of the
    training that cannot be configured via the top-level config.json file.
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
                # Adjust memory limit as needed for Windows systems
                # tf.config.set_logical_device_configuration(
                #     gpus[0],
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=7900)]
                # )
            elif system == "Linux":
                logger.info(f"Linux is detected. Running on GPU. {gpus}")
                # Adjust memory limit as needed for Linux systems
                # tf.config.set_logical_device_configuration(
                #     gpus[0],
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
                # )
            elif system == "Darwin":
                logger.info(
                    "MacOS is detected. If you're model isn't big enough to "
                    "benefit from GPU acceleration"
                    "consider uninstalling tensorflow-metal")
        else:
            logger.info("No GPU found. The model will run on CPU.")

        self.num_lstm_layers = config.get('num_lstm_layers', 3)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate: float = config.get("learning_rate", 0.0001)
        self.dropout_rate = config.get("dropout_rate", 0.3)
        self.timesteps = config.get("conv_width", 2)

    def create_sequences(self, data, labels, sequence_length):
        """
        Reshape data and labels into sequences for LSTM training.
        """
        sequence_data = []
        sequence_labels = []

        for i in range(len(data) - sequence_length):
            sequence_data.append(data[i:i + sequence_length])
            sequence_labels.append(labels[i + sequence_length])

        return np.array(sequence_data), np.array(sequence_labels)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """
        n_features = data_dictionary["train_features"].shape[1]
        n_output = data_dictionary["train_labels"].shape[1]

        # Robust Scaling of the features
        scaler = RobustScaler()
        train_X = scaler.fit_transform(data_dictionary["train_features"])
        train_y = data_dictionary["train_labels"].values

        # Reshape input to be 3D [samples, timestamps, features] using sequences
        sequence_length = self.timesteps  # set your desired sequence length
        train_X, train_y = self.create_sequences(train_X, train_y, sequence_length)

        # If a test set exists, transform and reshape it similarly
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            test_X = scaler.transform(data_dictionary["test_features"])
            test_y = data_dictionary["test_labels"].values
            test_X, test_y = self.create_sequences(test_X, test_y, sequence_length)
        else:
            test_X, test_y = None, None

        # Designing the model
        # Note: The input shape now takes sequence_length as the first dimension
        input_layer = Input(shape=(sequence_length, n_features))
        x = LSTM(100, return_sequences=True, recurrent_regularizer='l2')(
            input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x_res = x  # save for residual connection
        for _ in range(self.num_lstm_layers - 1):
            x = LSTM(100, return_sequences=True, recurrent_regularizer='l2')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            x = Add()([x, x_res])  # residual connection
            x_res = x
        x = LSTM(100, return_sequences=False, recurrent_regularizer='l2')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(units=36, activation='relu')(x)
        x = AlphaDropout(0.5)(x)
        output_layer = Dense(units=n_output)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9,
                        nesterov=True, clipvalue=0.5,
                        weight_decay=0.0001)
        model.compile(optimizer=optimizer, loss='mse', metrics=[rmse()])

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
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        sequence_length = self.timesteps
        num_rows = dk.data_dictionary["prediction_features"].shape[0]

        # Create input tensor with all windows
        input_data = np.array([dk.data_dictionary["prediction_features"][i:i + sequence_length]
                               for i in range(num_rows - sequence_length)])

        # Make predictions in a single batch
        predictions = self.model.predict(input_data)

        pred_df = DataFrame(predictions, columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        # Add rows of zeros at the beginning to adjust for the window size
        pred_df = pd.concat(
            [pd.DataFrame(np.zeros((sequence_length, len(pred_df.columns))),
                          columns=pred_df.columns),
             pred_df], axis=0).reset_index(drop=True)

        return (pred_df, dk.do_predict)
