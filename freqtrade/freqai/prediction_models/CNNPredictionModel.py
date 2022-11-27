import logging
from typing import Any, Dict, Tuple

from pandas import DataFrame
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import tensorflow as tf
from freqtrade.freqai.base_models.BaseTensorFlowModel import BaseTensorFlowModel, WindowGenerator
from tensorflow.keras.layers import Input, Conv1D, Dense
from tensorflow.keras.models import Model
import numpy as np

logger = logging.getLogger(__name__)

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

MAX_EPOCHS = 10


class CNNPredictionModel(BaseTensorFlowModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), fit().
    """

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """
        train_df = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        test_df = data_dictionary["test_features"]
        test_labels = data_dictionary["test_labels"]
        n_labels = len(train_labels.columns)

        if n_labels > 1:
            raise OperationalException(
                "Neural Net not yet configured for multi-targets. Please "
                " reduce number of targets to 1 in strategy."
            )

        n_features = len(data_dictionary["train_features"].columns)
        BATCH_SIZE = self.freqai_info.get("batch_size", 64)
        input_dims = [BATCH_SIZE, self.CONV_WIDTH, n_features]

        w1 = WindowGenerator(
            input_width=self.CONV_WIDTH,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=test_df,
            train_labels=train_labels,
            val_labels=test_labels,
            batch_size=BATCH_SIZE,
        )

        model = self.create_model(input_dims, n_labels)

        steps_per_epoch = np.ceil(len(test_df) / BATCH_SIZE)
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001, decay_steps=steps_per_epoch * 1000, decay_rate=1, staircase=False
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, mode="min", min_delta=0.0001
        )

        model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(lr_schedule),
            metrics=[tf.metrics.MeanAbsoluteError()],
        )

        model.fit(
            w1.train,
            epochs=MAX_EPOCHS,
            shuffle=False,
            validation_data=w1.val,
            callbacks=[early_stopping],
            verbose=1,
        )

        return model

    def predict(
        self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first=True
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :predictions: np.array of predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_dataframe)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_dataframe, dk.training_features_list, training_filter=False
        )
        filtered_dataframe = dk.normalize_data_from_metadata(filtered_dataframe)
        dk.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk, filtered_dataframe)

        if first:
            full_df = dk.data_dictionary["prediction_features"]

            w1 = WindowGenerator(
                input_width=self.CONV_WIDTH,
                label_width=1,
                shift=1,
                test_df=full_df,
                batch_size=len(full_df),
            )

            predictions = self.model.predict(w1.inference)
            len_diff = len(dk.do_predict) - len(predictions)
            if len_diff > 0:
                dk.do_predict = dk.do_predict[len_diff:]

        else:
            data = dk.data_dictionary["prediction_features"]
            data = tf.expand_dims(data, axis=0)
            predictions = self.model(data, training=False)

        predictions = predictions[:, 0, 0]
        pred_df = DataFrame(predictions, columns=dk.label_list)

        pred_df = dk.denormalize_labels_from_metadata(pred_df)

        return (pred_df, np.ones(len(pred_df)))

    def create_model(self, input_dims, n_labels) -> Any:

        input_layer = Input(shape=(input_dims[1], input_dims[2]))
        Layer_1 = Conv1D(filters=32, kernel_size=(self.CONV_WIDTH,), activation="relu")(input_layer)
        Layer_3 = Dense(units=32, activation="relu")(Layer_1)
        output_layer = Dense(units=n_labels)(Layer_3)
        return Model(inputs=input_layer, outputs=output_layer)
