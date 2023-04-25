import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from freqtrade.freqai.base_models.BasePyTorchModel import BasePyTorchModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class BasePyTorchRegressor(BasePyTorchModel):
    """
    A PyTorch implementation of a regressor.
    User must implement fit method
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.normalize_data_from_metadata(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        self.data_cleaning_predict(dk)
        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"],
            device=self.device
        )
        y = self.model.model(x)
        y = y.cpu()
        pred_df = DataFrame(y.detach().numpy(), columns=[dk.label_list[0]])
        return (pred_df, dk.do_predict)
