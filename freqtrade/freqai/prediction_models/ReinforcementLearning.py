import logging
from typing import Any, Tuple, Dict
from freqtrade.freqai.prediction_models.RL.RLPrediction_env import GymAnytrading
from freqtrade.freqai.prediction_models.RL.RLPrediction_agent import RLPrediction_agent
from pandas import DataFrame
import pandas as pd
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import numpy.typing as npt
from freqtrade.freqai.freqai_interface import IFreqaiModel

logger = logging.getLogger(__name__)


class ReinforcementLearningModel(IFreqaiModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def train(
        self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_dataframe: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :returns:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info("--------------------Starting training " f"{pair} --------------------")

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_dataframe,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        data_dictionary: Dict[str, Any] = dk.make_train_test_datasets(
            features_filtered, labels_filtered)
        dk.fit_labels()  # useless for now, but just satiating append methods

        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}' " features"
        )
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        model = self.fit(data_dictionary, pair)

        if pair not in self.dd.historic_predictions:
            self.set_initial_historic_predictions(
                data_dictionary['train_features'], model, dk, pair)

        self.dd.save_historic_predictions_to_disk()

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    def fit(self, data_dictionary: Dict[str, Any], pair: str = ''):

        train_df = data_dictionary["train_features"]

        sep = '/'
        coin = pair.split(sep, 1)[0]
        price = train_df[f"%-{coin}raw_price_{self.config['timeframe']}"]
        price.reset_index(inplace=True, drop=True)

        model_name = 'ppo'

        env_instance = GymAnytrading(train_df, price, self.CONV_WIDTH)

        agent_params = self.freqai_info['model_training_parameters']
        total_timesteps = agent_params.get('total_timesteps', 1000)

        agent = RLPrediction_agent(env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(model=model,
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        print('Training finished!')

        return trained_model

    def predict(
        self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first: bool = False
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
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

        pred_df = self.rl_model_predict(dk.data_dictionary["prediction_features"], dk, self.model)
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def rl_model_predict(self, dataframe: DataFrame,
                         dk: FreqaiDataKitchen, model: Any) -> DataFrame:

        output = pd.DataFrame(np.full((len(dataframe), 1), 2), columns=dk.label_list)

        def _predict(window):
            observations = dataframe.iloc[window.index]
            res, _ = model.predict(observations, deterministic=True)
            return res

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def set_initial_historic_predictions(
        self, df: DataFrame, model: Any, dk: FreqaiDataKitchen, pair: str
    ) -> None:

        pred_df = self.rl_model_predict(df, dk, model)
        pred_df.fillna(0, inplace=True)
        self.dd.historic_predictions[pair] = pred_df
        hist_preds_df = self.dd.historic_predictions[pair]

        for label in hist_preds_df.columns:
            if hist_preds_df[label].dtype == object:
                continue
            hist_preds_df[f'{label}_mean'] = 0
            hist_preds_df[f'{label}_std'] = 0

        hist_preds_df['do_predict'] = 0

        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            hist_preds_df['DI_values'] = 0

        for return_str in dk.data['extra_returns_per_train']:
            hist_preds_df[return_str] = 0
