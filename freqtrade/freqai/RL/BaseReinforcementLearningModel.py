import logging
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from abc import abstractmethod
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base3ActionRLEnv import Base3ActionRLEnv, Actions, Positions
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BaseReinforcementLearningModel(IFreqaiModel):
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

        model = self.fit_rl(data_dictionary, pair, dk)

        if pair not in self.dd.historic_predictions:
            self.set_initial_historic_predictions(
                data_dictionary['train_features'], model, dk, pair)

        self.dd.save_historic_predictions_to_disk()

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    @abstractmethod
    def fit_rl(self, data_dictionary: Dict[str, Any], pair: str, dk: FreqaiDataKitchen):
        """
        Agent customizations and abstract Reinforcement Learning customizations
        go in here. Abstract method, so this function must be overridden by
        user class.
        """

        return

    def get_state_info(self, pair):
        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))
        market_side = 0.5
        current_profit = 0
        for trade in open_trades:
            if trade.pair == pair:
                current_value = trade.open_trade_value
                openrate = trade.open_rate
                if 'long' in trade.enter_tag:
                    market_side = 1
                else:
                    market_side = 0
                current_profit = current_value / openrate - 1

        total_profit = 0
        closed_trades = Trade.get_trades(
            trade_filter=[Trade.is_open.is_(False), Trade.pair == pair])
        for trade in closed_trades:
            total_profit += trade.close_profit

        return market_side, current_profit, total_profit

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

    # TODO take care of this appendage. Right now it needs to be called because FreqAI enforces it.
    # But FreqaiRL needs more objects passed to fit() (like DK) and we dont want to go refactor
    # all the other existing fit() functions to include dk argument. For now we instantiate and
    # leave it.
    def fit(self, data_dictionary: Dict[str, Any], pair: str = '') -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :param data_dictionary: Dict = the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        return


class MyRLEnv(Base3ActionRLEnv):

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self.update_portfolio_log_returns(action)

        self._update_profit(action)
        step_reward = self._calculate_reward(action)
        self.total_reward += step_reward

        trade_type = None
        if self.is_tradesignal(action):  # exclude 3 case not trade
            # Update position
            """
            Action: Neutral, position: Long ->  Close Long
            Action: Neutral, position: Short -> Close Short

            Action: Long, position: Neutral -> Open Long
            Action: Long, position: Short -> Close Short and Open Long

            Action: Short, position: Neutral -> Open Short
            Action: Short, position: Long -> Close Long and Open Short
            """

            if action == Actions.Neutral.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
            elif action == Actions.Long.value:
                self._position = Positions.Long
                trade_type = "long"
            elif action == Actions.Short.value:
                self._position = Positions.Short
                trade_type = "short"
            else:
                print("case not defined")

            # Update last trade tick
            self._last_trade_tick = self._current_tick

            if trade_type is not None:
                self.trade_history.append(
                    {'price': self.current_price(), 'index': self._current_tick,
                     'type': trade_type})

        if self._total_profit < 0.2:
            self._done = True

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            tick=self._current_tick,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info
