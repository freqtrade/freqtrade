import logging
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from abc import abstractmethod
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnv import Base5ActionRLEnv, Actions, Positions
from freqtrade.persistence import Trade
import torch.multiprocessing
from stable_baselines3.common.monitor import Monitor
import torch as th
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')


class BaseReinforcementLearningModel(IFreqaiModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs['config'])
        th.set_num_threads(self.freqai_info.get('data_kitchen_thread_count', 4))
        self.reward_params = self.freqai_info['rl_config']['model_reward_parameters']
        self.train_env: Base5ActionRLEnv = None

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
        prices_train, prices_test = self.build_ohlc_price_dataframes(dk.data_dictionary, pair, dk)
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}' " features"
        )
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        self.set_train_and_eval_environments(data_dictionary, prices_train, prices_test)

        model = self.fit_rl(data_dictionary, dk)

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    def set_train_and_eval_environments(self, data_dictionary, prices_train, prices_test):
        """
        User overrides this in their prediction model if they are custom a MyRLEnv. Othwerwise
        leaving this will default to Base5ActEnv
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # environments
        if not self.train_env:
            self.train_env = MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                     reward_kwargs=self.reward_params)
            self.eval_env = Monitor(MyRLEnv(df=test_df, prices=prices_test,
                                    window_size=self.CONV_WIDTH,
                                    reward_kwargs=self.reward_params), ".")
        else:
            self.train_env.reset_env(train_df, prices_train, self.CONV_WIDTH, self.reward_params)
            self.eval_env.reset_env(train_df, prices_train, self.CONV_WIDTH, self.reward_params)
            self.train_env.reset()
            self.eval_env.reset()

    @abstractmethod
    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):
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

        pred_df = self.rl_model_predict(
            dk.data_dictionary["prediction_features"], dk, self.model)
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def rl_model_predict(self, dataframe: DataFrame,
                         dk: FreqaiDataKitchen, model: Any) -> DataFrame:

        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            market_side, current_profit, total_profit = self.get_state_info(dk.pair)
            observations = dataframe.iloc[window.index]
            observations['current_profit'] = current_profit
            observations['position'] = market_side
            res, _ = model.predict(observations, deterministic=True)
            return res

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def build_ohlc_price_dataframes(self, data_dictionary: dict,
                                    pair: str, dk: FreqaiDataKitchen) -> Tuple[DataFrame,
                                                                               DataFrame]:
        """
        Builds the train prices and test prices for the environment.
        """

        coin = pair.split('/')[0]
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # price data for model training and evaluation
        tf = self.config['timeframe']
        ohlc_list = [f'%-{coin}raw_open_{tf}', f'%-{coin}raw_low_{tf}',
                     f'%-{coin}raw_high_{tf}', f'%-{coin}raw_close_{tf}']
        rename_dict = {f'%-{coin}raw_open_{tf}': 'open', f'%-{coin}raw_low_{tf}': 'low',
                       f'%-{coin}raw_high_{tf}': ' high', f'%-{coin}raw_close_{tf}': 'close'}

        prices_train = train_df.filter(ohlc_list, axis=1)
        prices_train.rename(columns=rename_dict, inplace=True)
        prices_train.reset_index(drop=True)

        prices_test = test_df.filter(ohlc_list, axis=1)
        prices_test.rename(columns=rename_dict, inplace=True)
        prices_test.reset_index(drop=True)

        return prices_train, prices_test

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


class MyRLEnv(Base5ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    Adds 5 actions.
    """

    def calculate_reward(self, action):

        if self._last_trade_tick is None:
            return 0.

        # close long
        if action == Actions.Long_sell.value and self._position == Positions.Long:
            last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
            return float(np.log(current_price) - np.log(last_trade_price))

        if action == Actions.Long_sell.value and self._position == Positions.Long:
            if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
                current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
                return float((np.log(current_price) - np.log(last_trade_price)) * 2)

        # close short
        if action == Actions.Short_buy.value and self._position == Positions.Short:
            last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
            return float(np.log(last_trade_price) - np.log(current_price))

        if action == Actions.Short_buy.value and self._position == Positions.Short:
            if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
                current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
                return float((np.log(last_trade_price) - np.log(current_price)) * 2)

        return 0.
