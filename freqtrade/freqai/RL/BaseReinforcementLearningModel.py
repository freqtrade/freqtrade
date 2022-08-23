import logging
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from abc import abstractmethod
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnv import Base5ActionRLEnv, Actions, Positions
from freqtrade.persistence import Trade
import torch.multiprocessing
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
from typing import Callable
from datetime import datetime, timezone
from stable_baselines3.common.utils import set_random_seed
import gym
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')

SB3_MODELS = ['PPO', 'A2C', 'DQN', 'TD3', 'SAC']
SB3_CONTRIB_MODELS = ['TRPO', 'ARS']


class BaseReinforcementLearningModel(IFreqaiModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs['config'])
        th.set_num_threads(self.freqai_info['rl_config'].get('thread_count', 4))
        self.reward_params = self.freqai_info['rl_config']['model_reward_parameters']
        self.train_env: Base5ActionRLEnv = None
        self.eval_env: Base5ActionRLEnv = None
        self.eval_callback: EvalCallback = None
        self.model_type = self.freqai_info['rl_config']['model_type']
        if self.model_type in SB3_MODELS:
            import_str = 'stable_baselines3'
        elif self.model_type in SB3_CONTRIB_MODELS:
            import_str = 'sb3_contrib'
        else:
            raise OperationalException(f'{self.model_type} not available in stable_baselines3 or '
                                       f'sb3_contrib. please choose one of {SB3_MODELS} or '
                                       f'{SB3_CONTRIB_MODELS}')

        mod = __import__(import_str, fromlist=[
                         self.model_type])
        self.MODELCLASS = getattr(mod, self.model_type)
        self.policy_type = self.freqai_info['rl_config']['policy_type']

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

        self.set_train_and_eval_environments(data_dictionary, prices_train, prices_test, dk)

        model = self.fit_rl(data_dictionary, dk)

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    def set_train_and_eval_environments(self, data_dictionary, prices_train, prices_test, dk):
        """
        User overrides this as shown here if they are using a custom MyRLEnv
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = self.freqai_info["rl_config"]["eval_cycles"] * len(test_df)

        # environments
        if not self.train_env:
            self.train_env = MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                     reward_kwargs=self.reward_params, config=self.config)
            self.eval_env = Monitor(MyRLEnv(df=test_df, prices=prices_test,
                                    window_size=self.CONV_WIDTH,
                                    reward_kwargs=self.reward_params, config=self.config), ".")
            self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                              render=False, eval_freq=eval_freq,
                                              best_model_save_path=dk.data_path)
        else:
            self.train_env.reset()
            self.eval_env.reset()
            self.train_env.reset_env(train_df, prices_train, self.CONV_WIDTH, self.reward_params)
            self.eval_env.reset_env(test_df, prices_test, self.CONV_WIDTH, self.reward_params)
            self.eval_callback.__init__(self.eval_env, deterministic=True,
                                        render=False, eval_freq=eval_freq,
                                        best_model_save_path=dk.data_path)

    @abstractmethod
    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):
        """
        Agent customizations and abstract Reinforcement Learning customizations
        go in here. Abstract method, so this function must be overridden by
        user class.
        """

        return

    def get_state_info(self, pair):
        open_trades = Trade.get_trades_proxy(is_open=True)
        market_side = 0.5
        current_profit = 0
        trade_duration = 0
        for trade in open_trades:
            if trade.pair == pair:
                current_value = self.strategy.dp._exchange.get_rate(
                    pair, refresh=False, side="exit", is_short=trade.is_short)
                openrate = trade.open_rate
                now = datetime.now(timezone.utc).timestamp()
                trade_duration = (now - trade.open_date.timestamp()) / self.base_tf_seconds
                if 'long' in trade.enter_tag:
                    market_side = 1
                    current_profit = (current_value - openrate) / openrate
                else:
                    market_side = 0
                    current_profit = (openrate - current_value) / openrate

        # total_profit = 0
        # closed_trades = Trade.get_trades_proxy(pair=pair, is_open=False)
        # for trade in closed_trades:
        #     total_profit += trade.close_profit

        return market_side, current_profit, int(trade_duration)

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
            market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
            observations = dataframe.iloc[window.index]
            observations['current_profit'] = current_profit
            observations['position'] = market_side
            observations['trade_duration'] = trade_duration
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
        return


def make_env(env_id: str, rank: int, seed: int, train_df, price,
             reward_params, window_size, monitor=False, config={}) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:

        env = MyRLEnv(df=train_df, prices=price, window_size=window_size,
                      reward_kwargs=reward_params, id=env_id, seed=seed + rank, config=config)
        if monitor:
            env = Monitor(env, ".")
        return env
    set_random_seed(seed)
    return _init


class MyRLEnv(Base5ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    Adds 5 actions.
    """

    def calculate_reward(self, action):

        if self._last_trade_tick is None:
            return 0.

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            factor = 1
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor = self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float((np.log(current_price) - np.log(last_trade_price)) * factor)

        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            factor = 1
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor = self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(np.log(last_trade_price) - np.log(current_price) * factor)

        return 0.
