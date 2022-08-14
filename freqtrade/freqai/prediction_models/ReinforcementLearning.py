import logging
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from pandas import DataFrame
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.prediction_models.RL.RLPrediction_agent_TDQN import TDQN
from freqtrade.freqai.prediction_models.RL.RLPrediction_env_TDQN_5ac import DEnv
#from freqtrade.freqai.prediction_models.RL.RLPrediction_env_TDQN_3ac import DEnv
from freqtrade.persistence import Trade

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

        # train_df = data_dictionary["train_features"]
        # # train_labels = data_dictionary["train_labels"]
        # test_df = data_dictionary["test_features"]
        # # test_labels = data_dictionary["test_labels"]
        # # sep = '/'
        # # coin = pair.split(sep, 1)[0]
        # # price = train_df[f"%-{coin}raw_price_{self.config['timeframe']}"]
        # # price.reset_index(inplace=True, drop=True)
        # # price = price.to_frame()
        # price = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(train_df.index))
        # price_test = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(test_df.index))
        # #train_env = GymAnytrading(train_df, price, self.CONV_WIDTH)
        # agent_params = self.freqai_info['model_training_parameters']
        # reward_params = self.freqai_info['model_reward_parameters']
        # train_env = DEnv(df=train_df, prices=price, window_size=self.CONV_WIDTH, reward_kwargs=reward_params)
        # #eval_env = DEnv(df=test_df, prices=price_test, window_size=self.CONV_WIDTH, reward_kwargs=reward_params)
        # #env_instance = SubprocVecEnv([DEnv(df=train_df, prices=price, window_size=self.CONV_WIDTH, reward_kwargs=reward_params)])
        # #train_env.reset()
        # #eval_env.reset()
        # # model
        # #policy_kwargs = dict(net_arch=[512, 512, 512])
        # policy_kwargs = dict(activation_fn=th.nn.Tanh,
        #              net_arch=[256, 256, 256])
        # agent = RLPrediction_agent(train_env)
        # #eval_agent = RLPrediction_agent(eval_env)

        # # PPO
        # model_name = 'ppo'
        # model = agent.get_model(model_name, model_kwargs=agent_params, policy_kwargs=policy_kwargs)
        # trained_model = agent.train_model(model=model,
        #                                   tb_log_name=model_name,
        #                                   model_kwargs=agent_params,
        #                                   train_df=train_df,
        #                                   test_df=test_df,
        #                                   price=price,
        #                                   price_test=price_test,
        #                                   window_size=self.CONV_WIDTH)
        # # best_model = eval_agent.train_model(model=model,
        # #                                   tb_log_name=model_name,
        # #                                   model_kwargs=agent_params,
        # #                                   eval=eval_env)
        # # TDQN
        # # model_name = 'TDQN'
        # # model = TDQN('TMultiInputPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log='./tensorboard_log/',
        # #             learning_rate=agent_params["learning_rate"], gamma=0.9,
        # #             target_update_interval=5000, buffer_size=50000,
        # #             exploration_initial_eps=1, exploration_final_eps=0.1,
        # #             replay_buffer_class=ReplayBuffer
        # #            )
        # # trained_model = agent.train_model(model=model,
        # #                                   tb_log_name=model_name,
        # #                                   model_kwargs=agent_params)
        # #model.learn(
        # #     total_timesteps=5000,
        # #     callback=callback
        # # )

        agent_params = self.freqai_info['model_training_parameters']
        reward_params = self.freqai_info['model_reward_parameters']
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = agent_params["eval_cycles"] * len(test_df)
        total_timesteps = agent_params["train_cycles"] * len(train_df)

        # price data for model training and evaluation
        price = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(train_df.index))
        price_test = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(test_df.index))

        # environments
        train_env = DEnv(df=train_df, prices=price, window_size=self.CONV_WIDTH, reward_kwargs=reward_params)
        eval = DEnv(df=test_df, prices=price_test, window_size=self.CONV_WIDTH, reward_kwargs=reward_params)
        eval_env = Monitor(eval, ".")
        eval_env.reset()

        # this should be in config - TODO
        agent_type = 'tdqn'

        path = self.dk.data_path
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{path}/",
                             log_path=f"{path}/{agent_type}/logs/", eval_freq=int(eval_freq),
                             deterministic=True, render=False)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                      net_arch=[256, 256, 128])

        if agent_type == 'tdqn':
            model = TDQN('TMultiInputPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log=f"{path}/{agent_type}/tensorboard/",
                    learning_rate=0.00025, gamma=0.9,
                    target_update_interval=5000, buffer_size=50000,
                    exploration_initial_eps=1, exploration_final_eps=0.1,
                    replay_buffer_class=ReplayBuffer
                   )
        elif agent_type == 'ppo':
            model = PPO('MultiInputPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log=f"{path}/{agent_type}/tensorboard/",
                learning_rate=0.00025, gamma=0.9
            )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        print('Training finished!')

        return model



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
                current_profit = current_value / openrate -1

        total_profit = 0
        closed_trades = Trade.get_trades(trade_filter=[Trade.is_open.is_(False), Trade.pair == pair])
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
