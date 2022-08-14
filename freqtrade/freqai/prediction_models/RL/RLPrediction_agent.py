# common library

import gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, CheckpointCallback,
                                                EvalCallback, StopTrainingOnRewardThreshold)
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from freqtrade.freqai.prediction_models.RL import config
#from freqtrade.freqai.prediction_models.RL.RLPrediction_agent_v2 import TDQN
from freqtrade.freqai.prediction_models.RL.RLPrediction_env import DEnv


# from stable_baselines3.common.vec_env import DummyVecEnv

# from meta.env_stock_trading.env_stock_trading import StockTradingEnv

# RL models from stable-baselines


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}


MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class RLPrediction_agent:
    """Provides implementations for DRL algorithms
    Based on:
    https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/agents/stablebaselines3_models.py
    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        reward_kwargs=None,
        #total_timesteps=None,
        verbose=1,
        seed=None
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            #model_kwargs=model_kwargs,
            #total_timesteps=model_kwargs["total_timesteps"],
            seed=seed
            #**model_kwargs,
        )




        return model

    def train_model(self, model, tb_log_name, model_kwargs, train_df, test_df, price, price_test, window_size):


        agent_params = self.freqai_info['model_training_parameters']
        reward_params = self.freqai_info['model_reward_parameters']
        train_env = DEnv(df=train_df, prices=price, window_size=window_size, reward_kwargs=reward_params)
        eval_env = DEnv(df=test_df, prices=price_test, window_size=window_size, reward_kwargs=reward_params)

        # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
        #         name_prefix='rl_model')

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')

        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model', log_path='./logs/results', eval_freq=500)
        #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])


        model = model.learn(
            total_timesteps=model_kwargs["total_timesteps"],
            tb_log_name=tb_log_name,
            callback=callback,
            #callback=TensorboardCallback(),
        )
        return model
