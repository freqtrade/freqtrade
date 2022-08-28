import logging
from pathlib import Path
from typing import Any, Dict  # , Tuple

# import numpy.typing as npt
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (BaseReinforcementLearningModel,
                                                                make_env)


logger = logging.getLogger(__name__)


class ReinforcementLearner_multiproc(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
                                    tensorboard_log=Path(dk.full_path / "tensorboard"),
                                    **self.freqai_info['model_training_parameters']
                                    )
        else:
            logger.info('Continual learning activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.tensorboard_log = Path(dk.data_path / "tensorboard")
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=self.eval_callback
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model

    def set_train_and_eval_environments(self, data_dictionary, prices_train, prices_test, dk):
        """
        If user has particular environment configuration needs, they can do that by
        overriding this function. In the present case, the user wants to setup training
        environments for multiple workers.
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        env_id = "train_env"
        num_cpu = int(self.freqai_info["rl_config"]["thread_count"])
        self.train_env = SubprocVecEnv([make_env(self.MyRLEnv, env_id, i, 1, train_df, prices_train,
                                        self.reward_params, self.CONV_WIDTH, monitor=True,
                                        config=self.config) for i
                                        in range(num_cpu)])

        eval_env_id = 'eval_env'
        self.eval_env = SubprocVecEnv([make_env(self.MyRLEnv, eval_env_id, i, 1,
                                                test_df, prices_test,
                                                self.reward_params, self.CONV_WIDTH, monitor=True,
                                                config=self.config) for i
                                       in range(num_cpu)])
        self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                          render=False, eval_freq=len(train_df),
                                          best_model_save_path=dk.data_path)
