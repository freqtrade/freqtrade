import logging
from typing import Any, Dict

from pandas import DataFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.BaseReinforcementLearningModel import make_env
from freqtrade.freqai.RL.TensorboardCallback import TensorboardCallback


logger = logging.getLogger(__name__)


class ReinforcementLearner_multiproc(ReinforcementLearner):
    """
    Demonstration of how to build vectorized environments
    """

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, Any],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        User can override this if they are using a custom MyRLEnv
        :param data_dictionary: dict = common data dictionary containing train and test
            features/labels/weights.
        :param prices_train/test: DataFrame = dataframe comprised of the prices to be used in
            the environment during training
        or testing
        :param dk: FreqaiDataKitchen = the datakitchen for the current pair
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        env_info = self.pack_env_dict()

        env_id = "train_env"
        self.train_env = SubprocVecEnv([make_env(self.MyRLEnv, env_id, i, 1,
                                        train_df, prices_train,
                                        monitor=True,
                                        env_info=env_info) for i
                                        in range(self.max_threads)])

        eval_env_id = 'eval_env'
        self.eval_env = SubprocVecEnv([make_env(self.MyRLEnv, eval_env_id, i, 1,
                                                test_df, prices_test,
                                                monitor=True,
                                                env_info=env_info) for i
                                       in range(self.max_threads)])
        self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                          render=False, eval_freq=len(train_df),
                                          best_model_save_path=str(dk.data_path))

        actions = self.train_env.env_method("get_actions")[0]
        self.tensorboard_callback = TensorboardCallback(verbose=1, actions=actions)
