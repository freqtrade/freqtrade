import copy
import logging
import gc
from pathlib import Path
from typing import Any, Dict, Type, Callable, List, Optional, Union

import numpy as np
import torch as th
import pandas as pd
from pandas import DataFrame
from gymnasium import spaces
import matplotlib
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import HParam, Figure

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment, BaseActions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback


logger = logging.getLogger(__name__)


class ReinforcementLearner_DDPG_TD3(BaseReinforcementLearningModel):
    """
    Reinforcement Learning Model prediction model for DDPG and TD3.

    Users can inherit from this class to make their own RL model with custom
    environment/training controls. Define the file as follows:

    ```
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner

    class MyCoolRLModel(ReinforcementLearner):
    ```

    Save the file to `user_data/freqaimodels`, then run it with:

    freqtrade trade --freqaimodel MyCoolRLModel --config config.json --strategy SomeCoolStrat

    Here the users can override any of the functions
    available in the `IFreqaiModel` inheritance tree. Most importantly for RL, this
    is where the user overrides `MyRLEnv` (see below), to define custom
    `calculate_reward()` function, or to override any other parts of the environment.

    This class also allows users to override any other part of the IFreqaiModel tree.
    For example, the user can override `def fit()` or `def train()` or `def predict()`
    to take fine-tuned control over these processes.

    Another common override may be `def data_cleaning_predict()` where the user can
    take fine-tuned control over the data handling pipeline.
    """

    def __init__(self, **kwargs) -> None:
        """
        Model specific config
        """
        super().__init__(**kwargs)

        # Enable learning rate linear schedule
        self.lr_schedule: bool = self.rl_config.get("lr_schedule", False)

        # Enable tensorboard logging
        self.activate_tensorboard: bool = self.rl_config.get("activate_tensorboard", True)
        # TENSORBOARD CALLBACK DOES NOT RECOMMENDED TO USE WITH MULTIPLE ENVS,
        # IT WILL RETURN FALSE INFORMATIONS, NEVERTHLESS NOT THREAD SAFE WITH SB3!!!

        # Enable tensorboard rollout plot
        self.tensorboard_plot: bool = self.rl_config.get("tensorboard_plot", False)

    def get_model_params(self):
        """
        Get the model specific parameters
        """
        model_params = copy.deepcopy(self.freqai_info["model_training_parameters"])
        
        if self.lr_schedule:
            _lr = model_params.get('learning_rate', 0.0003)
            model_params["learning_rate"] = linear_schedule(_lr)
            logger.info(f"Learning rate linear schedule enabled, initial value: {_lr}")

        model_params["policy_kwargs"] = dict(
        net_arch=dict(vf=self.net_arch, pi=self.net_arch),
        activation_fn=th.nn.ReLU,
        optimizer_class=th.optim.Adam

        return model_params

    def get_callbacks(self, eval_freq, data_path) -> list:
        """
        Get the model specific callbacks
        """
        callbacks = []
        callbacks.append(self.eval_callback)
        if self.activate_tensorboard:
            callbacks.append(CustomTensorboardCallback())
        if self.tensorboard_plot:
            callbacks.append(FigureRecorderCallback())
        return callbacks

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test
            features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=self.net_arch)

        if self.activate_tensorboard:
            tb_path = Path(dk.full_path / "tensorboard" / dk.pair.split('/')[0])
        else:
            tb_path = None

        model_params = self.get_model_params()
        logger.info(f"Params: {model_params}")

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(self.policy_type, self.train_env,
                                    tensorboard_log=tb_path,
                                    **model_params)
        else:
            logger.info("Continual training activated - starting training from previously "
                        "trained agent.")
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=int(total_timesteps),
            #callback=[self.eval_callback, self.tensorboard_callback],
            callback=self.get_callbacks(len(train_df), str(dk.data_path)),
            progress_bar=self.rl_config.get("progress_bar", False)
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info("Callback found a best model.")
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info("Couldnt find best model, using final model instead.")

        return model

    MyRLEnv: Type[BaseEnvironment]

    class MyRLEnv(Base5ActionRLEnv):  # type: ignore[no-redef]
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.
        """
        def __init__(self, df, prices, reward_kwargs, window_size=10, starting_point=True, id="boxenv-1", seed=1, config={}, live=False, fee=0.0015, can_short=False, pair="", df_raw=None, action_space_type="Box"):
            super().__init__(df, prices, reward_kwargs, window_size, starting_point, id, seed, config, live, fee, can_short, pair, df_raw)
            
            # Define the action space as a continuous space between -1 and 1 for a single action dimension
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
            # Define the observation space as before
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size, self.total_features),
                dtype=np.float32
            )

        def calculate_reward(self, action: int) -> float:
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.

                        Warning!
            This is function is a showcase of functionality designed to show as many possible
            environment control features as possible. It is also designed to run quickly
            on small computers. This is a benchmark, it is *not* for live production.

            :param action: int = The action made by the agent for the current candle.
            :return:
            float = the reward to give to the agent for current step (used for optimization
                of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -2

            pnl = self.get_unrealized_profit()
            factor = 100.

            # reward agent for entering trades
            if (action == Actions.Long_enter.value
                    and self._position == Positions.Neutral):
                return 25
            if (action == Actions.Short_enter.value
                    and self._position == Positions.Neutral):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position in (Positions.Short, Positions.Long) and
                    action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                return float(pnl * factor)

            # close short
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                return float(pnl * factor)

            return 0.

        def step(self, action):
            """
            Logic for a single step (incrementing one candle in time)
            by the agent
            :param: action: int = the action type that the agent plans
                to take for the current step.
            :returns:
                observation = current state of environment
                step_reward = the reward from `calculate_reward()`
                _done = if the agent "died" or if the candles finished
                info = dict passed back to openai gym lib
            """
            
            # Ensure action is within the range [-1, 1]
            action = np.clip(action, -1, 1)
            
            # Apply noise for exploration
            self.noise_std = 0.3  # Standard deviation for exploration noise
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.tanh(action + noise)  # Ensure action is within -1 to 1
    
            # Map the continuous action to one of the five discrete actions
            discrete_action = self._map_continuous_to_discrete(action)
            
            #print(f"{self._current_tick} Action!!!: {action}")
            #print(f"{self._current_tick} Discrete Action!!!: {discrete_action}")

            self._done = False
            self._current_tick += 1
    
            if self._current_tick == self._end_tick:
                self._done = True
    
            self._update_unrealized_total_profit()
            step_reward = self.calculate_reward(discrete_action)
            self.total_reward += step_reward
            
            self.tensorboard_log(self.actions._member_names_[discrete_action], category="actions")
    
            trade_type = None
            if self.is_tradesignal(discrete_action):
    
                if discrete_action == Actions.Neutral.value:
                    self._position = Positions.Neutral
                    trade_type = "neutral"
                    self._last_trade_tick = None
                elif discrete_action == Actions.Long_enter.value:
                    self._position = Positions.Long
                    trade_type = "enter_long"
                    self._last_trade_tick = self._current_tick
                elif discrete_action == Actions.Short_enter.value:
                    self._position = Positions.Short
                    trade_type = "enter_short"
                    self._last_trade_tick = self._current_tick
                elif discrete_action == Actions.Long_exit.value:
                    self._update_total_profit()
                    self._position = Positions.Neutral
                    trade_type = "exit_long"
                    self._last_trade_tick = None
                elif discrete_action == Actions.Short_exit.value:
                    self._update_total_profit()
                    self._position = Positions.Neutral
                    trade_type = "exit_short"
                    self._last_trade_tick = None
                else:
                    print("case not defined")
    
                if trade_type is not None:
                    self.trade_history.append(
                        {"price": self.current_price(), "index": self._current_tick,
                         "type": trade_type, "profit": self.get_unrealized_profit()})
    
            if (self._total_profit < self.max_drawdown or
                    self._total_unrealized_profit < self.max_drawdown):
                self._done = True
    
            self._position_history.append(self._position)
    
            info = dict(
                tick=self._current_tick,
                action=discrete_action,
                total_reward=self.total_reward,
                total_profit=self._total_profit,
                position=self._position.value,
                trade_duration=self.get_trade_duration(),
                current_profit_pct=self.get_unrealized_profit()
            )
    
            observation = self._get_observation()
            # user can play with time if they want
            truncated = False
    
            self._update_history(info)
    
            return observation, step_reward, self._done, truncated, info
    
        def _map_continuous_to_discrete(self, action):
            """
            Map the continuous action (a value between -1 and 1) to one of the discrete actions.
            """
            action_value = action[0]  # Extract the single continuous action value
            
            # Define the number of discrete actions
            num_discrete_actions = 5
            
            # Calculate the step size for each interval
            step_size = 2 / num_discrete_actions  # (2 because range is from -1 to 1)
            
            # Generate the boundaries dynamically
            boundaries = th.linspace(-1 + step_size, 1 - step_size, steps=num_discrete_actions - 1)
            
            # Find the bucket index for the action value
            bucket_index = th.bucketize(th.tensor(action_value), boundaries, right=True)
            
            # Map the bucket index to discrete actions
            discrete_actions = [
                BaseActions.Neutral,
                BaseActions.Long_enter,
                BaseActions.Long_exit,
                BaseActions.Short_enter,
                BaseActions.Short_exit
            ]
            
            return discrete_actions[bucket_index].value

        def get_rollout_history(self) -> DataFrame:
            """
            Get environment data from the first to the last trade
            """
            _history_df = pd.DataFrame.from_dict(self.history)
            _trade_history_df = pd.DataFrame.from_dict(self.trade_history)
            _rollout_history = _history_df.merge(_trade_history_df, left_on="tick", right_on="index", how="left")
            
            _price_history = self.prices.iloc[_rollout_history.tick].copy().reset_index()

            history = pd.merge(
                _rollout_history,
                _price_history,
                left_index=True, right_index=True
            )
            return history

        def get_rollout_plot(self):
            """
            Plot trades and environment data
            """
            def transform_y_offset(ax, offset):
                return mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=offset, units="inches")

            def plot_markers(ax, ticks, marker, color, size, offset):
                ax.plot(ticks, marker=marker, color=color, markersize=size, fillstyle="full",
                        transform=transform_y_offset(ax, offset), linestyle="none")

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                nrows=5, ncols=1,
                figsize=(16, 9),
                height_ratios=[6, 1, 1, 1, 1],
                sharex=True
            )

            # Return empty fig if no trades
            if len(self.trade_history) == 0:
                return fig

            history = self.get_rollout_history()
            enter_long_prices = history.loc[history["type"] == "enter_long"]["price"]
            enter_short_prices = history.loc[history["type"] == "enter_short"]["price"]
            exit_long_prices = history.loc[history["type"] == "exit_long"]["price"]
            exit_short_prices = history.loc[history["type"] == "exit_short"]["price"]

            axs[0].plot(history["open"], linewidth=1, color="#c28ce3")
            plot_markers(axs[0], enter_long_prices, "^", "#4ae747", 5, -0.05)
            plot_markers(axs[0], enter_short_prices, "v", "#f53580", 5, 0.05)
            plot_markers(axs[0], exit_long_prices, "o", "#4ae747", 3, 0)
            plot_markers(axs[0], exit_short_prices, "o", "#f53580", 3, 0)

            axs[1].set_ylabel("pnl")
            axs[1].plot(history["current_profit_pct"], linewidth=1, color="#a29db9")
            axs[1].axhline(y=0, label='0', alpha=0.33)
            axs[2].set_ylabel("duration")
            axs[2].plot(history["trade_duration"], linewidth=1, color="#a29db9")
            axs[3].set_ylabel("total_reward")
            axs[3].plot(history["total_reward"], linewidth=1, color="#a29db9")
            axs[3].axhline(y=0, label='0', alpha=0.33)
            axs[4].set_ylabel("total_profit")
            axs[4].set_xlabel("tick")
            axs[4].plot(history["total_profit"], linewidth=1, color="#a29db9")
            axs[4].axhline(y=1, label='1', alpha=0.33)

            for _ax in axs:
                for _border in ["top", "right", "bottom", "left"]:
                    _ax.spines[_border].set_color("#5b5e4b")

            fig.suptitle(
                "Total Reward: %.6f" % self.total_reward + " ~ " +
                "Total Profit: %.6f" % self._total_profit
            )
            fig.tight_layout()

            return fig

        def close(self) -> None:
            gc.collect()
            th.cuda.empty_cache()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class CustomTensorboardCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def _on_training_start(self) -> None:
        _lr = self.model.learning_rate
        
        if self.model.__class__.__name__ == "DDPG":
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "buffer_size": self.model.buffer_size,
                "learning_rate": _lr if isinstance(_lr, float) else "lr_schedule",
                "learning_starts": self.model.learning_starts,
                "batch_size": self.model.batch_size,
                "tau": self.model.tau,
                "gamma": self.model.gamma,
                "train_freq": self.model.train_freq,
                "gradient_steps": self.model.gradient_steps,
            }

        elif self.model.__class__.__name__ == "TD3":
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning_rate": _lr if isinstance(_lr, float) else "lr_schedule",
                "buffer_size": self.model.buffer_size,
                "learning_starts": self.model.learning_starts,
                "batch_size": self.model.batch_size,
                "tau": self.model.tau,
                "gamma": self.model.gamma,
                "train_freq": self.model.train_freq,
                "gradient_steps": self.model.gradient_steps,
                "policy_delay": self.model.policy_delay,
                "target_policy_noise": self.model.target_policy_noise,
                "target_noise_clip": self.model.target_noise_clip,
            }
            
        else:
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning_rate": _lr if isinstance(_lr, float) else "lr_schedule",
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
            }

        # Convert hparam_dict values to str if they are not of type int, float, str, bool, or torch.Tensor
        hparam_dict = {k: (str(v) if not isinstance(v, (int, float, str, bool, th.Tensor)) else v) for k, v in hparam_dict.items()}
        
        metric_dict = {
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "info/total_profit": 1,
            "info/trades_count": 0,
            "info/trade_duration": 0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )   
    
    def _on_step(self) -> bool:

        local_info = self.locals["infos"][0]
        if self.training_env is None:
            return True

        tensorboard_metrics = self.training_env.env_method("get_wrapper_attr", "tensorboard_metrics")[0]

        for metric in local_info:
            if metric not in ["episode", "terminal_observation", "TimeLimit.truncated"]:
                self.logger.record(f"info/{metric}", local_info[metric])

        for category in tensorboard_metrics:
            for metric in tensorboard_metrics[category]:
                self.logger.record(f"{category}/{metric}", tensorboard_metrics[category][metric])

        return True
    
class FigureRecorderCallback(BaseCallback):
    """
    Tensorboard figures callback
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        try:
            # Access the rollout plot directly from the base environment
            figures = [env.unwrapped.get_rollout_plot() for env in self.training_env.envs]
        except AttributeError:
            # If the above fails, try getting it from the wrappers
            figures = self.training_env.env_method("get_wrapper_attr", "get_rollout_plot")

        for i, fig in enumerate(figures):
            self.logger.record(
                f"rollout/env_{i}",
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
        return True
