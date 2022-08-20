import logging
import torch as th
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from pathlib import Path
from stable_baselines3.dqn.policies import (CnnPolicy, DQNPolicy, MlpPolicy,
                                            QNetwork)
from torch import nn
import gym
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import BasePolicy

logger = logging.getLogger(__name__)


class ReinforcementLearnerCustomAgent(BaseReinforcementLearningModel):
    """
    User can customize agent by defining the class and using it directly.
    Here the example is "TDQN"
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        # TDQN is a custom agent defined below
        model = TDQN(self.policy_type, self.train_env,
                     tensorboard_log=Path(dk.data_path / "tensorboard"),
                     policy_kwargs=policy_kwargs,
                     **self.freqai_info['model_training_parameters']
                     )

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

# User creates their custom agent and networks as shown below


def create_mlp_(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    dropout = 0.2
    if len(net_arch) > 0:
        number_of_neural = net_arch[0]

    modules = [
        nn.Linear(input_dim, number_of_neural),
        nn.BatchNorm1d(number_of_neural),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(number_of_neural, number_of_neural),
        nn.BatchNorm1d(number_of_neural),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(number_of_neural, number_of_neural),
        nn.BatchNorm1d(number_of_neural),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(number_of_neural, number_of_neural),
        nn.BatchNorm1d(number_of_neural),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(number_of_neural, output_dim)
    ]
    return modules


class TDQNetwork(QNetwork):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 features_extractor: nn.Module,
                 features_dim: int,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True
                 ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images
        )
        action_dim = self.action_space.n
        q_net = create_mlp_(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net).apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            th.nn.init.kaiming_uniform_(m.weight)


class TDQNPolicy(DQNPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def make_q_net(self) -> TDQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return TDQNetwork(**net_args).to(self.device)


class TMultiInputPolicy(TDQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class TDQN(DQN):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "TMultiInputPolicy": TMultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[TDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[Path] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,  # No action noise
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )
