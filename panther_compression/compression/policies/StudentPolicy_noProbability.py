from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)

from compression.utils.ActionManager import ActionManager
from compression.utils.ObservationManager import ObservationManager

from colorama import init, Fore, Back, Style

from gymnasium import spaces


class StudentPolicy(BasePolicy):
    """
    Actor network (policy) for Dagger, taken from SAC of stable baselines3.
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py#L26

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule = Callable[[float], float], #TODO: Andrea: not used, dummy
        net_arch: [List[int]] = [64, 64],
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_dim: int = 2, # Size of input features
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(StudentPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor = features_extractor_class(observation_space),
            features_extractor_kwargs = features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        # Save arguments to re-create object at loading
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.name=Style.BRIGHT+Fore.WHITE+"  [Stu]"+Style.RESET_ALL

        self.om=ObservationManager();
        self.am=ActionManager();

        self.features_dim=self.om.getObservationSize()


        action_dim = get_action_dim(self.action_space)

        mlp = create_mlp(self.features_dim, action_dim, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
        self.my_nn = nn.Sequential(*mlp) #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html


    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def printwithName(self,data):
        print(self.name+data)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        features = self.extract_features(obs)
        output = th.tanh(self.my_nn(features))

        # self.printwithName(f"In forward, output before reshaping={output.shape}")
        before_shape=list(output.shape)
        #Note that before_shape[i,:,:] containes one action i
        output=th.reshape(output, (before_shape[0],) + self.am.getActionShape())
        # self.printwithName(f"In forward, returning shape={output.shape}")

        return output #self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.printwithName(f"Calling student")
        # self.printwithName(f"Received obs={observation}")
        # self.printwithName(f"Received obs={observation.numpy()}")
        # self.om.assertObsIsNormalized(observation.cpu().numpy().reshape(self.om.getObservationShape()), self.name)
        # self.printwithName(f"Received obs shape={observation.shape}")
        action = self.forward(observation, deterministic)
        # self.printwithName(f"action={action}")
        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)

        # self.printwithName(f"Returning action shape={action.shape}")
        return action
        