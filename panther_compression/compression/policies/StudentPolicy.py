from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
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

from compression.utils.other import ActionManager, ObservationManager
from colorama import init, Fore, Back, Style
import numpy as np


class StudentPolicy(BasePolicy):
    """
    Actor network (policy) for Dagger, taken from SAC of stable baselines3.
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py#L26

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features (a CNN when using images, a nn.Flatten() layer otherwise)
    :param obs_dim: Number of features
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule = Callable[[float], float], #TODO: Andrea: not used, dummy
        net_arch: [List[int]] = [64, 64],
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_numel: int = 2, # Size of input features
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

        self.obs_numel=self.om.getObservationSize()
        self.pos_numel=self.am.action_size_pos_ctrl_pts
        self.yaw_numel=self.am.action_size_yaw_ctrl_pts
        self.time_numel=self.am.action_size_time
        self.prob_numel=self.am.action_size_prob
        self.pos_yaw_time_numel=self.pos_numel+self.yaw_numel + self.time_numel

        action_numel = get_action_dim(self.action_space)

        mlp = create_mlp(self.obs_numel, self.pos_yaw_time_numel, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
        mlp.append(nn.Tanh())

        self.my_nn_obs2posyawtime = nn.Sequential(*mlp) #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

        mlp = create_mlp(self.pos_numel + self.obs_numel, self.prob_numel, net_arch, activation_fn)
        mlp.append(nn.Tanh())

        self.my_nn_pos2prob = nn.Sequential(*mlp)


    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                obs_numel=self.obs_numel,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def printwithName(self,data):
        print(self.name+data)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        features = self.extract_features(obs)

        pos_yaw_time = self.my_nn_obs2posyawtime(features)

        # print(f"pos_yaw_time.shape={pos_yaw_time.shape}")
        # print(f"self.pos_numel={self.pos_numel}")
        # print(f"self.yaw_numel={self.yaw_numel}")
        # print(f"self.am.traj_size={self.am.traj_size}")
        # print(f"self.am.traj_size_yaw_ctrl_pts={self.am.traj_size_yaw_ctrl_pts}")
        # print(f"self.am.getActionYawShape()={self.am.getActionYawShape()}")

        pos= pos_yaw_time[:,0:self.pos_numel]
        yaw= pos_yaw_time[:,self.pos_numel:self.pos_numel+self.yaw_numel]
        time= pos_yaw_time[:,self.pos_numel+self.yaw_numel:]

        prob = self.my_nn_pos2prob(th.cat((features,pos), dim=1))

        first_dim = list(pos_yaw_time.shape)[0]

        # print(f"shape of yaw is {yaw.shape}")
        # print(f"shape of time is {time.shape}")
        # print(f"tmp is {(first_dim,) + self.am.getActionYawShape()}")

        pos_reshaped = th.reshape(pos, (first_dim,) + self.am.getActionPosShape())
        yaw_reshaped = th.reshape(yaw, (first_dim,) + self.am.getActionYawShape())
        time_reshaped = th.reshape(time, (first_dim,) + self.am.getActionTimeShape())
        prob_reshaped = th.reshape(prob, (first_dim,) + self.am.getActionProbShape())

        # print(f"pos_reshaped={pos_reshaped.shape}")
        # print(f"yaw_reshaped={yaw_reshaped.shape}")
        # print(f"prob_reshaped={prob_reshaped.shape}")

        output= th.cat((pos_reshaped, yaw_reshaped, time_reshaped, prob_reshaped), dim=2)

        # print(f"output={output.shape}")
        # print(f"self.am.getActionShape()={self.am.getActionShape()}")


        # first_dim = list(pos_yaw.shape)[0]
        # output= th.reshape(output, (first_dim,) + self.am.getActionShape())

        # output = th.tanh(self.my_nn_obs2posyaw(features))

        # # self.printwithName(f"In forward, output before reshaping={output.shape}")
        
        # #Note that before_shape[i,:,:] containes one action i
        # output=th.reshape(output, (before_shape[0],) + self.am.getActionShape())
        # self.printwithName(f"In forward, returning shape={output.shape}")

        return output #self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def _predict(self, obs_n: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.printwithName(f"Calling student")
        # self.printwithName(f"Received obs={obs_n}")
        # obs=self.om.denormalizeObservation(obs_n.cpu().numpy().reshape(self.om.getObservationShape()))
        # self.om.printObservation(obs)

        # self.printwithName(f"Received obs={observation.numpy()}")
        # assertIsNormalized(observation.cpu().numpy())
        # self.printwithName(f"Received obs shape={observation.shape}")
        action = self.forward(obs_n, deterministic)
        # self.printwithName(f"action={action}")

        #Sort each of the trajectories from highest to lowest probability
        # indexes=th.argsort(action[:,:,-1], dim=1, descending=True) #TODO: Assumming here that the last number is the probability!
        # action = action[:,indexes,:]
        # self.printwithName(f"indexes={indexes}")     
        # self.printwithName(f"After sorting, action={action}")
        #############

        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)

        # self.printwithName(f"In predict_, returning shape={action.shape}")
        return action

    def predictSeveral(self, obs_n, deterministic: bool = False):

        #Note that here below we call predict, not _predict
        acts=[self.predict( obs_n[i,:], deterministic=deterministic)[0].reshape(self.am.getActionShape()) for i in range(len(obs_n))] #Note that len() returns the size along the first axis
        acts=np.stack(acts, axis=0)
        return acts
