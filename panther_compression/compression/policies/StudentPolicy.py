from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)

from compression.utils.other import ActionManager, ObservationManager, ObstaclesManager, getPANTHERparamsAsCppStruct, readPANTHERparams
from colorama import init, Fore, Back, Style
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

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
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule = Callable[[float], float], #TODO: Andrea: not used, dummy
        net_arch: [List[int]] = [64, 64],
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_lstm: bool = False,
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
        self.om=ObservationManager()
        self.am=ActionManager()
        self.obsm=ObstaclesManager()
        par = getPANTHERparamsAsCppStruct()
        self.features_dim=self.om.getObservationSize()
        print("features_dim=", self.features_dim)
        self.use_lstm = use_lstm
        self.agent_input_dim = self.om.getAgentObservationSize()
        self.lstm_each_obstacle_dim = self.obsm.getSizeEachObstacle()
        self.lstm_output_dim = par.lstm_output_dim
        self.lstm_num_layers = par.lstm_num_layers
        self.lstm_bidirectional = par.lstm_bidirectional
        self.lstm_dropout = par.lstm_dropout
        self.use_bn = par.use_bn
        self.features_extractor_class = features_extractor_class
        print("use_lstm=", self.use_lstm)
        if self.use_lstm:
            print("lstm_output_dim=", self.lstm_output_dim)
            print("lstm_num_layers=", self.lstm_num_layers)
            print("lstm_bidirectional=", self.lstm_bidirectional)
        action_dim = get_action_dim(self.action_space)

        ##
        ## If using closed form yaw
        ##

        if(self.am.use_closed_form_yaw_student==True):
            action_dim = action_dim - self.am.traj_size_yaw_ctrl_pts*self.am.num_traj_per_action

        ##
        ## If using LSTM
        ##

        if self.use_lstm:

            self.lstm = nn.LSTM(input_size=self.lstm_each_obstacle_dim, hidden_size=self.lstm_output_dim, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.lstm_dropout)
            self.batch_norm = nn.BatchNorm1d(self.lstm_output_dim)
            latent_pi_net = create_mlp(self.agent_input_dim+self.lstm_output_dim, -1, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
            self.latent_pi = nn.Sequential(*latent_pi_net)
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim

        else: # if not using LSTM

            latent_pi_net = create_mlp(self.features_dim, -1, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
            self.latent_pi = nn.Sequential(*latent_pi_net)
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim

        print(f"self.net_arch={self.net_arch}") #This is a list containing the number of neurons in each layer (excluding input and output)
        #features_dim is the number of inputs (i.e., the number of input layers)
        print(f"last_layer_dim={last_layer_dim}") 
        print(f"action_dim={action_dim}") 

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)

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

    def get_action_dist_params(self, obs_n: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs_n:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """

        if self.use_lstm:

            ##
            ## get features
            ##

            features = self.extract_features(obs_n)
            
            ##
            ## devide features into agent and obst and reshape obst_features for LSTM
            ##

            agent_features = features[None, :, :self.agent_input_dim] # TODO: pass # obstacles and change 33 #None is for keeping the same dimension
            obst_features = features[None, :, self.agent_input_dim:] # TODO: pass # obstacles and change 33
            
            batch_size = features.shape[0]
            num_of_obstacles = int(obst_features.shape[2]/self.lstm_each_obstacle_dim) # need to calculate here because num_of_obstacles depends on each simulation
            
            ##
            ## batch_first=False, so (sequence_length, batch_size, input_size) = (num_of_obstacles, batch_size, lstm_each_obstacle_dim)
            ## reshape obst_features from (1, 20, 66) to (2, 20, 33) #FYI this doesn't work: lstm_input = th.reshape(obst_features, (num_of_obstacles, batch_size, self.lstm_each_obstacle_dim))
            ## initliaze lstm_input as an empty tensor
            ## https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
            ##

            lstm_input = th.empty((num_of_obstacles, batch_size, self.lstm_each_obstacle_dim)).to(self.device)
            for i in range(num_of_obstacles):
                for j in range(batch_size):
                    lstm_input[i,j,:] = obst_features[0,j,i*self.lstm_each_obstacle_dim:(i+1)*self.lstm_each_obstacle_dim]
            
            ##
            ## LSTM layer
            ##

            lstm_out, (h_n, c_n) = self.lstm(lstm_input)

            # th.set_printoptions(profile="full")
            # print("lstm_out ", lstm_out)
            # print("lstm_out[-1] ", lstm_out[-1])
            # print("h_n ", h_n)
            # print("h_n[-1] ", h_n[-1])
            # assert h_n[-1] == lstm_out[-1] #this is true

            ##
            ## Batch normalization
            ## h_n.shape  ([lastm_num_layers, batch_size, lstm_hidden_size])
            ##

            if self.use_bn:
                bn_out = self.batch_norm(h_n[-1])
            else:
                bn_out = h_n[-1]
            ##
            ## FC layers
            ##

            lstm_out_cat = th.cat((agent_features[-1], bn_out), dim=1)
            latent_pi = self.latent_pi(lstm_out_cat) #lstm_out_cat[None,:] -- None is added for dimension match
            
            ##
            ## Last layer
            ##

            mean_actions = self.mu(latent_pi)

        else: # if not using LSTM

            features = self.extract_features(obs_n)
            latent_pi = self.latent_pi(features)
            mean_actions = self.mu(latent_pi)

        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions.float(), log_std, {}

    def forward(self, obs_n: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs_n)
        # Note: the action is squashed
        output=self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        before_shape=list(output.shape)

        if self.am.use_closed_form_yaw_student:
            tmp=(before_shape[0],) + (self.am.num_traj_per_action, self.am.traj_size_pos_ctrl_pts + 1)
            output=th.reshape(output, tmp)
            dummy_yaw=th.zeros(output.shape[0], self.am.num_traj_per_action, self.am.traj_size_yaw_ctrl_pts, device=obs_n.device)
            output=th.cat((output[:,:,0:-1], dummy_yaw, output[:,:,-1:]),2)
        else:
            output=th.reshape(output, (before_shape[0],) + self.am.getActionShape())

        return output

    # def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
    #     # return action and associated log prob
    #     return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, obs_n: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action = self.forward(obs_n, deterministic)
        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)

        return action

    def predictSeveral(self, obs_n, deterministic: bool = False):

        #Note that here below we call predict, not _predict
        self.features_extractor = self.features_extractor_class(self.observation_space)
        acts=[self.predict( obs_n[i,:], deterministic=deterministic)[0].reshape(self.am.getActionShape()) for i in range(len(obs_n))] #Note that len() returns the size along the first axis
        
        acts=np.stack(acts, axis=0)
        return acts
