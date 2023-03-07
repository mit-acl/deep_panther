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
        print("features_dim= ", self.features_dim)
        
        self.use_lstm = use_lstm

        self.agent_input_dim = self.om.getAgentObservationSize()
        self.lstm_each_obstacle_dim = self.obsm.getSizeEachObstacle()

        self.lstm_output_dim = par.lstm_output_dim
        print("use_lstm? ", self.use_lstm)
        if self.use_lstm:
            print("lstm_output_dim= ", self.lstm_output_dim)

        action_dim = get_action_dim(self.action_space)

        ###
        if(self.am.use_closed_form_yaw_student==True):
            action_dim = action_dim - self.am.traj_size_yaw_ctrl_pts*self.am.num_traj_per_action

        ####

        if self.use_lstm:

            # latent_pi_net = create_lstm_mlp(self.features_dim, -1, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
            # self.latent_pi = nn.Sequential(*latent_pi_net)
            # last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim

            self.lstm = nn.LSTM(self.lstm_each_obstacle_dim, self.lstm_output_dim)
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

            print("using LSTM")

            ##
            ## get features
            ##

            features = self.extract_features(obs_n)
            
            ##
            ## devide features into agent and obst and reshape obst_features for LSTM
            ##

            agent_features = features[None, 0, :self.agent_input_dim] # TODO: pass # obstacles and change 33 #None is for keeping the same dimension
            obst_features = features[None, 0, self.agent_input_dim:] # TODO: pass # obstacles and change 33
            num_of_obstacles = int(obst_features.shape[1]/self.lstm_each_obstacle_dim) # need to calculate here because num_of_obstacles depends on each simulation
            lstm_input = th.reshape(obst_features, (num_of_obstacles, self.lstm_each_obstacle_dim))
            
            ##
            ## LSTM layer
            ##

            lstm_out, _ = self.lstm(lstm_input)

            ##
            ## FC layers
            ##

            lstm_out_cat = th.cat((agent_features[0], lstm_out[-1])) #agent[0] for dimension match, lstm_out[-1] because we only need the last output from LSTM.
            latent_pi = self.latent_pi(lstm_out_cat[None,:]) #lstm_out_cat[None,:] -- None is added for dimension match
            
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
        output=self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs);

        # self.printwithName(f"In forward, output before reshaping={output.shape}")
        before_shape=list(output.shape)

        ###
        if(self.am.use_closed_form_yaw_student==True):
            tmp=(before_shape[0],) + (self.am.num_traj_per_action, self.am.traj_size_pos_ctrl_pts + 1)

            output=th.reshape(output, tmp)

            dummy_yaw=th.zeros(output.shape[0], self.am.num_traj_per_action, self.am.traj_size_yaw_ctrl_pts, device=obs_n.device)
            output=th.cat((output[:,:,0:-1], dummy_yaw, output[:,:,-1:]),2)
        else:
            output=th.reshape(output, (before_shape[0],) + self.am.getActionShape())


        # self.printwithName(f"In forward, returning shape={output.shape}")
        
        return output

    # def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
    #     # return action and associated log prob
    #     return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

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


    # def predictAndDenormalize(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     action =self._predict(observation, deterministic)
    #     return self.am.denormalizeAction(action)
    
    # def create_lstm_mlp(
    #     lstm_input_dim: int,
    #     lstm_output_dim: int,
    #     mlp_input_dim: int,
    #     mlp_output_dim: int,
    #     mlp_net_arch: List[int],
    #     activation_fn: Type[nn.Module] = nn.ReLU,
    #     squash_output: bool = False,
    #     with_bias: bool = True,
    # ) -> List[nn.Module]:
    #     """
    #     refered to https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

    #     Create a LSTM layer followd by multi layer perceptron (MLP), which is
    #     a collection of fully-connected layers each followed by an activation function.

    #     Note that Agent current state goes into MLP, but observations first go into LSTM, and then it will generates a fixed
    #     size vector, h, which will then go into MLP together with the agent current state

    #     :param lstm_input_dim: Dimension of the input vector
    #     :param lstm_output_dim:
    #     :param mlp_input_dim: this is equal to agent's state dim + lstm_output_dim
    #     :param mlp_output_dim:
    #     :param mlp_net_arch: Architecture of the neural net
    #     :param activation_fn: The activation function to use after each layer.
    #     :param squash_output: Whether to squash the output using a Tanh
    #         activation function
    #     :param with_bias: If set to False, the layers will not learn an additive bias
    #     :return:
    #     """

    #     ##
    #     ## lstm layer
    #     ##

    #     modules = [nn.LSTM(lstm_input_dim, lstm_output_dim, bias=with_bias), activation_fn()]

    #     ##
    #     ## FC
    #     ##

    #     modules.append(nn.Linear(mlp_input_dim, net_arch[0], bias=with_bias))

    #     for idx in range(len(mlp_net_arch) - 1):
    #         modules.append(nn.Linear(mlp_net_arch[idx], mlp_net_arch[idx + 1], bias=with_bias))
    #         modules.append(activation_fn())

    #     last_layer_dim = mlp_net_arch[-1] if len(mlp_net_arch) > 0 else input_dim
    #     modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))

    #     if squash_output:
    #         modules.append(nn.Tanh())
    #     return modules