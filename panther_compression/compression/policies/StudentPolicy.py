from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import time
from statistics import mean
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
LOG_STD_MAX = 20
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
        self.agent_input_dim = self.om.getAgentObservationSize()
        self.lstm_each_obstacle_dim = self.obsm.getSizeEachObstacle()
        self.use_lstm = use_lstm
        self.use_bn = par.use_bn
        self.lstm_dropout = par.lstm_dropout
        self.lstm_output_dim = par.lstm_output_dim
        self.lstm_num_layers = par.lstm_num_layers
        self.lstm_bidirectional = par.lstm_bidirectional
        self.use_lstm_oa = par.use_lstm_oa
        self.use_bn_oa = par.use_bn_oa
        self.lstm_dropout_oa = par.lstm_dropout_oa
        self.lstm_output_dim_oa = par.lstm_output_dim_oa
        self.lstm_num_layers_oa = par.lstm_num_layers_oa
        self.lstm_bidirectional_oa = par.lstm_bidirectional_oa
        self.features_extractor_class = features_extractor_class
        print("use_lstm=", self.use_lstm)
        if self.use_lstm:
            print("lstm_output_dim=", self.lstm_output_dim)
            print("lstm_num_layers=", self.lstm_num_layers)
            print("lstm_bidirectional=", self.lstm_bidirectional)
        action_dim = get_action_dim(self.action_space)
        self.computation_times = []
        self.use_num_obses = True

        ##
        ## If using closed form yaw
        ##

        if(self.am.use_closed_form_yaw_student==True):
            action_dim = action_dim - self.am.traj_size_yaw_ctrl_pts*self.am.num_traj_per_action

        ##
        ## If using LSTM
        ##

        if self.use_lstm:

            ##
            ## LSTM for obstacles
            ##

            self.lstm = nn.LSTM(input_size=self.lstm_each_obstacle_dim, hidden_size=self.lstm_output_dim, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.lstm_dropout)
            self.batch_norm = nn.BatchNorm1d(self.lstm_output_dim)
            latent_fc = create_mlp(self.agent_input_dim+self.lstm_output_dim, -1, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
            self.latent_fc = nn.Sequential(*latent_fc)

            if self.use_lstm_oa:
                
                ##
                ## LSTM for other agents
                ##

                self.lstm_other_agents = nn.LSTM(input_size=self.lstm_each_obstacle_dim, hidden_size=self.lstm_output_dim_oa, num_layers=self.lstm_num_layers_oa, bidirectional=self.lstm_bidirectional_oa, dropout=self.lstm_dropout_oa)
                self.batch_norm_other_agents = nn.BatchNorm1d(self.lstm_output_dim_oa)
                
                ## overwrite the latent_fc
                latent_fc = create_mlp(self.agent_input_dim+self.lstm_output_dim+self.lstm_output_dim_oa, -1, net_arch, activation_fn) #Create multi layer perceptron, see
                self.latent_fc = nn.Sequential(*latent_fc)

        else: # if not using LSTM

            latent_fc = create_mlp(self.features_dim, -1, net_arch, activation_fn) #Create multi layer perceptron, see https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
            self.latent_fc = nn.Sequential(*latent_fc)

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim

        print(f"self.net_arch={self.net_arch}") #This is a list containing the number of neurons in each layer (excluding input and output)
        #features_dim is the number of inputs (i.e., the number of input layers)
        print(f"last_layer_dim={last_layer_dim}") 
        print(f"action_dim={action_dim}") 

        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.tanh = nn.Tanh()
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
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

    def get_action_dist_params(self, obs_n: th.Tensor, num_obst: float, num_oa: float) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs_n:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """

        if self.use_lstm:

            if self.use_lstm_oa:
                actions = self.get_actions_with_two_lstms(obs_n, num_obst, num_oa)
            else:
                actions = self.get_actions_with_one_lstm(obs_n, num_obst+num_oa)

        else: # if not using LSTM

            features = self.extract_features(obs_n)
            latent_fc = self.latent_fc(features)
            actions = self.mu(latent_fc)

        actions = self.tanh(actions)
        return actions.float()
    
    def forward(self, obs_n: th.Tensor, num_obst: float, num_oa:float, deterministic: bool = True) -> th.Tensor:
        
        ##
        ## get actions
        ##

        actions = self.get_action_dist_params(obs_n, num_obst, num_oa)
        
        ##
        ## Squashing the action
        ##
        
        output = actions
        before_shape=list(output.shape)

        if self.am.use_closed_form_yaw_student:
            tmp=(before_shape[0],) + (self.am.num_traj_per_action, self.am.traj_size_pos_ctrl_pts + 1)
            output=th.reshape(output, tmp)
            dummy_yaw=th.zeros(output.shape[0], self.am.num_traj_per_action, self.am.traj_size_yaw_ctrl_pts, device=obs_n.device)
            output=th.cat((output[:,:,0:-1], dummy_yaw, output[:,:,-1:]),2)
        else:
            output=th.reshape(output, (before_shape[0],) + self.am.getActionShape())

        return output

    def set_use_num_obses(self, use_num_obses):
        self.use_num_obses = use_num_obses
    
    def set_num_obs_num_oa(self, num_obs, num_oa):
        self.num_obs = num_obs
        self.num_oa = num_oa
    
    def get_actions_with_one_lstm(self, obs_n: th.Tensor, num_obst_and_oa: float) -> th.Tensor:

        """
        One LSTM for both obstacles and other agents
        """

        features = self.extract_features(obs_n)
        
        ##
        ## devide features into agent and obst and reshape obst_features for LSTM
        ##

        obst_and_oa_end_index = self.agent_input_dim + num_obst_and_oa*self.lstm_each_obstacle_dim

        agent_features = features[None, :, :self.agent_input_dim] #None is for keeping the same dimension
        obst_features = features[None, :, self.agent_input_dim:obst_and_oa_end_index]
        
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
        latent_fc = self.latent_fc(lstm_out_cat) #lstm_out_cat[None,:] -- None is added for dimension match
        
        ##
        ## Last layer
        ##

        actions = self.mu(latent_fc)

        return actions
    
    def get_actions_with_two_lstms(self, obs_n: th.Tensor, num_obst: float, num_oa: float) -> th.Tensor:

        """
        Two LSTMs for both obstacles and other agents
        """

        features = self.extract_features(obs_n)
        
        ##
        ## devide features into agent and obst and reshape obst_features for LSTM
        ##

        obst_oa_separation_index = self.agent_input_dim + num_obst*self.lstm_each_obstacle_dim

        agent_features = features[None, :, :self.agent_input_dim] #None is for keeping the same dimension
        obst_features = features[None, :, self.agent_input_dim:obst_oa_separation_index]
        oa_features = features[None, :, obst_oa_separation_index:obst_oa_separation_index + num_oa*self.lstm_each_obstacle_dim]
        
        batch_size = features.shape[0]
        num_of_obstacles = int(obst_features.shape[2]/self.lstm_each_obstacle_dim) # need to calculate here because num_of_obstacles depends on each simulation
        
        ##
        ## Obstacle LSTM layer
        ##

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
            bn_out_obst = self.batch_norm(h_n[-1])
        else:
            bn_out_obst = h_n[-1]

        ##
        ## Other Agents LSTM layer
        ##

        lstm_input = th.empty((num_oa, batch_size, self.lstm_each_obstacle_dim)).to(self.device)
        for i in range(num_oa):
            for j in range(batch_size):
                lstm_input[i,j,:] = oa_features[0,j,i*self.lstm_each_obstacle_dim:(i+1)*self.lstm_each_obstacle_dim]

        lstm_out, (h_n, c_n) = self.lstm_other_agents(lstm_input)

        if self.use_bn:
            bn_out_oa = self.batch_norm(h_n[-1])
        else:
            bn_out_oa = h_n[-1]

        ##
        ## FC layers
        ##

        lstm_out_cat = th.cat((agent_features[-1], bn_out_obst, bn_out_oa), dim=1)
        latent_fc = self.latent_fc(lstm_out_cat) #lstm_out_cat[None,:] -- None is added for dimension match
        
        ##
        ## Last layer
        ##

        actions = self.mu(latent_fc)

        return actions

    def _predict(self, obs_n: th.Tensor, deterministic: bool = True) -> th.Tensor:

        if self.use_num_obses:
            start = time.time()
            action = self.forward(obs_n, self.num_obses[self.i_index], self.num_oas[self.i_index] , deterministic) # hard-coded 10 and 33 for now
            end = time.time()
        else:
            start = time.time()
            action = self.forward(obs_n, self.num_obs, self.num_oa, deterministic)
            end = time.time()
        self.computation_times.append(end - start)
        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)
        return action

    def predictSeveral(self, obs_n, deterministic: bool = True):

        self.features_extractor = self.features_extractor_class(self.observation_space)
        acts=[]
        for i in range(len(obs_n)):
            self.i_index = i
            acts.append(self.predict( obs_n[i,:], deterministic=deterministic)[0].reshape(self.am.getActionShape()))
        # acts=[self.predict( obs_n[i, :], deterministic=deterministic)[0].reshape(self.am.getActionShape()) for i in range(len(obs_n))] #Note that len() returns the size along the first axis
        acts=np.stack(acts, axis=0)
        return acts
    
    def predictSeveralWithComputationTimeVerbose(self, obs_n, num_obses, num_oas, deterministic: bool = True):

        #Note that here below we call predict, not _predict
        self.num_obses = num_obses
        self.num_oas = num_oas
        self.features_extractor = self.features_extractor_class(self.observation_space)
        acts=[]
        for i in range(len(obs_n)):
            self.i_index = i
            acts.append(self.predict( obs_n[i,:], deterministic=deterministic)[0].reshape(self.am.getActionShape()))
        # acts=[self.predict( obs_n[i, :], deterministic=deterministic)[0].reshape(self.am.getActionShape()) for i in range(len(obs_n))] #Note that len() returns the size along the first axis
        acts=np.stack(acts, axis=0)
        return acts, self.computation_times
