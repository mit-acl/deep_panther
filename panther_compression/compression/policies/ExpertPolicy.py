import sys
import numpy as np
import copy
from random import random
from compression.utils.other import ActionManager, ObservationManager
from colorama import init, Fore, Back, Style

class ExpertPolicy(object):

    def __init__(self):
        # self.max_act = 12.0
        # self.max_obs = 10.0
        self.am=ActionManager();
        self.om=ObservationManager();


        self.action_shape=self.am.getActionShape();
        self.observation_shape=self.om.getObservationShape();

        self.name=Style.BRIGHT+Fore.BLUE+"[Exp]"+Style.RESET_ALL

        self.reset()
        print(self.name+f" Initialized.")

    def printwithName(self,data):
        print(self.name+data)

    def reset(self):
        pass
        # self.mpc_state_size = 8
        # self.mpc_act_size = 3 

    def predict(self, obs, deterministic=True):
        obs=obs.reshape(self.observation_shape) #Not sure why this is needed
        assert obs.shape==self.observation_shape, self.name+f"ERROR: obs.shape={obs.shape} but self.observation_shape={self.observation_shape}"
        
        obs=self.om.denormalizeObservation(obs);

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.

        self.printwithName(f"Got obs={obs}")
        # self.printwithName(f"Got obs shape={obs.shape}")

        # action=np.random.rand(1, self.mpc_act_size)

        # action_normalized = (1/self.max_act)*np.array([action]).reshape((1, self.mpc_act_size))

        # assert not np.isnan(np.sum(action_normalized)), f"trying to output nan"
        
        Q=random(); #This is the reward I think

        action = self.am.getRandomAction()
        self.printwithName(f" Returned action={action}")
        # self.printwithName(f"Returned action shape={action.shape}")

        action=self.am.normalizeAction(action)

        assert action.shape==self.action_shape

        return action, {"Q": Q}

