import sys
import numpy as np
import copy
from random import random

class ExpertPolicy(object):

    def __init__(self):
        self.max_act = 12.0
        self.max_obs = 10.0
        self.reset()
        print("[ExpertPolicy] Initialized.")

    def reset(self):
        self.mpc_state_size = 8
        self.mpc_act_size = 3 

    def predict(self, obs, deterministic=True):

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.

        print(f"[Expert] obs={obs}")

        action=np.random.rand(1, self.mpc_act_size)

        action_normalized = (1/self.max_act)*np.array([action]).reshape((1, self.mpc_act_size))

        assert not np.isnan(np.sum(action_normalized)), f"trying to output nan"
        
        Q=random(); #This is the reward I think

        return action, {"Q": Q}

