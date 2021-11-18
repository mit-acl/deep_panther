from imitation.algorithms import bc
from compression.utils.other import ObservationManager,ActionManager, State, convertPPObstacles2Obstacles, convertPPState2State
import numpy as np
import copy 
import py_panther

class StudentCaller():
    def __init__(self, policy_path):
        self.student_policy=bc.reconstruct_policy(policy_path)
        self.om=ObservationManager();
        self.am=ActionManager();

    def predict(self, w_init_ppstate, w_ppobstacles, w_gterm): #pp stands for py_panther

        print("Hi there!")

        w_init_state=convertPPState2State(w_init_ppstate)

        w_gterm=w_gterm.reshape(3,1)

        w_obstacles=convertPPObstacles2Obstacles(w_ppobstacles)

        #Construct observation
        observation=self.om.construct_f_obsFrom_w_state_and_w_obs(w_init_state, w_obstacles, w_gterm)

        action,info = self.student_policy.predict(observation, deterministic=True) 

        action=action.reshape(self.am.getActionShape())

        print("action.shape= ", action.shape)
        print("action=", action)   


        # w_pos_ctrl_pts,_ = self.am.actionAndState2_w_pos_ctrl_pts_and_knots(action,w_init_state)

        # print("w_pos_ctrl_pts=", w_pos_ctrl_pts)

        my_solOrGuess= self.am.actionAndState2w_ppSolOrGuess(action,w_init_state);

        # py_panther.solOrGuessl


        return my_solOrGuess   