import sys
import numpy as np
import copy
from random import random
from compression.utils.other import ActionManager, ObservationManager #, getPANTHERparamsAsCppStruct
from colorama import init, Fore, Back, Style
# import py_panther

class ExpertPolicy(object):

    def __init__(self):
        self.am=ActionManager();
        self.om=ObservationManager();

        self.action_shape=self.am.getActionShape();
        self.observation_shape=self.om.getObservationShape();

        # self.par=getPANTHERparamsAsCppStruct();

        # self.my_SolverIpopt=py_panther.SolverIpopt(self.par);


        self.reset()

    def printwithName(self,data):
        name=Style.BRIGHT+Fore.BLUE+"[Exp]"+Style.RESET_ALL
        print(name+data)

    def reset(self):
        pass

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.
    def predict(self, obs, deterministic=True):
        obs=obs.reshape(self.observation_shape) #Not sure why this is needed
        assert obs.shape==self.observation_shape, self.name+f"ERROR: obs.shape={obs.shape} but self.observation_shape={self.observation_shape}"
        
        obs=self.om.denormalizeObservation(obs);

        # self.printwithName(f"Got obs={obs}")
        # self.printwithName(f"Got obs shape={obs.shape}")

        # action=np.random.rand(1, self.mpc_act_size)

        # action_normalized = (1/self.max_act)*np.array([action]).reshape((1, self.mpc_act_size))

        # assert not np.isnan(np.sum(action_normalized)), f"trying to output nan"



        ## Call the optimization
        # init_state=py_panther.state();
        # init_state.pos=np.array([[-10], [0], [0]]);
        # init_state.vel=np.array([[0], [0], [0]]);
        # init_state.accel=np.array([[0], [0], [0]]);

        # final_state=py_panther.state();
        # final_state.pos=np.array([[10], [0], [0]]);
        # final_state.vel=np.array([[0], [0], [0]]);
        # final_state.accel=np.array([[0], [0], [0]]);


        # self.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, 2.0);
        # self.my_SolverIpopt.setFocusOnObstacle(True);

        # obstacle=py_panther.obstacleForOpt()
        # obstacle.bbox_inflated=np.array([[0.5],[0.5],[0.5]])

        # ctrl_pts=[]; #np.array([[],[],[]])
        # for i in range(int(self.par.fitter_num_seg + self.par.fitter_deg_pos)):
        #     print ("i=",i)
        #     ctrl_pts.append(np.array([[0],[0],[0]]))

        # obstacle.ctrl_pts = ctrl_pts

        # obstacles=[obstacle];

        # self.my_SolverIpopt.setObstaclesForOpt(obstacles);

        # self.my_SolverIpopt.optimize();
        #### End of call the optimization



        
        Q=random(); #This is the reward I think

        # action = self.am.getRandomAction()
        # action = self.am.getDummyOptimalAction()

        # self.printwithName(f" Returned action={action}")
        # self.printwithName(f"Returned action shape={action.shape}")

        # action=self.am.normalizeAction(action)

        action=self.am.getDummyOptimalNormalizedAction()

        assert action.shape==self.action_shape

        return action, {"Q": Q}

