import sys
import numpy as np
import copy
from random import random
from compression.utils.other import ActionManager, ObservationManager, getPANTHERparamsAsCppStruct, ExpertDidntSucceed
from colorama import init, Fore, Back, Style
import py_panther

######################################################################################################
###### https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# from contextlib import contextmanager,redirect_stderr,redirect_stdout
# from os import devnull
# @contextmanager
# def suppress_stdout_stderr():
#     """A context manager that redirects stdout and stderr to devnull"""
#     with open(devnull, 'w') as fnull:
#         with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#             yield (err, out)


# class DummyFile(object):
#     def write(self, x): pass

# @contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = DummyFile()
#     yield
#     sys.stdout = save_stdout
######################################################################################################

class ExpertPolicy(object):

    def __init__(self):
        self.am=ActionManager();
        self.om=ObservationManager();

        self.action_shape=self.am.getActionShape();
        self.observation_shape=self.om.getObservationShape();

        self.par=getPANTHERparamsAsCppStruct();

        self.my_SolverIpopt=py_panther.SolverIpopt(self.par);

        self.name=Style.BRIGHT+Fore.BLUE+"[Exp]"+Style.RESET_ALL
        self.reset()

    def printwithName(self,data):
        
        print(self.name+data)

    def printFailedOpt(self):
        print(self.name+" Called optimizer--> "+Style.BRIGHT+Fore.RED +"Failed"+ Style.RESET_ALL)

    def printSucessOpt(self):
        print(self.name+" Called optimizer--> "+Style.BRIGHT+Fore.GREEN +"Success"+ Style.RESET_ALL)


    def reset(self):
        pass

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.
    def predict(self, obs_n, deterministic=True):
        obs_n=obs_n.reshape(self.observation_shape) #Not sure why this is needed
        assert obs_n.shape==self.observation_shape, self.name+f"ERROR: obs.shape={obs_n.shape} but self.observation_shape={self.observation_shape}"
        
        # self.printwithName(f"Got Normalized obs={obs_n}")

        obs=self.om.denormalizeObservation(obs_n);

        # self.printwithName(f"Got obs={obs}")
        # self.printwithName(f"Got the following observation")
        # self.om.printObservation(obs)
        # self.printwithName(f"Got obs shape={obs.shape}")

        # self.om.printObs(obs)



        # ## Call the optimization
        init_state=py_panther.state(); #This is initialized as zero. This is A
        final_state=py_panther.state();#This is initialized as zero. This is G
        final_state.pos=self.om.getf_g(obs);

        total_time=self.par.factor_alloc*py_panther.getMinTimeDoubleIntegrator3DFromState(init_state, final_state, self.par.v_max, self.par.a_max)

        self.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
        self.my_SolverIpopt.setFocusOnObstacle(True);
        self.my_SolverIpopt.setObstaclesForOpt(self.om.getObstacles(obs));

        # with nostdout():
        succeed=self.my_SolverIpopt.optimize(True);

        

        if(succeed==False):
            self.printFailedOpt();
            raise ExpertDidntSucceed()
        else:
            self.printSucessOpt();

        best_solution=self.my_SolverIpopt.getBestSolution();

        # self.printwithName("Optimizer called, best solution= ")

        # best_solution.printInfo()

        action=self.am.solOrGuess2action(best_solution)

        action=self.am.normalizeAction(action)

        # self.printwithName("===================================================")
        
        Q=0.0; #Not used right now I think


        self.am.assertAction(action)

        return action, {"Q": Q}


        # #### End of call the optimization
        # self.printwithName("===================================================")
        # action = self.am.getRandomAction()
        # action = self.am.getDummyOptimalAction()

        # self.printwithName(f" Returned action={action}")
        # self.printwithName(f"Returned action shape={action.shape}")

        # action=self.am.normalizeAction(action)

        # action=self.am.getDummyOptimalNormalizedAction()