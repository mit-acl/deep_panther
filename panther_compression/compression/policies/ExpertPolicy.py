import sys
import numpy as np
import copy
from random import random, shuffle
from compression.utils.other import ActionManager, ObservationManager, getPANTHERparamsAsCppStruct, ExpertDidntSucceed, computeTotalTime
from colorama import init, Fore, Back, Style
import py_panther
import math 
from gym import spaces


###
from joblib import Parallel, delayed
import multiprocessing
##

class ExpertPolicy(object):


    #The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
    #See https://stackoverflow.com/a/68672/6057617
    #Note that, even though the class variables are not thread safe (see https://stackoverflow.com/a/1073230/6057617), we are using multiprocessing here, not multithreading
    #Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
    my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct())

    def __init__(self):
        self.am=ActionManager();
        self.om=ObservationManager();

        self.action_shape=self.am.getActionShape();
        self.observation_shape=self.om.getObservationShape();

        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)

        par=getPANTHERparamsAsCppStruct()
        self.par_v_max=par.v_max
        self.par_a_max=par.a_max
        self.par_factor_alloc=par.factor_alloc
        self.drone_extra_radius_for_NN=par.drone_extra_radius_for_NN

        # self.my_SolverIpopt=py_panther.SolverIpopt(self.par);

        self.name=Style.BRIGHT+Fore.BLUE+"  [Exp]"+Style.RESET_ALL
        self.reset()

    def printwithName(self,data):
        
        print(self.name+data)

    def printFailedOpt(self, info):
        print(self.name+" Called optimizer--> "+Style.BRIGHT+Fore.RED +"Failed"+ Style.RESET_ALL+". "+ info)

    def printSucessOpt(self, info):
        print(self.name+" Called optimizer--> "+Style.BRIGHT+Fore.GREEN +"Success"+ Style.RESET_ALL+". "+ info)


    def reset(self):
        pass

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.
    def predict(self, obs_n, deterministic=True):

        if(self.om.isNanObservation(obs_n)):
            return self.am.getNanAction(), {"Q": 0.0} 

        # print(f"self.observation_shape={self.observation_shape}")
        # obs_n=obs_n.reshape((-1,*self.observation_shape)) #Not sure why this is needed
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
        init_state=self.om.getInit_f_StateFromObservation(obs);        
        final_state=self.om.getFinal_f_StateFromObservation(obs);        

        # invsqrt3_vector=math.sqrt(3)*np.ones((3,1));
        # total_time=self.par.factor_alloc*py_panther.getMinTimeDoubleIntegrator3DFromState(init_state, final_state, self.par.v_max*invsqrt3_vector, self.par.a_max*invsqrt3_vector)
        total_time=computeTotalTime(init_state, final_state, self.par_v_max, self.par_a_max, self.par_factor_alloc)

        ExpertPolicy.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
        ExpertPolicy.my_SolverIpopt.setFocusOnObstacle(True);
        obstacles=self.om.getObstacles(obs)
        ####
        for i in range(len(obstacles)):
            obstacles[i].bbox_inflated = obstacles[i].bbox_inflated  +  self.drone_extra_radius_for_NN*np.ones_like(obstacles[i].bbox_inflated)
        ExpertPolicy.my_SolverIpopt.setObstaclesForOpt(obstacles);

        # with nostdout():
        succeed=ExpertPolicy.my_SolverIpopt.optimize(True);
        info=ExpertPolicy.my_SolverIpopt.getInfoLastOpt();

        if(succeed==False):
            self.printFailedOpt(info);
            # exit();
            # raise ExpertDidntSucceed()
            return self.am.getNanAction(), {"Q": 0.0} 
        else:
            self.printSucessOpt(info);

        best_solutions=ExpertPolicy.my_SolverIpopt.getBestSolutions();


        ##############################################################
        # if(self.am.use_closed_form_yaw_student==True):
        #     #Set the yaw elements to zero
        #     for i in range(len(best_solutions)):
        #         best_solutions[i].qy=[0] * len(best_solutions[i].qy)
        ###############################################################

        # for tmp in best_solutions:
        #     tmp.printInfo();

        #     print("YAW ctrl pts= ", tmp.qy)
        #     print("Total time= ", tmp.getTotalTime())


        # self.printwithName("Optimizer called, best solution= ")
        # best_solution.printInfo()
        # ###HACK
        # ctrl_pts=self.om.getCtrlPtsObstacleI(obs, 0);
        # indexes_elevation=list(range(len(best_solutions)))
        # shuffle(indexes_elevation)
        # print(f"indexes_elevation= {indexes_elevation}")
        # for i in range(len(best_solutions)):
        #     # print("\nbest_solutions[i].qp ANTES= ", best_solutions[i].qp)
        #     novale_tmp=[];
        #     tmp=(ctrl_pts[0] + indexes_elevation[i]*np.array([[0],[1],[0]])).reshape((3,))
        #     # print("Setting to ", tmp)
        #     for j in range(len(best_solutions[i].qp)):
        #         novale_tmp.append(tmp)
        #     best_solutions[i].qp=novale_tmp
        #     # print(f"[After]       best_solutions[i].qp={best_solutions[i].qp}")

        #     # print("\nbest_solutions[i].qp DESPUES= ", best_solutions[i].qp)
        # #######

        action=self.am.solsOrGuesses2action(best_solutions)

        # for i in range(action.shape[0]):
        #     print(self.am.getYawCtrlPts(action, i))
        #
        action_normalized=self.am.normalizeAction(action)

        # self.printwithName("===================================================")
        
        # self.printwithName(f"action_normalized= {action_normalized}")
        # self.printwithName(f"action= {action}")
        # self.printwithName(f"action=")
        # self.am.printAction(action)


        Q=0.0; #Not used right now I think


        self.am.assertAction(action_normalized)

        return action_normalized, {"Q": Q}


        # #### End of call the optimization
        # self.printwithName("===================================================")
        # action = self.am.getRandomAction()
        # action = self.am.getDummyOptimalAction()

        # self.printwithName(f" Returned action={action}")
        # self.printwithName(f"Returned action shape={action.shape}")

        # action=self.am.normalizeAction(action)

        # action=self.am.getDummyOptimalNormalizedAction()
    def predictSeveral(self, obs_n, deterministic=True):

        def my_func(thread_index):
            return self.predict( obs_n[thread_index,:,:], deterministic=deterministic)[0]

        num_jobs=min(multiprocessing.cpu_count(),len(obs_n)); #Note that the class variable my_SolverIpopt will be created once per job created (but only in the first call to predictSeveral I think)
        acts = Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(len(obs_n))))) #, prefer="threads"
        acts=np.stack(acts, axis=0)
        return acts

        #Other options: 
        # import multiprocessing_on_dill as multiprocessing
        # from pathos.multiprocessing import ProcessingPool as Pool

        # https://stackoverflow.com/a/1073230/6057617
        # pool=multiprocessing.Pool(12)
        # results=pool.map(my_func, list(range(num_jobs)))

        # pool=Pool(12)
        # acts=pool.map(my_func, list(range(num_jobs)))

        # acts=[self.predict( obs_n[i,:,:], deterministic=deterministic)[0] for i in range(len(obs_n))] #Note that len() returns the size alon the first axis