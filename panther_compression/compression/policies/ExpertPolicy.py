import sys
import numpy as np
import time
from statistics import mean
import copy
from random import random, shuffle
from compression.utils.other import ActionManager, ObservationManager, ObstaclesManager, getPANTHERparamsAsCppStruct, ExpertDidntSucceed, computeTotalTime
from colorama import init, Fore, Back, Style
import py_panther
import math 
from gym import spaces
from joblib import Parallel, delayed
import multiprocessing

class ExpertPolicy(object):

    #The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
    #See https://stackoverflow.com/a/68672/6057617
    #Note that, even though the class variables are not thread safe (see https://stackoverflow.com/a/1073230/6057617), we are using multiprocessing here, not multithreading
    #Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
    my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct())

    def __init__(self):
        self.am=ActionManager()
        self.om=ObservationManager()
        self.obsm=ObstaclesManager()

        self.action_shape=self.am.getActionShape()
        self.observation_shape=self.om.getObservationShape()

        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)

        par=getPANTHERparamsAsCppStruct()
        self.num_max_of_obst = par.num_max_of_obst
        self.par_v_max=par.v_max
        self.par_a_max=par.a_max
        self.par_factor_alloc=par.factor_alloc
        self.drone_extra_radius_for_NN=par.drone_extra_radius_for_NN

        # self.my_SolverIpopt=py_panther.SolverIpopt(self.par);

        self.computation_times_verbose = False

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

    def predict(self, obs_n, deterministic=True):
        """ From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        In the case of policies, the prediction is an action.
        In the case of critics, it is the estimated value of the observation."""

        if(self.om.isNanObservation(obs_n)):
            return self.am.getNanAction(), {"Q": 0.0} 
        obs=self.om.denormalizeObservation(obs_n)

        ##
        ## Replicate obstacles to meet num_max_of_obst defined in main.m
        ## Not sure why but if we have a fewer obstacles in training environment thatn num_max_of_obst, then we get error
        ## So we need to work this around by replicating obstacles
        ##

        while obs.shape[1] < self.om.getAgentObservationSize() + self.num_max_of_obst * self.obsm.getSizeEachObstacle():
            obs = np.append(obs, obs[0, self.om.getAgentObservationSize():self.om.getAgentObservationSize()+self.obsm.getSizeEachObstacle()])
            obs = np.reshape(obs, (1, obs.shape[0]))

        ##
        ## Call the optimization
        ##
        
        init_state=self.om.getInit_f_StateFromObservation(obs);        
        final_state=self.om.getFinal_f_StateFromObservation(obs);        
        total_time=computeTotalTime(init_state, final_state, self.par_v_max, self.par_a_max, self.par_factor_alloc)
        ExpertPolicy.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time)
        ExpertPolicy.my_SolverIpopt.setFocusOnObstacle(True)
        obstacles=self.om.getObstaclesForCasadi(obs)
        ExpertPolicy.my_SolverIpopt.setObstaclesForOpt(obstacles)

        start = time.time()
        succeed=ExpertPolicy.my_SolverIpopt.optimize(True)
        end = time.time()
        computation_time = end - start

        info=ExpertPolicy.my_SolverIpopt.getInfoLastOpt()

        ##
        ## Print results
        ##

        if not succeed:
            self.printFailedOpt(info)
            # print("ExpertDidntSucceed (note that optimizers success/fail prompt is not ordered with env. (because it's parallelized)")
            return self.am.getNanAction(), {"Q": 0.0}, computation_time
        else:
            self.printSucessOpt(info)

        best_solutions=ExpertPolicy.my_SolverIpopt.getBestSolutions()
        action=self.am.solsOrGuesses2action(best_solutions)
        action_normalized=self.am.normalizeAction(action)
        Q=0.0; #Not used right now I think
        self.am.assertAction(action_normalized)

        if self.computation_times_verbose:
            return action_normalized, {"Q": Q}, computation_time
        else:
            return action_normalized, {"Q": Q}

    def predictSeveral(self, obs_n, deterministic=True):

        # https://stackoverflow.com/a/68672/6057617

        def my_func(thread_index):
            return self.predict(obs_n[thread_index,:,:], deterministic=deterministic)[0]

        num_jobs=min(multiprocessing.cpu_count(),len(obs_n)); #Note that the class variable my_SolverIpopt will be created once per job created (but only in the first call to predictSeveral I think)

        acts=Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(len(obs_n))))) #, prefer="threads"
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

    def predictSeveralWithComputationTimeVerbose(self, obs_n, deterministic=True):

        self.computation_times_verbose = True #This is a hack. need to get rid of this

        def my_func(thread_index):
            return self.predict(obs_n[thread_index,:,:], deterministic=deterministic)

        num_jobs=min(multiprocessing.cpu_count(),len(obs_n)); #Note that the class variable my_SolverIpopt will be created once per job created (but only in the first call to predictSeveral I think)

        output = Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(len(obs_n))))) #, prefer="threads"
        # acts = [row[0] for row in output]
        # computation_times = [row[2] for row in output]
        
        acts = np.array(output)[:,0]

        if not self.computation_times_verbose:
            computation_times = np.array(output)[:,2]
            computation_times = np.stack(computation_times, axis=0)

        acts = np.stack(acts, axis=0)

        return acts, np.mean(computation_times)