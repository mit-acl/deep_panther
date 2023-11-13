import numpy as np
import time
import torch as th

from stable_baselines3.common import utils as sb3_utils

from .ActionManager import ActionManager
from .ObservationManager import ObservationManager
from .utils import cast_gym_to_gymnasium, convertPPState2State, convertPPObstacles2Obstacles
from .CostComputer import CostComputer, ClosedFormYawSubstituter

class StudentCaller():
	def __init__(self, policy_path):
		# self.student_policy=bc.reconstruct_policy(policy_path)
		self.student_policy=policy = th.load(policy_path, map_location=sb3_utils.get_device("auto")) #Same as doing bc.reconstruct_policy(policy_path)
		self.student_policy.observation_space = cast_gym_to_gymnasium(self.student_policy.observation_space) # TODO: temporary fix
		self.om=ObservationManager();
		self.am=ActionManager();
		self.cc=CostComputer();
		self.cfys=ClosedFormYawSubstituter();

		# self.index_smallest_augmented_cost = 0
		# self.index_best_safe_traj = None
		# self.index_best_unsafe_traj = None

		self.costs_and_violations_of_action = None# CostsAndViolationsOfAction


	def predict(self, w_init_ppstate, w_ppobstacles, w_gterm): #pp stands for py_panther


		w_init_state=convertPPState2State(w_init_ppstate)

		w_gterm=w_gterm.reshape(3,1)

		w_obstacles=convertPPObstacles2Obstacles(w_ppobstacles)

		#Construct observation
		f_obs=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(w_init_state, w_gterm, w_obstacles)
		f_obs_n=self.om.normalizeObservation(f_obs)

		# print(f"Going to call student with this raw sobservation={f_obs}")
		# print(f"Which is...")
		# self.om.printObservation(f_obs)

		start = time.time()
		action_normalized,info = self.student_policy.predict(f_obs_n, deterministic=True) 

		action_normalized=action_normalized.reshape(self.am.getActionShape())

		#################################
		#### USE CLOSED FORM FOR YAW #####
		if(self.am.use_closed_form_yaw_student==True):
			action_normalized=self.cfys.substituteWithClosedFormYaw(action_normalized, w_init_state, w_obstacles) #f_action_n, w_init_state, w_obstacles
		##################################
		end = time.time()
		print(f" Calling the NN + Closed form yaw took {(end - start)*(1e3)} ms")

		action=self.am.denormalizeAction(action_normalized)

		# print("action.shape= ", action.shape)
		# print("action=", action)   

		all_solOrGuess=[]

		self.costs_and_violations_of_action=self.cc.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, action_normalized)

		self.index_best_safe_traj = None
		self.index_best_unsafe_traj = None
		for i in range( np.shape(action)[0]): #For each row of action
			traj=action[i,:].reshape(1,-1);

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state);

			my_solOrGuess.cost = self.costs_and_violations_of_action.costs[i]
			my_solOrGuess.obst_avoidance_violation = self.costs_and_violations_of_action.obst_avoidance_violations[i]
			my_solOrGuess.dyn_lim_violation = self.costs_and_violations_of_action.dyn_lim_violations[i]
			my_solOrGuess.aug_cost = self.cc.computeAugmentedCost(my_solOrGuess.cost, my_solOrGuess.obst_avoidance_violation, my_solOrGuess.dyn_lim_violation)

			all_solOrGuess.append(my_solOrGuess)


		return all_solOrGuess   

	def getIndexBestTraj(self):
		return self.costs_and_violations_of_action.index_best_traj
