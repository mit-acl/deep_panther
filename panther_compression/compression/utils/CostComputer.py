import numpy as np

import py_panther

from .ActionManager import ActionManager
from .ObservationManager import ObservationManager
from .ObstaclesManager import ObstaclesManager
from .MyClampedUniformBSpline import MyClampedUniformBSpline

from .yaml_utils import getPANTHERparamsAsCppStruct
from .utils import computeTotalTime, listOfNdVectors2numpyNXmatrix, numpyNXmatrixToListOfNdVectors

from joblib import Parallel, delayed
import multiprocessing

class CostsAndViolationsOfAction():
	def __init__(self, costs, obst_avoidance_violations, dyn_lim_violations, index_best_traj):
		self.costs=costs
		self.obst_avoidance_violations=obst_avoidance_violations
		self.dyn_lim_violations=dyn_lim_violations
		self.index_best_traj=index_best_traj

class CostComputer():

    #The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
    #See https://stackoverflow.com/a/68672/6057617
    #Note that, even though the class variables are not thread safe (see https://stackoverflow.com/a/1073230/6057617), we are using multiprocessing here, not multithreading
    #Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support

	my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct());
	
	par=getPANTHERparamsAsCppStruct();

	def __init__(self, dim=3, num_obs=1, additional_config=None):
		# self.par=getPANTHERparamsAsCppStruct();
		self.dim=dim
		self.am=ActionManager(dim=dim, additional_config=additional_config);
		self.om=ObservationManager(dim=dim, num_obs=num_obs, additional_config=additional_config);	
		self.obsm=ObstaclesManager(dim=dim, num_obs=num_obs, additional_config=additional_config);
		self.num_obstacles=self.obsm.getNumObs();

	def setUpSolverIpoptAndGetppSolOrGuess(self, f_obs_n, f_traj_n):

		#Denormalize observation and action
		f_obs = self.om.denormalizeObservation(f_obs_n);
		f_traj = self.am.denormalizeTraj(f_traj_n);

		#Set up SolverIpopt
		# print("\n========================")
		init_state=self.om.getInit_f_StateFromObservation(f_obs)
		final_state=self.om.getFinal_f_StateFromObservation(f_obs)
		total_time=computeTotalTime(init_state, final_state, CostComputer.par.v_max, CostComputer.par.a_max, CostComputer.par.factor_alloc)
		# print(f"init_state=")
		# init_state.printHorizontal();
		# print(f"final_state=")
		# final_state.printHorizontal();
		# print(f"total_time={total_time}")
		CostComputer.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
		CostComputer.my_SolverIpopt.setFocusOnObstacle(True);
		obstacles=self.om.getObstacles(f_obs)

		# print(f"obstacles=")

		# for obs in obstacles:
		# 	obs.printInfo()

		CostComputer.my_SolverIpopt.setObstaclesForOpt(obstacles);

		###############################
		f_state=self.om.get_f_StateFromf_obs(f_obs)
		f_ppSolOrGuess=self.am.f_trajAnd_f_State2f_ppSolOrGuess(f_traj, f_state)
		###############################

		return f_ppSolOrGuess;		


	def computeObstAvoidanceConstraintsViolation(self, f_obs_n, f_traj_n):

		#Denormalize observation and action
		f_obs = self.om.denormalizeObservation(f_obs_n);
		f_traj = self.am.denormalizeTraj(f_traj_n);

		total_time=self.am.getTotalTimeTraj(f_traj)

		###Debugging
		if(total_time<1e-5):
			print(f"total_time={total_time}")
			print(f"f_traj_n={f_traj_n}")
			print(f"f_traj={f_traj}")
		######

		f_state = self.om.get_f_StateFromf_obs(f_obs)
		f_posBS, f_yawBS = self.am.f_trajAnd_f_State2fBS(f_traj, f_state, no_deriv=True)

		violation=0


		for i in range(self.num_obstacles):
			f_posObs_ctrl_pts=listOfNdVectors2numpyNXmatrix(self.dim, self.om.getCtrlPtsObstacleI(f_obs, i))
			bbox=self.om.getBboxInflatedObstacleI(f_obs, i)
			# print(f"f_posObs_ctrl_pts={f_posObs_ctrl_pts}")
			# print(f"f_posBS.ctrl_pts={f_posBS.ctrl_pts}")

			# start=time.time();

			f_posObstBS = MyClampedUniformBSpline(0.0, CostComputer.par.fitter_total_time, CostComputer.par.fitter_deg_pos, self.dim, CostComputer.par.fitter_num_seg, f_posObs_ctrl_pts, True) 

			# print(f" compute MyClampedUniformBSpline creation took {(time.time() - start)*(1e3)} ms")


			# print("\n============")
			# start=time.time();

			#TODO: move num to a parameter
			for t in np.linspace(start=0.0, stop=total_time, num=15).tolist():

				obs = f_posObstBS.getPosT(t);
				drone = f_posBS.getPosT(t);
				
				obs_drone = drone - obs #position of the drone wrt the obstacle
				if np.all(abs(obs_drone[:, 0]) <= bbox[:, 0]/2):
					for i in range(obs_drone.shape[0]):
						obs_dronecoord = obs_drone[i, 0]
						tmp = bbox[i, 0] / 2
						violation += min(abs(tmp - obs_dronecoord), abs(obs_dronecoord - (-tmp)) )

		return violation

	def computeDynLimitsConstraintsViolation(self, f_obs_n, f_traj_n):

		f_ppSolOrGuess=self.setUpSolverIpoptAndGetppSolOrGuess(f_obs_n, f_traj_n)
		violation=CostComputer.my_SolverIpopt.computeDynLimitsConstraintsViolation(f_ppSolOrGuess)

		# #Debugging (when called using the traj from the expert)
		# if(violation>1e-5):
		# 	print("THERE IS VIOLATION in dyn lim")
		# 	exit()

		return violation   

	def computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(self, f_obs_n, f_traj_n):

		# start1=time.time();

		# start=time.time();
		cost =  self.computeCost(f_obs_n, f_traj_n)
		# print(f"--- computeCost took {(time.time() - start)*(1e3)} ms")
		
		# start=time.time();
		obst_avoidance_violation = self.computeObstAvoidanceConstraintsViolation(f_obs_n, f_traj_n)
		# print(f"violation: {obst_avoidance_violation}")
		# print(f"--- computeObstAvoidanceConstraintsViolation took {(time.time() - start)*(1e3)} ms")


		# start=time.time();
		dyn_lim_violation = self.computeDynLimitsConstraintsViolation(f_obs_n, f_traj_n)
		# print(f"--- computeDynLimitsConstraintsViolation took {(time.time() - start)*(1e3)} ms")

		# start=time.time();
		augmented_cost = self.computeAugmentedCost(cost, obst_avoidance_violation, dyn_lim_violation)
		# print(f" computeAugmentedCost took {(time.time() - start)*(1e3)} ms")

		
		# print(f" compute ALL COSTS {(time.time() - start1)*(1e3)} ms")


		return cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost

	def computeCost(self, f_obs_n, f_traj_n): 
		
		f_ppSolOrGuess=self.setUpSolverIpoptAndGetppSolOrGuess(f_obs_n, f_traj_n)
		tmp=CostComputer.my_SolverIpopt.computeCost(f_ppSolOrGuess)

		return tmp   

	# def computeAugmentedCost(self, f_obs_n, f_traj_n):
	# 	cost=self.computeCost(f_obs_n, f_traj_n)
	# 	obst_avoidance_violation=self.computeObstAvoidanceConstraintsViolation(f_obs_n, f_traj_n)
	# 	dyn_lim_violation=self.computeDynLimitsConstraintsViolation(f_obs_n, f_traj_n)

	# 	print(f"cost={cost}, obst_avoidance_violation={obst_avoidance_violation}, dyn_lim_violation={dyn_lim_violation}")

	# 	return cost + obst_avoidance_violation + dyn_lim_violation

	def computeAugmentedCost(self, cost, obst_avoidance_violation, dyn_lim_violation):
		return cost + CostComputer.par.lambda_obst_avoidance_violation*obst_avoidance_violation + CostComputer.par.lambda_dyn_lim_violation*dyn_lim_violation

	def getIndexBestTraj(self, f_obs_n, f_action_n):
		tmp=self.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, f_action_n)
		return tmp.index_best_traj
		# smallest_augmented_cost = float('inf')
		# index_smallest_augmented_cost = 0
		# for i in range(f_action_n.shape[0]):
		# 	f_traj_n = self.am.getTrajFromAction(f_action_n, i)

		# 	_, _, _, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_observation_n, traj_n)

		# 	# self.printwithNameAndColor(f"augmented cost traj_{i}={augmented_cost}")
		# 	if(augmented_cost < smallest_augmented_cost):
		# 		smallest_augmented_cost = augmented_cost
		# 		index_smallest_augmented_cost = i
		# return index_smallest_augmented_cost

	def getCostsAndViolationsOfActionFromObsAndAction(self, f_obs, f_action):
		f_obs_n = self.om.normalizeObservation(f_obs)
		f_action_n = self.am.normalizeAction(f_action)
		return self.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, f_action_n)

	def getCostsAndViolationsOfActionFromObsnAndActionn(self, f_obs_n, f_action_n):
		costs=[]
		obst_avoidance_violations=[]
		dyn_lim_violations=[]
		augmented_costs=[];

		alls=[];

		smallest_augmented_safe_cost = float('inf')
		smallest_augmented_unsafe_cost = float('inf')
		index_best_safe_traj = None
		index_best_unsafe_traj = None


		######### PARALLEL OPTION
		# start=time.time();
		def my_func(thread_index):
			traj_n=f_action_n[thread_index,:].reshape(1,-1);
			cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_obs_n, traj_n)
			return [cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost] 

		num_of_trajs=np.shape(f_action_n)[0]
		num_jobs=multiprocessing.cpu_count()#min(multiprocessing.cpu_count(),num_of_trajs); #Note that the class variable my_SolverIpopt will be created once per job created (but only in the first call to predictSeveral I think)
		alls = Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(num_of_trajs)))) #, prefer="threads"

		for i in range( np.shape(f_action_n)[0]): #For each row of action
			costs.append(alls[i][0])
			obst_avoidance_violations.append(alls[i][1])
			dyn_lim_violations.append(alls[i][2])
			augmented_costs.append(alls[i][3])

		# print(f" computeParallel took {(time.time() - start)*(1e3)} ms")
		##########################

		# start=time.time();

		for i in range( np.shape(f_action_n)[0]): #For each row of action

			######### NON-PARALLEL OPTION
			# traj_n=f_action_n[i,:].reshape(1,-1);
			# cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_obs_n, traj_n)
			# costs.append(cost)
			# obst_avoidance_violations.append(obst_avoidance_violation)
			# dyn_lim_violations.append(dyn_lim_violation)
			# augmented_costs.append(augmented_cost)
			##############################

			is_in_collision = (obst_avoidance_violations[i]>1e-5)

			if(is_in_collision==False):
				if(augmented_costs[i] < smallest_augmented_safe_cost):
					smallest_augmented_safe_cost = augmented_costs[i]
					index_best_safe_traj = i
			else:
				if(augmented_costs[i] < smallest_augmented_unsafe_cost):
					smallest_augmented_unsafe_cost = augmented_costs[i]
					index_best_unsafe_traj = i

		if(index_best_safe_traj is not None):
			# print(f"Choosing traj {index_best_safe_traj} ")
			index_best_traj = index_best_safe_traj

		elif(index_best_unsafe_traj is not None):
			# print(f"Choosing traj {index_best_unsafe_traj} ")
			index_best_traj= index_best_unsafe_traj
		else:
			print("This should never happen!!")
			exit();		

		# print(f" computeNOTParallel took {(time.time() - start)*(1e3)} ms")


		result=CostsAndViolationsOfAction(costs=costs, obst_avoidance_violations=obst_avoidance_violations, dyn_lim_violations=dyn_lim_violations, index_best_traj=index_best_traj)


		return result;

class ClosedFormYawSubstituter():
	def __init__(self, dim=3):
		self.dim = dim
		self.cy=py_panther.ClosedFormYawSolver();
		self.am=ActionManager(dim=dim);

	def substituteWithClosedFormYaw(self, f_action_n, w_init_state, w_obstacles):

		# print("In substituteWithClosedFormYaw")

		f_action=self.am.denormalizeAction(f_action_n)

		#Compute w_ppobstacles

		#####
		for i in range( np.shape(f_action)[0]): #For each row of action
			traj=f_action[i,:].reshape(1,-1);

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state);

			my_solOrGuess.qy=self.cy.getyCPsfrompCPSUsingClosedForm(
				my_solOrGuess.qp,
				my_solOrGuess.getTotalTime(),
				numpyNXmatrixToListOfNdVectors(w_obstacles[0].ctrl_pts),
				w_init_state.w_yaw,
				w_init_state.yaw_dot,
				0.0
			)

			tmp=np.array(my_solOrGuess.qy[2:-1])
			f_action[i,self.am.traj_size_pos_ctrl_pts:self.am.traj_size_pos_ctrl_pts+self.am.traj_size_yaw_ctrl_pts]=tmp  - w_init_state.w_yaw*np.ones(tmp.shape)#+ my_solOrGuess.qy[0]
			
			# all_solOrGuess.append(my_solOrGuess)

		f_action_n=self.am.normalizeAction(f_action) #Needed because we have modified action in the previous loop

		return f_action_n