import yaml
import os
import numpy as np
import pyquaternion
import math
from scipy.interpolate import BSpline
import py_panther
from colorama import init, Fore, Back, Style
# from imitation.algorithms.bc import bc
import random
import pytest
import time
import sys
import numpy.matlib
import torch as th
from stable_baselines3.common import utils
import geometry_msgs.msg
import tf.transformations 
# from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
from joblib import Parallel, delayed
import multiprocessing
import gym
import rospy

class ExpertDidntSucceed(Exception):
	  pass

class TfMatrix():
	def __init__(self, T):
		self.T=T
		##Debugging
		self.debug()
	def __mul__(self, data): #Overload multiplication operator
		if data.ndim==1:
			tmp=self.T@np.concatenate(data, np.array([1]))
			return tmp[0:3]
		elif data.ndim==2:
			# print("\ndata before=\n", data)
			#Apply the transformation matrix to each of the columns
			tmp=self.T@np.concatenate((data, np.ones((1, data.shape[1]))), axis=0)
			# print("\nusing T=\n", self.T)
			#self.debug()
			# print("\ndata after=\n", tmp)
			return tmp[0:3,:]
		else:
			raise NotImplementedError 
	def inv(self):
		return TfMatrix(np.linalg.inv(self.T))
	def debug(self):
		# print("=============DEBUGING INIT==========")
		R=self.T[0:3,0:3]
		# print("*****Using R=\n",R)
		# print("*****Using R@R.T=\n",R@R.T)
		np.testing.assert_allclose(R@R.T-np.identity(3), 0, atol=1e-07)
		np.testing.assert_allclose(self.T[3,:]-np.array([[0, 0, 0, 1]]), 0, atol=1e-07)
		# print("=============DEBUGING==========")
	def rot(self): #Rotational part
		return self.T[0:3,0:3]
	def translation(self): #Translational part
		return self.T[0:3,3].reshape(3,1)

def posAccelYaw2TfMatrix(w_pos, w_accel, yaw):
	axis_z=[0,0,1]

	##
	## Hopf fibration approach
	##

	thrust=w_accel + np.array([[0.0], [0.0], [9.81]])
	thrust_normalized=thrust/np.linalg.norm(thrust)

	a=thrust_normalized[0]
	b=thrust_normalized[1]
	c=thrust_normalized[2]

	tmp=(1/math.sqrt(2*(1+c)))
	q_w = tmp*(1+c) #w
	q_x = tmp*(-b)  #x
	q_y = tmp*(a)   #y
	q_z = 0         #z
	qabc=pyquaternion.Quaternion(q_w, q_x, q_y, q_z)  #Constructor is Quaternion(w,x,y,z), see http://kieranwynn.github.io/pyquaternion/#object-initialisation

	q_w = math.cos(yaw/2.0);  #w
	q_x = 0;                  #x 
	q_y = 0;                  #y
	q_z = math.sin(yaw/2.0);  #z
	qpsi=pyquaternion.Quaternion(q_w, q_x, q_y, q_z)  #Constructor is Quaternion(w,x,y,z)

	w_q_b=qabc * qpsi

	w_T_b = w_q_b.transformation_matrix
	
	w_T_b[0:3,3]=w_pos.flatten()

	return TfMatrix(w_T_b)

def getPANTHERparamsAsCppStruct():

	params_yaml=readPANTHERparams()
	params_yaml["b_T_c"]=np.array([[0, 0, 1, 0],
								  [-1, 0, 0, 0],
								  [0, -1, 0, 0],
								  [0, 0, 0, 1]])
	par=py_panther.parameters()
	for key in params_yaml:
		exec('%s = %s' % ('par.'+key, 'params_yaml["'+key+'"]')) #See https://stackoverflow.com/a/60487422/6057617 and https://www.pythonpool.com/python-string-to-variable-name/
	return par

def readPANTHERparams():

	params_yaml_1=[]
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../../../panther/param/panther.yaml', "r") as stream:
		try:
			params_yaml_1=yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	params_yaml_2=[]
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../../../panther/matlab/casadi_generated_files/params_casadi.yaml', "r") as stream:
		try:
			params_yaml_2=yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	# params_yaml = dict(params_yaml_1.items() + params_yaml_2.items()) #Doesn't work in Python 3
	params_yaml = {**params_yaml_1, **params_yaml_2}                        # NOTE: Python 3.5+ ONLY
	return params_yaml

class Obstacle():
	def __init__(self, ctrl_pts, bbox_inflated, is_obstacle=True):
		self.ctrl_pts=ctrl_pts
		self.bbox_inflated=bbox_inflated
		self.is_obstacle=is_obstacle

def convertPPObstacle2Obstacle(ppobstacle): #pp stands for py_panther
	assert type(ppobstacle.ctrl_pts) is list
	ctrl_pts=np.array([[],[],[]])
	for ctrl_pt in ppobstacle.ctrl_pts:
		ctrl_pts=np.concatenate((ctrl_pts, ctrl_pt.reshape(3,1)), axis=1)
	obstacle=Obstacle(ctrl_pts, ppobstacle.bbox_inflated.reshape(3,1))
	return obstacle

def convertPPObstacles2Obstacles(ppobstacles): #pp stands for py_panther
	obstacles=[]
	for ppobstacle in ppobstacles:
		obstacles.append(convertPPObstacle2Obstacle(ppobstacle))
	return obstacles

def convertPPState2State(ppstate):
	p=ppstate.pos.reshape(3,1)
	v=ppstate.vel.reshape(3,1)
	a=ppstate.accel.reshape(3,1)
	yaw=np.array([[ppstate.yaw]])
	dyaw=np.array([[ppstate.dyaw]])
	return State(p, v, a, yaw, dyaw)

##
##
## GTermManager
##
##

class GTermManager():
	def __init__(self):
		self.params = readPANTHERparams()
		self.x_max = self.params["training_env_x_max"]
		self.x_min = self.params["training_env_x_min"]
		self.y_max = self.params["training_env_y_max"]
		self.y_min = self.params["training_env_y_min"]
		self.z_max = self.params["training_env_z_max"]
		self.z_min = self.params["training_env_z_min"]
		self.goal_seen_radius = self.params["goal_seen_radius"]
		self.newRandomPos()

	def newRandomPos(self):
		self.w_gterm = np.array([
			[random.uniform(self.goal_seen_radius, self.x_max)],
			[0], # we will do YAWING in the beginning in actual sim/hw, so we will keep the Y coordinate fixed
			[random.uniform(self.z_min, self.z_max)]
		])

	def newRandomPosFarFrom_w_Position(self, w_position):
		dist=0.0
		while dist < self.goal_seen_radius: #Goal at least goal_seen_radius meters away from the current position
			self.newRandomPos()
			dist=np.linalg.norm(self.w_gterm-w_position)

	def setPos(self, pos):
		self.w_gterm=pos

	def get_w_GTermPos(self):
		return self.w_gterm

##
##
## ObstaclesManager
##
##

class ObstaclesManager():
	"""
	The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
	See https://stackoverflow.com/a/68672/6057617
	Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
	Note that pickle is needed when saving the Student Policy (which has a ObservationManager, which has a ObstaclesManager, which has a Fitter )
	"""	
	
	params=readPANTHERparams()
	fitter = py_panther.Fitter(params["fitter_num_samples"])

	def __init__(self):
		self.params = readPANTHERparams()
		self.random_num_of_obstacles_in_training = self.params["random_num_of_obstacles_in_training"]
		self.num_obs = self.params["num_of_obstacles_in_training"]
		self.x_max = self.params["training_env_x_max"]
		self.x_min = self.params["training_env_x_min"]
		self.y_max = self.params["training_env_y_max"]
		self.y_min = self.params["training_env_y_min"]
		self.z_max = self.params["training_env_z_max"]
		self.z_min = self.params["training_env_z_min"]
		self.goal_seen_radius = self.params["goal_seen_radius"]

		# self.fitter_total_time=params["fitter_total_time"]
		self.fitter_num_seg=self.params["fitter_num_seg"]
		self.fitter_deg_pos=self.params["fitter_deg_pos"]
		self.fitter_total_time=self.params["fitter_total_time"]
		self.fitter_num_samples=self.params["fitter_num_samples"]
		
		# self.fitter = py_panther.Fitter(self.fitter_num_samples)
		self.newRandomPos()

	def newRandomPos(self):

		##
		## if random_num_of_obstacles_in_training is True, then we will have a random number of obstacles
		##

		self.num_obs = random.randint(1, self.num_obs) if self.random_num_of_obstacles_in_training else self.num_obs

		##
		## create self.random_pos with the size of (3xself.num_obs)
		##

		self.random_pos = []
		for i in range(self.num_obs):
			self.random_pos.append(np.array([
				[random.uniform(self.x_min, self.x_max)],
				[random.uniform(self.y_min, self.y_max)],
				[random.uniform(self.z_min, self.z_max)]
			]))

		##
		## bbox size
		##

		if self.params["use_noised_obst_size"]:
			noised_obst_x = np.random.normal(loc=self.params["training_obst_size"][0], scale=0.2)
			noised_obst_y = np.random.normal(loc=self.params["training_obst_size"][1], scale=0.2)
			noised_obst_z = np.random.normal(loc=self.params["training_obst_size"][2], scale=0.2)
			self.bbox_inflated= np.array([noised_obst_x + self.params["drone_bbox"][0], noised_obst_y + self.params["drone_bbox"][1], noised_obst_z + self.params["drone_bbox"][2]])
		else:
			self.bbox_inflated= np.array(self.params["training_obst_size"]) + np.array(self.params["drone_bbox"])
		
		##
		## ideal params (got from sim_base_station.launch)
		##

		self.random_offset=0
		self.random_scale=np.array([[2],[2],[2]])

	def setPos(self, pos):
		self.random_pos=pos

	def getNumObs(self):
		# return self.num_obs
		return int(rospy.get_param("/SQ01s/panther/num_max_of_obst"))

	def getCPsPerObstacle(self):
		return self.fitter_num_seg + self.fitter_deg_pos # 7 + 3

	def getSizeAllObstacles(self):
		# Size of the ctrl_pts + bbox
		# return self.num_obs*(3*self.getCPsPerObstacle() + 3)
		return int(rospy.get_param("/SQ01s/panther/num_max_of_obst"))*(3*self.getCPsPerObstacle() + 3)

	def getSizeEachObstacle(self):
		return 3*self.getCPsPerObstacle() + 3

	def renewwObstaclePos(self):
		if np.random.uniform(0, 1) < 1 - self.params["prob_choose_cross"]:
			self.newRandomPos()
		else:
			w_pos_obstacle, _ = self.getObsAndGtermToCrossPath()
			self.setPos(w_pos_obstacle)

	def getObsAndGtermToCrossPath(self):

		w_pos_obstacle = []
		center = np.zeros((3,1))
		for i in range(self.num_obs):
			radius_obstacle_pos = random.uniform(self.goal_seen_radius*1.5, (self.x_max / 2))
			std_deg = 30
			theta_obs = random.uniform(-std_deg*np.pi/180, std_deg*np.pi/180) 
			height_g_term = random.uniform(self.params["training_env_z_min"], self.params["training_env_z_max"])
			height_obstacle = height_g_term + random.uniform(-0.25, 0.25)
			w_pos_obstacle.append(center + np.array([[radius_obstacle_pos*math.cos(theta_obs)],[radius_obstacle_pos*math.sin(theta_obs)],[height_obstacle]]))
		
		w_pos_g_term = center + np.array([[random.uniform((self.x_max / 2)*1.5, self.x_max)], [0], [random.uniform(self.z_min, self.z_max)]])

		return w_pos_obstacle, w_pos_g_term
	
	def getFutureWPosStaticObstacles(self):

		w_obs=[]
		for i in range(self.num_obs):
			w_ctrl_pts_ob=np.array([[],[],[]])
			for j in range(self.fitter_num_seg + self.fitter_deg_pos):
				w_ctrl_pts_ob=np.concatenate((w_ctrl_pts_ob, self.random_pos[i]), axis=1)
				# w_ctrl_pts_ob.append(np.array([[2],[2],[2]]))

			bbox_inflated= self.bbox_inflated
			w_obs.append(Obstacle(w_ctrl_pts_ob, bbox_inflated))

		return w_obs

	def getFutureWPosDynamicObstacles(self,t):

		w_obs=[]
		for i in range(self.num_obs):
			trefoil=Trefoil(pos=self.random_pos[i], scale=self.random_scale, offset=self.random_offset, slower=1.5)
			samples=[]
			for t_interm in np.linspace(t, t + self.fitter_total_time, num=self.fitter_num_samples):#.tolist():
				samples.append(trefoil.getPosT(t_interm))

			w_ctrl_pts_ob_list=ObstaclesManager.fitter.fit(samples)
			w_ctrl_pts_ob=listOf3dVectors2numpy3Xmatrix(w_ctrl_pts_ob_list)
			bbox_inflated= self.bbox_inflated
			w_obs.append(Obstacle(w_ctrl_pts_ob, bbox_inflated))

		return w_obs

##
##
## OtherAgentsManager
##
##

class OtherAgentsManager():
	"""
	Provide other agents trajectories in the training environment
	"""	
	params=readPANTHERparams()
	fitter = py_panther.Fitter(params["fitter_num_samples"])

	def __init__(self, policy):

		self.om = ObservationManager()
		self.am = ActionManager()
		self.cost_computer = CostComputer()

		self.params = readPANTHERparams()
		self.x_max = self.params["training_env_x_max"]
		self.x_min = self.params["training_env_x_min"]
		self.y_max = self.params["training_env_y_max"]
		self.y_min = self.params["training_env_y_min"]
		self.z_max = self.params["training_env_z_max"]
		self.z_min = self.params["training_env_z_min"]
		self.goal_radius = self.params["goal_radius"]
		self.goal_seen_radius = self.params["goal_seen_radius"]

		self.fitter_num_seg=self.params["fitter_num_seg"]
		self.fitter_deg_pos=self.params["fitter_deg_pos"]
		self.fitter_total_time=self.params["fitter_total_time"]
		self.fitter_num_samples=self.params["fitter_num_samples"]
	
		self.other_agent_bbox_inflated= np.array(self.params["training_other_agent_size"]) + np.array(self.params["drone_bbox"])
		self.policy = policy
		self.num_of_other_agents = self.params["num_of_other_agents_in_training"]

		ctrl_pts = self.am.get_zero_ctrl_pts_for_student()
		bbox_inflated = self.am.get_bbox_inflated_for_student()
		w_student = []
		for i in range(int(rospy.get_param("/SQ01s/panther/num_max_of_obst"))):
			w_student.append(Obstacle(ctrl_pts, bbox_inflated, is_obstacle=False))
		self.reset(w_student)

	def reset(self, w_obstacles):

		self.get_new_state()
		previous_oaf_obs = self.get_oaf_obs_from_w_obs(w_obstacles)
		self.previous_oaf_obs_n = self.om.normalizeObservation(previous_oaf_obs)

	def get_new_state(self):

		##
		## create self.w_state
		##

		self.w_pos = np.array([
			[random.uniform(self.x_min, self.x_max)],
			[random.uniform(self.y_min, self.y_max)],
			[random.uniform(self.z_min, self.z_max)]
		])

		self.w_vel = np.array([[0], [0], [0]])
		self.w_acc = np.array([[0], [0], [0]])
		self.w_yaw = np.array([[0]])
		self.w_dyaw = np.array([[0]])

		self.w_state = State(self.w_pos, self.w_vel, self.w_acc, self.w_yaw, self.w_dyaw)

		##
		## get terminal goal
		##

		dist = 0.0
		min_dist = 5.0
		while dist < min_dist:
			self.w_gterm_pos = np.array([
				[random.uniform(self.x_min, self.x_max)],
				[random.uniform(self.y_min, self.y_max)],
				[random.uniform(self.z_min, self.z_max)]
			])
			dist = np.linalg.norm(self.w_gterm_pos - self.w_pos)
	
	def update_state(self, oaf_traj, dt):

		##
		## update self.w_state
		##

		w_posBS, w_yawBS = self.am.f_trajAnd_w_State2wBS(oaf_traj, self.w_state)
		self.w_state = State(w_posBS.getPosT(dt), w_posBS.getVelT(dt), w_posBS.getAccelT(dt), w_yawBS.getPosT(dt), w_yawBS.getVelT(dt))

		##
		## goal reached check
		##

		dist_current_2gterm=np.linalg.norm(self.w_state.w_pos-self.w_gterm_pos) #From the current position to the goal
		dist_endtraj_2gterm=np.linalg.norm(w_posBS.getLastPos()-self.w_gterm_pos) #From the end of the current traj to the goal
	
		if dist_current_2gterm < self.goal_seen_radius and dist_endtraj_2gterm < self.goal_radius:
			ctrl_pts = self.am.get_zero_ctrl_pts_for_student()
			bbox_inflated = self.am.get_bbox_inflated_for_student()
			w_student = []
			for i in range(int(rospy.get_param("/SQ01s/panther/num_max_of_obst"))):
				w_student.append(Obstacle(ctrl_pts, bbox_inflated, is_obstacle=False))
			self.reset(w_student)
			print('[other agent] goal reached')
	
	def get_static_other_agents(self):

		##
		## create constrol points (repeated w_pos for 10 times)
		##

		ctrl_pts = np.tile(self.w_pos, (1, 10))
		bbox_inflated = self.other_agent_bbox_inflated
		w_obstacles = []
		for i in range(self.num_of_other_agents):
			w_obstacles.append(Obstacle(ctrl_pts, bbox_inflated, is_obstacle=False))
		return w_obstacles

	def getFutureWPosOtherAgents(self, time, w_obstacles_and_student, w_obstacles, dt):

		##
		## convert w_obstacles_and_student to f_obs in other agent frame (oaf)
		##
		
		oaf_obs = self.get_oaf_obs_from_w_obs(w_obstacles_and_student)
		oaf_obs_n = self.om.normalizeObservation(oaf_obs)

		##
		## get action from policy
		##

		oaf_action_n, _ = self.get_oaf_action_n_from_oaf_obs_n(oaf_obs_n)

		if self.am.isNanAction(oaf_action_n) or not self.am.actionIsNormalized(oaf_action_n):
			print(f"Nan action!")
			# return w_obstacles and num_of_other_agents
			w_other_agents = self.get_static_other_agents()
			w_obstacles_and_other_agents = w_obstacles[:]
			w_obstacles_and_other_agents.extend(w_other_agents)
			return w_obstacles_and_other_agents, self.num_of_other_agents

		##
		## denormalization
		##

		oaf_action = self.am.denormalizeAction(oaf_action_n)

		##
		## get w_obastacles_and_other_agents
		##

		index_smallest_augmented_cost = self.cost_computer.getIndexBestTraj(self.previous_oaf_obs_n, oaf_action_n)
		oaf_traj = self.am.getTrajFromAction(oaf_action, index_smallest_augmented_cost)
		w_other_agents = self.am.f_traj_and_w_state_2_w_student_for_other_agents(oaf_traj, self.w_state)
		w_obstacles_and_other_agents = w_obstacles[:]
		w_obstacles_and_other_agents.extend(w_other_agents)
		
		##
		## update previous_oaf_obs_n
		##

		self.previous_oaf_obs_n = oaf_obs_n

		##
		## update other agents state
		##

		self.update_state(oaf_traj, dt)

		# return w_obstacles and num_of_other_agents
		return w_obstacles_and_other_agents, self.num_of_other_agents
	
	def get_oaf_action_n_from_oaf_obs_n(self, oaf_obs_n):
		"""
		Convert oaf obstacles to oaf action
		"""
		print("call policy in other agent manager")
		oaf_action_n = self.policy.predict(oaf_obs_n)
		return oaf_action_n

	def get_oaf_obs_from_w_obs(self, w_obstacles):
		"""
		Convert world obstacles to oaf obstacles
		"""
		f_observation = self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.w_gterm_pos, w_obstacles)
		return f_observation

##
##
## Trefoil
##
##

class Trefoil():
	def __init__(self, pos, scale, offset, slower):
		self.x=pos[0,0]
		self.y=pos[1,0]
		self.z=pos[2,0]
		self.scale_x=scale[0,0]
		self.scale_y=scale[1,0]
		self.scale_z=scale[2,0]
		self.offset=offset
		self.slower=slower
		#slower=1.0; #The higher, the slower the obstacles move" 

	def getPosT(self,t):
		tt=t/self.slower

		x_trefoil=(self.scale_x/6.0)*(math.sin(tt+self.offset) + 2*math.sin(2*tt+self.offset)) + self.x
		y_trefoil=(self.scale_y/5.0)*(math.cos(tt+self.offset) - 2*math.cos(2*tt+self.offset)) + self.y
		z_trefoil=(self.scale_z/2.0)*(-math.sin(3*tt+self.offset)) + self.z

		return np.array([[x_trefoil], [y_trefoil], [z_trefoil]])

##
##
## State
##
##

class State():
	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
		assert w_pos.shape==(3,1)
		assert w_vel.shape==(3,1)
		assert w_accel.shape==(3,1)
		assert w_yaw.shape==(1,1)
		assert yaw_dot.shape==(1,1)
		self.w_pos = w_pos
		self.w_vel = w_vel
		self.w_accel = w_accel
		self.w_yaw = w_yaw
		self.yaw_dot = yaw_dot
		self.w_T_f= posAccelYaw2TfMatrix(self.w_pos, np.array([[0.0],[0.0],[0.0]]), w_yaw) #pos, accel, yaw
		ez=np.array([[0.0],[0.0],[1.0]])
		np.testing.assert_allclose(self.w_T_f.T[0:3,2].reshape(3,1)-ez, 0, atol=1e-07)
		self.f_T_w= self.w_T_f.inv()

	def f_pos(self):
		return self.f_T_w*self.w_pos

	def f_vel(self):
		f_vel=self.f_T_w.rot()@self.w_vel
		# assert (np.linalg.norm(f_vel)-np.linalg.norm(self.w_vel)) == pytest.approx(0.0), f"f_vel={f_vel} (norm={np.linalg.norm(f_vel)}), w_vel={self.w_vel} (norm={np.linalg.norm(self.w_vel)}), f_R_w={self.f_T_w.rot()}, "
		return f_vel

	def f_accel(self):
		self.f_T_w.debug()
		f_accel=self.f_T_w.rot()@self.w_accel
		# assert (np.linalg.norm(f_accel)-np.linalg.norm(self.w_accel)) == pytest.approx(0.0), f"f_accel={f_accel} (norm={np.linalg.norm(f_accel)}), w_accel={self.w_accel} (norm={np.linalg.norm(self.w_accel)}), f_R_w={self.f_T_w.rot()}, " 
		return f_accel

	def f_yaw(self):
		return np.array([[0.0]])
	
	def getW_T_b(self):
		return posAccelYaw2TfMatrix(self.w_pos, self.w_accel, self.w_yaw)

	def print_w_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In w frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw [deg], "+ \
		Fore.MAGENTA +f"dyaw [deg/s]: "+ \
		Fore.RED +f"{self.w_pos.T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.w_vel.T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.w_accel.T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.w_yaw*180/np.pi}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot*180/np.pi}"+Style.RESET_ALL)

	def print_f_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In f frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw [deg], "+ \
		Fore.MAGENTA +f"dyaw [deg/s]: "+ \
		Fore.RED +f"{self.f_pos().T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.f_vel().T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.f_accel().T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.f_yaw()*180/np.pi}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot*180/np.pi}"+Style.RESET_ALL)

def generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg):
	
	result= np.concatenate((t0*np.ones(deg), \
							np.linspace(t0, tf, num=num_seg+1),\
							tf*np.ones(deg)))

	return 	result

##
##
## MyClampedUniformBSpline
##
##

class MyClampedUniformBSpline():
	def __init__(self,t0, tf, deg, dim, num_seg, ctrl_pts, no_deriv=False):

		assert dim==ctrl_pts.shape[0]

		deg=int(deg)
		dim=int(dim)
		num_seg=int(num_seg)

		self.pos_bs=[]; #BSpline of all the coordinates
		if(deg>=1):
			self.vel_bs=[]; #BSpline of all the coordinates
		if(deg>=2):
			self.accel_bs=[]; #BSpline of all the coordinates
		if(deg>=3):
			self.jerk_bs=[]; #BSpline of all the coordinates
		self.deg=deg
		self.num_seg=num_seg
		self.dim=dim
		self.knots=generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg)

		##
		## Debugging
		##

		if (abs(tf-t0)<1e-5):
			print(f"t0={t0}, tf={tf}, deg={deg}, num_seg={num_seg}")
			print(f"self.knots={self.knots}")

		##
		## Debugging End
		##

		self.ctrl_pts=ctrl_pts
		for i in range(dim):
			self.pos_bs.append( BSpline(self.knots, self.ctrl_pts[i,:], self.deg) )
			if(no_deriv==False):
				if(deg>=1):
					self.vel_bs.append( self.pos_bs[i].derivative(1) ); #BSpline of all the coordinates
				if(deg>=2):
					self.accel_bs.append( self.pos_bs[i].derivative(2) ); #BSpline of all the coordinates
				if(deg>=3):
					self.jerk_bs.append( self.pos_bs[i].derivative(3) ); #BSpline of all the coordinates

	def getPosT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.pos_bs[i](t)
		return result
		
	def getVelT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.vel_bs[i](t)
		return result

	def getAccelT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.accel_bs[i](t)
		return result

	def getJerkT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.jerk_bs[i](t)
		return result

	def getLastPos(self):
		result1=self.ctrl_pts[0:3,-1].reshape(self.dim,1)
		result2=self.getPosT(self.knots[-1])
		np.testing.assert_allclose(result1-result2, 0, atol=1e-07)
		return result1

	def getT0(self):
		return self.knots[0]

	def getTf(self):
		return self.knots[-1]

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		raise RuntimeError("The norm is zero")
	return v / norm

def computeTotalTime(init_state, final_state, par_vmax, par_amax, par_factor_alloc):
	# invsqrt3_vector=math.sqrt(3)*np.ones((3,1))
	total_time=par_factor_alloc*py_panther.getMinTimeDoubleIntegrator3DFromState(init_state, final_state, par_vmax, par_amax)
	return total_time

def wrapInmPiPi(data): #https://stackoverflow.com/a/15927914
	"""
	Wraps an angle in [-pi, pi)
	Works also for np arrays
	"""
	return (data + np.pi) % (2 * np.pi) - np.pi

##
##
## ObservationManager
##
##

class ObservationManager():
	def __init__(self):
		self.obsm=ObstaclesManager()
		
		#Observation = [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
		# Where f_ctrl_pts_oi = [cp0.transpose(), cp1.transpose(), ...]
		self.observation_size= 3 + 3 + 1 + 3 + self.obsm.getSizeAllObstacles()

		params=readPANTHERparams()
		self.v_max=np.array(params["v_max"]).reshape(3,1)
		self.a_max=np.array(params["a_max"]).reshape(3,1)
		self.j_max=np.array(params["j_max"]).reshape(3,1)
		self.ydot_max=params["ydot_max"]
		# self.max_dist2goal=params["max_dist2goal"]
		self.max_dist2obs=params["max_dist2obs"]
		self.max_side_bbox_obs=params["max_side_bbox_obs"]
		self.Ra=params["Ra"]
		self.num_max_of_obst = int(rospy.get_param("/SQ01s/panther/num_of_trajs_per_replan")) # from casadi
		self.use_lstm = params["use_lstm"] 
		
		ones13=np.ones((1,3))
		#Note that the sqrt(3) is needed because the expert/student plan in f_frame --> bouding ball around the box v_max, a_max,... 
		margin_v_factor = params["margin_v_factor"]
		margin_a_factor = params["margin_a_factor"]
		margin_ydot_factor = params["margin_ydot_factor"]
		margin_v=margin_v_factor * math.sqrt(3) #math.sqrt(3)
		margin_a=margin_a_factor * math.sqrt(3) #math.sqrt(3)
		margin_ydot=margin_ydot_factor

		# for agent's own state
		self.normalization_constant=np.concatenate((margin_v*self.v_max.T*ones13, margin_a*self.a_max.T*ones13, margin_ydot*self.ydot_max*np.ones((1,1)), self.Ra*ones13), axis=1)
		
		# for obstacles
		self.normalization_constant=np.concatenate((self.normalization_constant, self.max_dist2obs*np.ones((1,3*self.obsm.getCPsPerObstacle())), self.max_side_bbox_obs*ones13), axis=1)
		
		# self.normalization_constant is tiled in self.getNormalizationVector() for each obstacle depending on the number of obstacles, so no need to do this anymore
		# for i in range(self.obsm.getNumObs()):
		# 	self.normalization_constant=np.concatenate((self.normalization_constant, self.max_dist2obs*np.ones((1,3*self.obsm.getCPsPerObstacle())), self.max_side_bbox_obs*ones13), axis=1)

		# assert print("Shape observation=", observation.shape==)

	def randomVel(self):
		return np.random.uniform(-self.v_max,self.v_max)

	def randomAccel(self):
		return np.random.uniform(-self.a_max,self.a_max)

	def randomYdot(self):
		return np.random.uniform(-self.ydot_max,self.ydot_max, size=(1,1))

	def randomYaw(self):
		return wrapInmPiPi(np.random.uniform(-np.pi,np.pi, size=(1,1)))

	def obsIsNormalized(self, observation_normalized):
		assert observation_normalized.shape == self.getObservationShape()

		"""
		check which elements are not in [-1,1]
		if first 3 elements are not in [-1,1] --> f_v is not normalized 
		if 4th to 6th elements are not in [-1,1] --> f_a is not normalized 
		if 7th element is not in [-1,1] --> yaw_dot is not normalized
		if 8th to 10th elements are not in [-1,1] --> f_g (the goal) is not normalized
		if 11th to end elements are not in [-1,1] --> obstacle is not normalized
		"""

		if not np.logical_and(observation_normalized[0][0:3] >= -1, observation_normalized[0][0:3] <= 1).all():
			print(Fore.GREEN + "	f_v is not normalized" + Style.RESET_ALL)
			return False
		elif not np.logical_and(observation_normalized[0][3:6] >= -1, observation_normalized[0][3:6] <= 1).all():
			print(Fore.GREEN + "	f_a is not normalized" + Style.RESET_ALL)
			return False
		elif not np.logical_and(observation_normalized[0][6] >= -1, observation_normalized[0][6] <= 1):
			print(Fore.GREEN + "	yaw_dot is not normalized" + Style.RESET_ALL)
			print("yaw_dot=", observation_normalized[0][6])
			return False
		elif not np.logical_and(observation_normalized[0][7:10] >= -1, observation_normalized[0][7:10] <= 1).all():
			print(Fore.GREEN + "	f_g is not normalized" + Style.RESET_ALL)
			return False
		elif not np.logical_and(observation_normalized[0][10:] >= -1, observation_normalized[0][10:] <= 1).all():
			print(Fore.GREEN + "	obstacle is not normalized" + Style.RESET_ALL)
			return False
		else:
			return True
		# return np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()
	
	def obsIsNormalizedWithVerbose(self, observation_normalized):
		assert observation_normalized.shape == self.getObservationShape()

		"""
		check which elements are not in [-1,1]
		if first 3 elements are not in [-1,1] --> f_v is not normalized 
		if 4th to 6th elements are not in [-1,1] --> f_a is not normalized 
		if 7th element is not in [-1,1] --> yaw_dot is not normalized
		if 8th to 10th elements are not in [-1,1] --> f_g (the goal) is not normalized
		if 11th to end elements are not in [-1,1] --> obstacle is not normalized
		"""

		is_normalized = True
		which_dyn_limit_violated = []

		if not np.logical_and(observation_normalized[0][0:3] >= -1, observation_normalized[0][0:3] <= 1).all():
			print(Fore.GREEN + "	f_v is not normalized" + Style.RESET_ALL)
			is_normalized = False
			which_dyn_limit_violated.append("f_v")
		elif not np.logical_and(observation_normalized[0][3:6] >= -1, observation_normalized[0][3:6] <= 1).all():
			print(Fore.GREEN + "	f_a is not normalized" + Style.RESET_ALL)
			is_normalized = False
			which_dyn_limit_violated.append("f_a")
		elif not np.logical_and(observation_normalized[0][6] >= -1, observation_normalized[0][6] <= 1):
			print(Fore.GREEN + "	yaw_dot is not normalized" + Style.RESET_ALL)
			print("yaw_dot=", observation_normalized[0][6])
			is_normalized = False
			which_dyn_limit_violated.append("yaw_dot")
		elif not np.logical_and(observation_normalized[0][7:10] >= -1, observation_normalized[0][7:10] <= 1).all():
			print(Fore.GREEN + "	f_g is not normalized" + Style.RESET_ALL)
			is_normalized = False
			which_dyn_limit_violated.append("f_g")
		elif not np.logical_and(observation_normalized[0][10:] >= -1, observation_normalized[0][10:] <= 1).all():
			print(Fore.GREEN + "	obstacle is not normalized" + Style.RESET_ALL)
			is_normalized = False
			which_dyn_limit_violated.append("obstacle")
		else:
			is_normalized = True
		
		return is_normalized, which_dyn_limit_violated
		# return np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()

	def assertObsIsNormalized(self, observation_normalized, msg_before=""):

		if not self.obsIsNormalized(observation_normalized):
			print(msg_before+"The observation is not normalized")
			print(f"NORMALIZED VALUE={observation_normalized}")
			observation=self.denormalizeObservation(observation_normalized)
			print(f"VALUE={observation}")
			self.printObservation(observation)
			raise AssertionError()

	def printObservation(self, obs):
		print("----The observation is:")
		print(Fore.BLUE +f"f_v.T={self.getf_v(obs).T}"+Style.RESET_ALL)
		print(Fore.GREEN +f"f_a.T={self.getf_a(obs).T}"+Style.RESET_ALL)
		print(Fore.MAGENTA +f"yaw_dot={self.getyaw_dot(obs)}"+Style.RESET_ALL)
		print(Fore.CYAN +f"f_g.T={self.getf_g(obs).T}"+Style.RESET_ALL)

		for i in range(self.obsm.getNumObs()):
			print(f"Obstacle {i}:")
			ctrl_pts=self.getCtrlPtsObstacleI(obs,i) 
			bbox_inflated=self.getBboxInflatedObstacleI(obs,i)

			print(Fore.YELLOW +f"ctrl_pts={ctrl_pts}"+Style.RESET_ALL)
			print(Fore.BLUE +f"bbox_inflated.T={bbox_inflated.T}"+Style.RESET_ALL)

		print("----------------------")

	def get_f_StateFromf_obs(self,f_obs):
		pos=np.zeros((3,1))
		vel=self.getf_v(f_obs)
		accel=self.getf_a(f_obs)
		yaw=np.zeros((1,1))
		yaw_dot=self.getyaw_dot(f_obs)
		state=State(pos, vel, accel, yaw, yaw_dot)
		return state

	def getIndexStartObstacleI(self,i):
		return 10+(3*self.obsm.getCPsPerObstacle()+3)*i

	def getCtrlPtsObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)
		ctrl_pts=[]
		num_cps_per_obstacle=self.obsm.getCPsPerObstacle()
		for j in range(num_cps_per_obstacle):
			index_start_cpoint=index_start_obstacle_i+3*j
			cpoint_j=obs[0,index_start_cpoint:index_start_cpoint+3].reshape(3,1)
			ctrl_pts.append(cpoint_j)

		return ctrl_pts

	def getBboxInflatedObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)
		tmp=index_start_obstacle_i+(3*self.obsm.getCPsPerObstacle())
		bbox_inflated = obs[0,tmp:tmp+3].reshape(3,1)
		return bbox_inflated

	def getf_v(self, obs):
		return obs[0,0:3].reshape((3,1)) #Column vector

	def getf_a(self, obs):
		return obs[0,3:6].reshape((3,1)) #Column vector

	def getyaw_dot(self, obs):
		return obs[0,6].reshape((1,1)) 

	def getf_g(self, obs):
		return obs[0,7:10].reshape((3,1)) #Column vector

	def getObstacles(self, obs):
		obstacles=[]
		num_obs = int((obs.shape[1]-10)/(3*self.obsm.getCPsPerObstacle()+3))
		for i in range(num_obs):
			ctrl_pts=self.getCtrlPtsObstacleI(obs,i)
			bbox_inflated=self.getBboxInflatedObstacleI(obs,i)
			obstacle=py_panther.obstacleForOpt()
			obstacle.ctrl_pts=ctrl_pts
			obstacle.bbox_inflated=bbox_inflated
			obstacles.append(obstacle)
		return obstacles

	def getObstaclesForCasadi(self, obs):
		obstacles=[]
		num_obs = self.calculateObstacleNumber(obs)
		for i in range(num_obs): # num_max_of_obst is Casadi
			ctrl_pts=self.getCtrlPtsObstacleI(obs,i) 
			bbox_inflated=self.getBboxInflatedObstacleI(obs,i)
			obstacle=py_panther.obstacleForOpt()
			obstacle.ctrl_pts=ctrl_pts
			obstacle.bbox_inflated=bbox_inflated
			obstacles.append(obstacle)
		return obstacles

	def getInit_f_StateFromObservation(self, obs):
		init_state=py_panther.state()  #Everything initialized as zero
		init_state.pos= np.array([[0.0],[0.0],[0.0]]) #Because it's in f frame
		init_state.vel= self.getf_v(obs)
		init_state.accel= self.getf_a(obs)
		init_state.yaw= 0.0  #Because it's in f frame
		init_state.dyaw = self.getyaw_dot(obs)
		return init_state

	def getFinal_f_StateFromObservation(self, obs):
		final_state=py_panther.state()  #Everything initialized as zero
		final_state.pos= self.getf_g(obs)
		# final_state.vel= 
		# final_state.accel= 
		# final_state.yaw= 
		# final_state.dyaw = 
		return final_state

	def getNanObservation(self):
		return np.full(self.getObservationShape(), np.nan)

	def isNanObservation(self, obs):
		return np.isnan(np.sum(obs))

	def calculateObstacleNumber(self, obs):
		return int((obs.shape[1] - 10) / (3*self.obsm.getCPsPerObstacle()+3))

	def getNormalizationVector(self, obs):
		num_obs = self.calculateObstacleNumber(obs)
		normalization_vector = self.normalization_constant
		for i in range(num_obs-1):
			normalization_vector = np.concatenate((normalization_vector, self.normalization_constant[0, 10:].reshape(1, -1)), axis=1)
		return normalization_vector
	
	def normalizeObservation(self, observation):
		""" Normalize in [-1,1] """
		observation_normalized = observation / self.getNormalizationVector(observation)
		# assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()
		return observation_normalized
	
	def normalizeObservationWithNumObstAndOtherAgents(self, observation, num_obst, num_other_agents):

		normalization_vector = self.normalization_constant
		for i in range(num_obst-1):
			normalization_vector = np.concatenate((normalization_vector, self.normalization_constant[0, 10:].reshape(1, -1)), axis=1)

		ones13=np.ones((1,3))
		for i in range(num_other_agents):
			normalization_vector = np.concatenate((normalization_vector, self.max_dist2obs*np.ones((1,3*10)), self.max_side_bbox_obs*ones13), axis=1)

		observation_normalized = observation /normalization_vector

		return observation_normalized

	def denormalizeObservation(self,observation_normalized):
		""" Denormalize from [-1,1] to original range """
		observation=observation_normalized*self.getNormalizationVector(observation_normalized)
		return observation

	def getObservationSize(self):
		return self.observation_size

	def getAgentObservationSize(self):
		return self.observation_size - self.obsm.getSizeAllObstacles()

	def getObservationShape(self):
		return (1,self.observation_size)


	def getRandomObservation(self):
		random_observation=self.denormalizeObservation(self.getRandomNormalizedObservation())
		return random_observation

	def getRandomNormalizedObservation(self):
		random_normalized_observation=np.random.uniform(-1,1, size=self.getObservationShape())
		return random_normalized_observation

	def get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self, w_state, w_gterm_pos, w_obstacles):

		f_gterm_pos=w_state.f_T_w * w_gterm_pos
		dist2gterm=np.linalg.norm(f_gterm_pos)
		f_g= min(dist2gterm-1e-4, self.Ra)*normalize(f_gterm_pos)
		observation=np.concatenate((w_state.f_vel().flatten(), w_state.f_accel().flatten(), w_state.yaw_dot.flatten(), f_g.flatten()))
		
		##
		## Convert obs to f frame and append them to observation
		##

		for w_obstacle in w_obstacles:
			assert type(w_obstacle.ctrl_pts).__module__ == np.__name__, "the ctrl_pts should be a numpy matrix, not a list"
			observation=np.concatenate((observation, (w_state.f_T_w*w_obstacle.ctrl_pts).flatten(order='F'), (w_obstacle.bbox_inflated).flatten()))
		
		##
		## Reshape observation
		##

		observation=observation.reshape((1,-1))
		return observation

	def getNormalized_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self, w_state, w_gterm_pos, w_obstacles):
		f_observation=self.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(w_state, w_gterm_pos, w_obstacles)
		f_observationn=self.normalizeObservation(f_observation) #Observation normalized
		return f_observationn

def numpy3XmatrixToListOf3dVectors(data):
	data_list=[]
	for i in range(data.shape[1]):
		data_list.append(data[:,i])
	return data_list

def listOf3dVectors2numpy3Xmatrix(data_list):
	data_matrix=np.empty((3, len(data_list)))
	for i in range(len(data_list)):
		data_matrix[:,i]=data_list[i].reshape(3,)
	return data_matrix

def getZeroState():
	p0=np.array([[0.0],[0.0],[0.0]])
	v0=np.array([[0.0],[0.0],[0.0]])
	a0=np.array([[0.0],[0.0],[0.0]])
	y0=np.array([[0.0]])
	ydot0=np.array([[0.0]])
	zero_state= State(p0, v0, a0, y0, ydot0)

	return zero_state

##
##
## ActionManager
##
##

class ActionManager():
	def __init__(self):
		params=readPANTHERparams()
		self.deg_pos=params["deg_pos"]
		self.deg_yaw=params["deg_yaw"]
		self.num_seg=params["num_seg"]
		self.use_closed_form_yaw_student=params["use_closed_form_yaw_student"]
		self.make_yaw_NN=params["make_yaw_NN"]
		self.margin_yaw_factor=params["margin_yaw_factor"]

		""" Define action and observation space

		 action = np.array([ctrl_points_pos   ctrl_points_yaw  total time, prob that traj])
		
		 --> where ctrlpoints_pos has the ctrlpoints that are not determined by init pos/vel/accel, and final vel/accel
		     i.e., len(ctrlpoints_pos)= (num_seg_pos + deg_pos - 1 + 1) - 3 - 2 = num_seg_pos + deg_pos - 5
			  and ctrl_points_pos=[cp3.transpose(), cp4.transpose(),...]
		
		 --> where ctrlpoints_yaw has the ctrlpoints that are not determined by init pos/vel, and final vel
		     i.e., len(ctrlpoints_yaw)= (num_seg_yaw + deg_yaw - 1 + 1) - 2 - 1 = num_seg_yaw + deg_yaw - 3 """

		self.num_traj_per_action=int(rospy.get_param("/SQ01s/panther/num_of_trajs_per_replan")) # TODO: SQ01s dependency
		self.total_num_pos_ctrl_pts = self.num_seg + self.deg_pos
		self.traj_size_pos_ctrl_pts = 3*(self.total_num_pos_ctrl_pts - 5)
		self.traj_size_yaw_ctrl_pts = (self.num_seg + self.deg_yaw - 3)
		# self.traj_size = self.traj_size_pos_ctrl_pts + self.traj_size_yaw_ctrl_pts + 1 + 1; # Last two numbers are time and prob that traj is real
		self.traj_size = self.traj_size_pos_ctrl_pts + self.traj_size_yaw_ctrl_pts + 1; # Last number is time
		self.action_size = self.num_traj_per_action*self.traj_size
		self.Npos = self.num_seg + self.deg_pos-1

		self.max_dist2BSPoscPoint=params["max_dist2BSPoscPoint"]
		self.max_yawcPoint=self.margin_yaw_factor*2.0*math.pi # not sure why but this was 4e3*math.pi before (Kota's change)
		self.fitter_total_time=params["fitter_total_time"]

		self.training_other_agent_size = params["training_other_agent_size"]
		self.drone_bbox = params["drone_bbox"]

		# print("self.max_dist2BSPoscPoint= ", self.max_dist2BSPoscPoint)
		# print("self.max_yawcPoint= ", self.max_yawcPoint)
		self.normalization_constant_traj=np.concatenate((self.max_dist2BSPoscPoint*np.ones((1, self.traj_size_pos_ctrl_pts)), \
													self.max_yawcPoint*np.ones((1, self.traj_size_yaw_ctrl_pts))), axis=1)

		self.normalization_constant=np.matlib.repmat(self.normalization_constant_traj, self.num_traj_per_action, 1)

	def actionIsNormalized(self, action_normalized):
		assert action_normalized.shape == self.getActionShape()
		return np.logical_and(action_normalized >= -1, action_normalized <= 1).all()

	def assertActionIsNormalized(self, action_normalized, msg_before=""):
		if not self.actionIsNormalized(action_normalized):
			print(msg_before+"The action is not normalized")
			print(f"NORMALIZED VALUE={action_normalized}")
			action=self.denormalizeAction(action_normalized)
			print(f"VALUE={action}")
			# self.printObservation(observation);
			raise AssertionError()

	def assertAction(self,action):
		assert action.shape==self.getActionShape(), f"[Env] ERROR: action.shape={action.shape} but should be={self.getActionShape()}"
		assert not np.isnan(np.sum(action)), f"Action has nan"

	def getDummyOptimalNormalizedAction(self):
		action=self.getDummyOptimalAction()
		return self.normalizeAction(action)

	def getDummyOptimalAction(self):
		# return 0.6*np.ones(self.getActionShape())

		dummy=np.ones((self.num_traj_per_action,self.traj_size))

		for i in range((dummy.shape[0])):
			for j in range(0,self.traj_size_pos_ctrl_pts,3):
				dummy[i,j]=i+j/10

		return dummy

	def getNanAction(self):
		return np.full(self.getActionShape(), np.nan)

	def isNanAction(self, act):
		# return np.isnan(np.sum(act))
		return np.isnan(act).any()

	def normalizeAction(self, action):
		action_normalized=np.empty(action.shape)
  
		action_normalized[:,0:-1]=action[:,0:-1]/self.normalization_constant #Elementwise division
		action_normalized[:,-1]=(2.0/self.fitter_total_time)*action[:,-1]-1 #Note that action[0,-1] is in [0, fitter_total_time]
		# action_normalized[:,-1]=(2.0/1.0)*action[:,-1]-1 #Note that action[0,-1] is in [0, 1]

		for index_traj in range(self.num_traj_per_action):
			time_normalized=self.getTotalTime(action_normalized, index_traj)
			slack=1-abs(time_normalized)
			if(slack<0):
				if abs(slack)<1e-5: #Can happen due to the tolerances in the optimization
					# print(f"Before= {action_normalized[0,-1]}")
					action_normalized[index_traj,-1]=np.clip(time_normalized, -1.0, 1.0) #Saturate within limits
					# print(f"After= {action_normalized[0,-1]}")
				else:
					# assert False, f"time_normalized={time_normalized}"
					print(Fore.GREEN + "time is not normalized" + Style.RESET_ALL)

		# assert np.logical_and(action_normalized >= -1, action_normalized <= 1).all(), f"action_normalized={action_normalized}, last element={action_normalized[0,-1]}"
		return action_normalized

	# def getProb(self,action, index_traj):
	# 	return action[index_traj,-1]

	def getTotalTime(self,action, index_traj):
		return action[index_traj,-1]

	def getPosCtrlPts(self, action, index_traj):
		return action[index_traj,0:self.traj_size_pos_ctrl_pts].reshape((3,-1), order='F')

	def getYawCtrlPts(self, action, index_traj):
		return action[index_traj,self.traj_size_pos_ctrl_pts:-1].reshape((1,-1))

	# def getProbTraj(self,traj):
	# 	return self.getProb(traj, 0)

	def getTotalTimeTraj(self,traj):
		return self.getTotalTime(traj, 0)

	def getPosCtrlPtsTraj(self, traj):
		return self.getPosCtrlPts(traj,0)

	def getYawCtrlPtsTraj(self, traj):
		return self.getYawCtrlPts(traj, 0)

	def printAction(self, action):
		print(f"Raw={action}")
		for index_traj in range(self.num_traj_per_action):
			print(Fore.WHITE+f"Traj {index_traj}: "+Style.RESET_ALL)
			print(Fore.RED+f"  f pos ctrl_pts=\n{self.getPosCtrlPts(action, index_traj)}"+Style.RESET_ALL)
			print(Fore.YELLOW+f"  f yaw ctrl_pts={self.getYawCtrlPts(action, index_traj)}"+Style.RESET_ALL)
			print(Fore.BLUE+f"  Total time={self.getTotalTime(action, index_traj)}"+Style.RESET_ALL)

	def denormalizeAction(self, action_normalized):
		# assert np.logical_and(action_normalized >= -1, action_normalized <= 1).all(), f"action_normalized={action_normalized}"
		action=np.empty(action_normalized.shape)
		action[:,0:-1]=action_normalized[:,0:-1]*self.normalization_constant #Elementwise multiplication
		action[:,-1]=(self.fitter_total_time/2.0)*(action_normalized[:,-1]+1) #Note that action[:,-2] is in [0, fitter_total_time]
		# action[:,-1]=(1.0/2.0)*(action_normalized[:,-1]+1) #Note that action[:,-1] is in [0, 1]
		return action

	def denormalizeTraj(self, traj_normalized):
		#TODO: this could be done more efficiently
		dummy_action_normalized=np.matlib.repmat(traj_normalized, self.num_traj_per_action, 1)
		dummy_action=self.denormalizeAction(dummy_action_normalized)
		traj=dummy_action[0,:].reshape(traj_normalized.shape)
		return traj

	def getActionSize(self):
		return self.action_size

	def getActionShape(self):
		return (self.num_traj_per_action,self.traj_size)

	def getTrajShape(self):
		return (1,self.traj_size)

	def getRandomAction(self):
		random_action=self.denormalizeAction(self.getRandomNormalizedAction())
		return random_action

	def getRandomNormalizedAction(self):
		random_normalized_action=np.random.uniform(-1,1, size=self.getActionShape())
		return random_normalized_action

	def getDegPos(self):
		return self.deg_pos

	def getDegYaw(self):
		return self.deg_yaw

	def f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(self, f_traj, w_state):

		assert type(w_state)==State

		total_time=self.getTotalTimeTraj(f_traj)
		knots_pos=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_pos, self.num_seg)
		f_pos_ctrl_pts = self.getPosCtrlPtsTraj(f_traj)

		#Convert to w frame
		w_pos_ctrl_pts = w_state.w_T_f * f_pos_ctrl_pts

		pf=w_pos_ctrl_pts[:,-1].reshape((3,1)) #Assumming final vel and accel=0 
		p0=w_state.w_pos
		v0=w_state.w_vel
		a0=w_state.w_accel

		p=self.deg_pos

		t1 = knots_pos[1]
		t2 = knots_pos[2]
		tpP1 = knots_pos[p + 1]
		t1PpP1 = knots_pos[1 + p + 1]

		# See Mathematica Notebook
		# the first three control points are already set with the initial conditions
		q0_pos = p0
		q1_pos = p0 + (-t1 + tpP1) * v0 / p
		q2_pos = (p * p * q1_pos - (t1PpP1 - t2) * (a0 * (t2 - tpP1) + v0) - p * (q1_pos + (-t1PpP1 + t2) * v0)) / ((-1 + p) * p)

		w_pos_ctrl_pts=np.concatenate((q0_pos, q1_pos, q2_pos, w_pos_ctrl_pts, pf, pf), axis=1) #Assumming final vel and accel=0

		return w_pos_ctrl_pts, knots_pos

	def get_bbox_inflated_for_student(self):
		return np.array(self.training_other_agent_size) + np.array(self.drone_bbox)
	
	def get_zero_ctrl_pts_for_student(self):
		return np.zeros((3,10)) #TODO: hardcoded

	def f_traj_and_w_state_2_w_student_for_other_agents(self, f_traj, w_state):
		w_agent_for_other_agents = []
		# TODO: make this multi other agents
		w_pos_ctrl_pts, _ = self.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(f_traj, w_state)
		bbox_inflated = np.array(self.training_other_agent_size) + np.array(self.drone_bbox)
		w_agent_for_other_agents.append(Obstacle(ctrl_pts=w_pos_ctrl_pts, bbox_inflated=bbox_inflated, is_obstacle=False))
		return w_agent_for_other_agents

	def f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(self, f_traj, w_state):

		assert type(w_state)==State

		total_time=self.getTotalTimeTraj(f_traj)
		knots_yaw=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_yaw, self.num_seg)
		f_yaw_ctrl_pts= self.getYawCtrlPtsTraj(f_traj)

		w_yaw_ctrl_pts =  f_yaw_ctrl_pts + w_state.w_yaw*np.ones(f_yaw_ctrl_pts.shape)

		y0=w_state.w_yaw
		y_dot0=w_state.yaw_dot
		yf=w_yaw_ctrl_pts[0,-1].reshape((1,1)); #Assumming final vel =0

		p=self.deg_yaw
		t1 = knots_yaw[1]
		tpP1 = knots_yaw[p + 1]
		q0_yaw = y0
		q1_yaw = y0 + (-t1 + tpP1) * y_dot0 / p

		w_yaw_ctrl_pts=np.concatenate((q0_yaw, q1_yaw, w_yaw_ctrl_pts, yf), axis=1) #Assumming final vel and accel=0

		return w_yaw_ctrl_pts, knots_yaw

	def f_trajAnd_w_State2w_ppSolOrGuess(self, f_traj, w_state): #pp stands for py_panther
		w_p_ctrl_pts,knots_p=self.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(f_traj, w_state)
		w_y_ctrl_pts,knots_y=self.f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_traj, w_state)

		#Create and fill solOrGuess object
		w_sol_or_guess=py_panther.solOrGuess()

		w_sol_or_guess.qp=numpy3XmatrixToListOf3dVectors(w_p_ctrl_pts)
		w_sol_or_guess.qy=numpy3XmatrixToListOf3dVectors(w_y_ctrl_pts)

		w_sol_or_guess.knots_p=knots_p
		w_sol_or_guess.knots_y=knots_y

		w_sol_or_guess.solver_succeeded=True
		# w_sol_or_guess.prob=self.getProbTraj(f_traj);
		w_sol_or_guess.cost=0.0 #TODO
		w_sol_or_guess.obst_avoidance_violation=0.0 #TODO
		w_sol_or_guess.dyn_lim_violation=0.0 #TODO
		w_sol_or_guess.is_guess=False #TODO

		w_sol_or_guess.deg_p=self.deg_pos #TODO
		w_sol_or_guess.deg_y=self.deg_yaw #TODO

		return w_sol_or_guess

	def f_trajAnd_f_State2f_ppSolOrGuess(self, f_traj, f_state):
		assert np.linalg.norm(f_state.w_pos)<1e-7, "The pos should be zero"
		assert np.linalg.norm(f_state.w_yaw)<1e-7, "The yaw should be zero"
		return self.f_trajAnd_w_State2w_ppSolOrGuess(f_traj, f_state)

# 	def f_obs_f_traj_2f_ppSolOrGuess(self, f_traj): #pp stands for py_panther
# 		zero_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))
# class State():
# 	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
# 		pos=np.zeros(3,1)
# 		vel=np.zeros(3,1)
# 		return self.f_trajAnd_w_State2w_ppSolOrGuess(f_traj, zero_state)

	def getTrajFromAction(self, action, index_traj):
		return action[index_traj,:].reshape(1,-1)

	def f_trajAnd_w_State2wBS(self, f_traj, w_state, no_deriv=False):

		w_pos_ctrl_pts,_=self.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(f_traj, w_state)
		w_yaw_ctrl_pts,_=self.f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_traj, w_state)
		total_time=self.getTotalTimeTraj(f_traj)

		w_posBS = MyClampedUniformBSpline(0.0, total_time, self.deg_pos, 3, self.num_seg, w_pos_ctrl_pts, no_deriv) #def __init__():[BSpline(knots_pos, w_pos_ctrl_pts[0,:], self.deg_pos)
		w_yawBS = MyClampedUniformBSpline(0.0, total_time, self.deg_yaw, 1, self.num_seg, w_yaw_ctrl_pts, no_deriv)
		
		return w_posBS, w_yawBS

	def f_trajAnd_f_State2fBS(self, f_traj, f_state, no_deriv=False):
		assert np.linalg.norm(f_state.w_pos) < 1e-7, "The pos should be zero"
		assert np.linalg.norm(f_state.w_yaw) < 1e-7, "The yaw should be zero"
		f_posBS, f_yawBS = self.f_trajAnd_w_State2wBS(f_traj, f_state, no_deriv)
		return f_posBS, f_yawBS

	def solOrGuess2traj(self, sol_or_guess):
		traj=np.array([[]])

		#Append position control points
		for i in range(3,len(sol_or_guess.qp)-2):
			# print(sol_or_guess.qp[i].reshape(1,3))
			traj=np.concatenate((traj, sol_or_guess.qp[i].reshape(1,3)), axis=1)

		#Append yaw control points
		for i in range(2,len(sol_or_guess.qy)-1):
			traj=np.concatenate((traj, np.array([[sol_or_guess.qy[i]]])), axis=1)

		#Append total time
		assert sol_or_guess.knots_p[-1]==sol_or_guess.knots_y[-1]
		assert sol_or_guess.knots_p[0]==0
		assert sol_or_guess.knots_y[0]==0
		traj=np.concatenate((traj, np.array([[sol_or_guess.knots_p[-1]]])), axis=1)

		#Append prob of that traj
		# traj=np.concatenate((traj, np.array( [[sol_or_guess.prob]]  )), axis=1)

		assert traj.shape==self.getTrajShape()
		return traj

	def solsOrGuesses2action(self, sols_or_guesses):
		assert len(sols_or_guesses)==self.num_traj_per_action
		action=np.empty((0,self.traj_size))
		for sol_or_guess in sols_or_guesses:
			traj=self.solOrGuess2traj(sol_or_guess)
			action=np.concatenate((action, traj), axis=0)

		assert action.shape==self.getActionShape()
		return action

class ClosedFormYawSubstituter():
	def __init__(self):
		self.cy=py_panther.ClosedFormYawSolver()
		self.am=ActionManager()

	def substituteWithClosedFormYaw(self, f_action_n, w_init_state, w_obstacles):

		# print("In substituteWithClosedFormYaw")

		f_action=self.am.denormalizeAction(f_action_n)

		#Compute w_ppobstacles
		for i in range( np.shape(f_action)[0]): #For each row of action
			traj=f_action[i,:].reshape(1,-1)

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state)

			my_solOrGuess.qy=self.cy.getyCPsfrompCPSUsingClosedForm(my_solOrGuess.qp, my_solOrGuess.getTotalTime(), numpy3XmatrixToListOf3dVectors(w_obstacles[0].ctrl_pts),   w_init_state.w_yaw,   w_init_state.yaw_dot, 0.0)

			tmp=np.array(my_solOrGuess.qy[2:-1])
			f_action[i,self.am.traj_size_pos_ctrl_pts:self.am.traj_size_pos_ctrl_pts+self.am.traj_size_yaw_ctrl_pts]=tmp  - w_init_state.w_yaw*np.ones(tmp.shape)#+ my_solOrGuess.qy[0]
			
			# all_solOrGuess.append(my_solOrGuess)

		f_action_n=self.am.normalizeAction(f_action) #Needed because we have modified action in the previous loop

		return f_action_n

##
##
##	StudentCaller
##
##

class StudentCaller():

	""" StudentCaller().predict() is called in panther.cpp """
	def __init__(self, policy_path):
		# self.student_policy=bc.reconstruct_policy(policy_path)
		self.student_policy=policy = th.load(policy_path, map_location=utils.get_device("auto")) #Same as doing bc.reconstruct_policy(policy_path) 
		self.om=ObservationManager()
		self.am=ActionManager()
		self.obsm=ObstaclesManager()
		self.cc=CostComputer()
		self.cfys=ClosedFormYawSubstituter()
		self.params = readPANTHERparams()
		# self.index_smallest_augmented_cost = 0
		# self.index_best_safe_traj = None
		# self.index_best_unsafe_traj = None
		self.costs_and_violations_of_action = None# CostsAndViolationsOfAction

	def predict(self, w_init_ppstate, w_ppobstacles, w_gterm, num_obst, num_oa): #pp stands for py_panther

		w_init_state=convertPPState2State(w_init_ppstate)
		w_gterm=w_gterm.reshape(3,1)
		w_obstacles=convertPPObstacles2Obstacles(w_ppobstacles)

		## 
		## Construct observation
		##
		
		f_obs=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(w_init_state, w_gterm, w_obstacles)
		f_obs_n=self.om.normalizeObservation(f_obs)

		##
        ## To work around the problem of the following error:
        ##     ValueError: Error: Unexpected observation shape (1, 43) for Box environment, please use (1, 76) or (n_env, 1, 76) for the observation shape.
        ## This is bascially comparing the observation size to the fixed size of self.observation_space.shape
        ## 

		self.student_policy.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=f_obs_n.shape)
		self.student_policy.features_extractor = self.student_policy.features_extractor_class(self.student_policy.observation_space)

		use_num_obses = False # to make sure LSTM takes a various number of obstacles
		self.student_policy.set_use_num_obses(use_num_obses)
		self.student_policy.set_num_obs_num_oa(num_obst, num_oa)
		with th.no_grad():
			action_normalized = self.student_policy._predict(th.as_tensor(f_obs_n), deterministic=True) 
		action_normalized = action_normalized.reshape(self.am.getActionShape())

		##
		## USE CLOSED FORM FOR YAW
		##

		if(self.am.use_closed_form_yaw_student==True):
			action_normalized=self.cfys.substituteWithClosedFormYaw(action_normalized, w_init_state, w_obstacles) #f_action_n, w_init_state, w_obstacles

		action=self.am.denormalizeAction(action_normalized)

		##
		## This funciton uses Casadi, and it needs a fixed size of obstacles
		## The work around is to add dummy obstacles, which is the last element of f_obs_n
		## 

		num_obs = int((f_obs_n.shape[1] - 10) / (3*self.obsm.getCPsPerObstacle()+3))
		for i in range(num_obs, int(rospy.get_param("/SQ01s/panther/num_max_of_obst"))):
			f_obs_n = np.concatenate((f_obs_n, f_obs_n[:,-3*self.obsm.getCPsPerObstacle()-3:]), axis=1)

		self.costs_and_violations_of_action=self.cc.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, action_normalized)
		
		all_solOrGuess=[]

		self.index_best_safe_traj = None
		self.index_best_unsafe_traj = None

		for i in range( np.shape(action)[0]): #For each row of action
			traj=action[i,:].reshape(1,-1)

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state)
			my_solOrGuess.cost = self.costs_and_violations_of_action.costs[i]
			my_solOrGuess.obst_avoidance_violation = self.costs_and_violations_of_action.obst_avoidance_violations[i]
			my_solOrGuess.dyn_lim_violation = self.costs_and_violations_of_action.dyn_lim_violations[i]
			my_solOrGuess.aug_cost = self.cc.computeAugmentedCost(my_solOrGuess.cost, my_solOrGuess.obst_avoidance_violation, my_solOrGuess.dyn_lim_violation)

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state)

			all_solOrGuess.append(my_solOrGuess)

		return all_solOrGuess   

	def getIndexBestTraj(self):
		return self.costs_and_violations_of_action.index_best_traj

def TfMatrix2RosQuatAndVector3(tf_matrix):

  translation_ros=geometry_msgs.msg.Vector3()
  rotation_ros=geometry_msgs.msg.Quaternion()

  translation=tf_matrix.translation()
  translation_ros.x=translation[0]
  translation_ros.y=translation[1]
  translation_ros.z=translation[2]
  # q=tr.quaternion_from_matrix(w_state.w_T_f.T)
  quaternion=tf.transformations.quaternion_from_matrix(tf_matrix.T) #See https://github.com/ros/geometry/issues/64
  rotation_ros.x=quaternion[0] #See order at http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
  rotation_ros.y=quaternion[1]
  rotation_ros.z=quaternion[2]
  rotation_ros.w=quaternion[3]

  return rotation_ros, translation_ros

def TfMatrix2RosPose(tf_matrix):

  rotation_ros, translation_ros=TfMatrix2RosQuatAndVector3(tf_matrix)

  pose_ros=geometry_msgs.msg.Pose()
  pose_ros.position.x=translation_ros.x
  pose_ros.position.y=translation_ros.y
  pose_ros.position.z=translation_ros.z

  pose_ros.orientation=rotation_ros

  return pose_ros

class CostsAndViolationsOfAction():
	def __init__(self, costs, obst_avoidance_violations, dyn_lim_violations, index_best_traj):
		self.costs=costs
		self.obst_avoidance_violations=obst_avoidance_violations
		self.dyn_lim_violations=dyn_lim_violations
		self.index_best_traj=index_best_traj

##
##
## CostComputer
##
##

class CostComputer():
	"""
    The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
    See https://stackoverflow.com/a/68672/6057617
    Note that, even though the class variables are not thread safe (see https://stackoverflow.com/a/1073230/6057617), we are using multiprocessing here, not multithreading
    Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
	"""

	my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct())
	am=ActionManager()
	om=ObservationManager()
	par=getPANTHERparamsAsCppStruct()
	obsm=ObstaclesManager()

	def __init__(self):
		# self.par=getPANTHERparamsAsCppStruct();
		self.num_obstacles=CostComputer.obsm.getNumObs()

	def setUpSolverIpoptAndGetppSolOrGuess(self, f_obs_n, f_traj_n):

		#Denormalize observation and action
		f_obs = CostComputer.om.denormalizeObservation(f_obs_n)
		f_traj = CostComputer.am.denormalizeTraj(f_traj_n)

		#Set up SolverIpopt
		init_state=CostComputer.om.getInit_f_StateFromObservation(f_obs)
		final_state=CostComputer.om.getFinal_f_StateFromObservation(f_obs)
		total_time=computeTotalTime(init_state, final_state, CostComputer.par.v_max, CostComputer.par.a_max, CostComputer.par.factor_alloc)
		CostComputer.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time)
		CostComputer.my_SolverIpopt.setFocusOnObstacle(True)
		obstacles=CostComputer.om.getObstaclesForCasadi(f_obs)

		CostComputer.my_SolverIpopt.setObstaclesForOpt(obstacles)
		f_state=CostComputer.om.get_f_StateFromf_obs(f_obs)
		f_ppSolOrGuess=CostComputer.am.f_trajAnd_f_State2f_ppSolOrGuess(f_traj, f_state)

		return f_ppSolOrGuess	

	def computeObstAvoidanceConstraintsViolation(self, f_obs_n, f_traj_n):

		##
		## Denormalize observation and action
		##

		assert all( (traj >= -1 and traj <= 1) for traj in f_traj_n[0])
		# assert CostComputer.om.obsIsNormalized(f_obs_n)

		f_obs = CostComputer.om.denormalizeObservation(f_obs_n)
		f_traj = CostComputer.am.denormalizeTraj(f_traj_n)

		##
		## total_time
		##

		total_time=CostComputer.am.getTotalTimeTraj(f_traj)

		##
		## Debugging
		##

		if(total_time<1e-5):
			print(f"total_time={total_time}")
			print(f"f_traj_n={f_traj_n}")
			print(f"f_traj={f_traj}")
		
		##
		## loop over obstacles
		##

		f_state = CostComputer.om.get_f_StateFromf_obs(f_obs)
		f_posBS, f_yawBS = CostComputer.am.f_trajAnd_f_State2fBS(f_traj, f_state, no_deriv=True)
		violation = 0.0

		for i in range(self.om.calculateObstacleNumber(f_obs)):
			f_posObs_ctrl_pts = listOf3dVectors2numpy3Xmatrix(CostComputer.om.getCtrlPtsObstacleI(f_obs, i))
			inflated_bbox = CostComputer.om.getBboxInflatedObstacleI(f_obs, i)
			f_posObstBS = MyClampedUniformBSpline(0.0, CostComputer.par.fitter_total_time, CostComputer.par.fitter_deg_pos, 3, CostComputer.par.fitter_num_seg, f_posObs_ctrl_pts, True) 

			min_total_time = min(total_time, CostComputer.par.fitter_total_time)

			for t in np.linspace(start=0.0, stop=min_total_time, num=15).tolist(): #TODO: move num to a parameter
				obs = f_posObstBS.getPosT(t)
				drone = f_posBS.getPosT(t)
				obs_drone = drone - obs #position of the drone wrt the obstacle

				##
				## check collisions
				##

				if abs(obs_drone[0,0]) < inflated_bbox[0,0]/2 and \
					abs(obs_drone[1,0]) < inflated_bbox[1,0]/2 and \
					abs(obs_drone[2,0]) < inflated_bbox[2,0]/2:

					for i in range(3):
						obs_dronecoord = obs_drone[i,0]
						tmp = inflated_bbox[i,0]/2
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
		cost = self.computeCost(f_obs_n, f_traj_n)		
		obst_avoidance_violation = self.computeObstAvoidanceConstraintsViolation(f_obs_n, f_traj_n)
		dyn_lim_violation = self.computeDynLimitsConstraintsViolation(f_obs_n, f_traj_n)
		augmented_cost = self.computeAugmentedCost(cost, obst_avoidance_violation, dyn_lim_violation)
		return cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost

	def computeCost(self, f_obs_n, f_traj_n): 
		f_ppSolOrGuess=self.setUpSolverIpoptAndGetppSolOrGuess(f_obs_n, f_traj_n)
		tmp=CostComputer.my_SolverIpopt.computeCost(f_ppSolOrGuess) 
		return tmp

	def computeAugmentedCost(self, cost, obst_avoidance_violation, dyn_lim_violation):
		return cost + CostComputer.par.lambda_obst_avoidance_violation*obst_avoidance_violation + CostComputer.par.lambda_dyn_lim_violation*dyn_lim_violation

	def getIndexBestTraj(self, f_obs_n, f_action_n):

		tmp=self.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, f_action_n)
		return tmp.index_best_traj
		# smallest_augmented_cost = float('inf')
		# index_smallest_augmented_cost = 0
		# for i in range(f_action_n.shape[0]):
		# 	f_traj_n = self.am.getTrajFromAction(f_action_n, i)

		# 	_, _, _, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_obs_n, traj_n)

		# 	# self.printwithNameAndColor(f"augmented cost traj_{i}={augmented_cost}")
		# 	if(augmented_cost < smallest_augmented_cost):
		# 		smallest_augmented_cost = augmented_cost
		# 		index_smallest_augmented_cost = i
		# return index_smallest_augmented_cost


	def getCostsAndViolationsOfActionFromObsnAndActionn(self, f_obs_n, f_action_n):

		costs=[]
		obst_avoidance_violations=[]
		dyn_lim_violations=[]
		augmented_costs=[]
		alls=[]

		smallest_augmented_safe_cost = float('inf')
		smallest_augmented_unsafe_cost = float('inf')
		index_best_safe_traj = None
		index_best_unsafe_traj = None

		##
		## PARALLEL OPTION
		##

		# def my_func(thread_index):
		# 	traj_n=f_action_n[thread_index,:].reshape(1,-1)
		# 	cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_obs_n, traj_n)
		# 	return [cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost] 

		# num_of_trajs=np.shape(f_action_n)[0]

		# print(num_of_trajs)
		# exit()

		# num_jobs=multiprocessing.cpu_count() #min(multiprocessing.cpu_count(),num_of_trajs); #Note that the class variable my_SolverIpopt will be created once per job created (but only in the first call to predictSeveral I think)
		# alls = Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(num_of_trajs)))) #, prefer="threads"

		# for i in range( np.shape(f_action_n)[0]): #For each row of action
		# 	costs.append(alls[i][0])
		# 	obst_avoidance_violations.append(alls[i][1])
		# 	dyn_lim_violations.append(alls[i][2])
		# 	augmented_costs.append(alls[i][3])

		# start=time.time();

		for i in range( np.shape(f_action_n)[0]): #For each row of action

			##
			## NON-PARALLEL OPTION
			##

			traj_n=f_action_n[i,:].reshape(1,-1)
			cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = self.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(f_obs_n, traj_n)
			costs.append(cost)
			obst_avoidance_violations.append(obst_avoidance_violation)
			dyn_lim_violations.append(dyn_lim_violation)
			augmented_costs.append(augmented_cost)

			##
			## END NON-PARALLEL OPTION
			##

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
			index_best_traj = index_best_safe_traj

		elif(index_best_unsafe_traj is not None):
			index_best_traj= index_best_unsafe_traj
		else:
			print("This should never happen!!")
			exit();		

		result=CostsAndViolationsOfAction(costs=costs, obst_avoidance_violations=obst_avoidance_violations, dyn_lim_violations=dyn_lim_violations, index_best_traj=index_best_traj)

		return result