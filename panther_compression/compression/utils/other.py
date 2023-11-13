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

from gymnasium import spaces

from .ActionManager import ActionManager
from .ObservationManager import ObservationManager
from .ObstaclesManager import ObstaclesManager

from .yaml_utils import getPANTHERparamsAsCppStruct

import numpy.matlib

########
import torch as th
from stable_baselines3.common import utils
########

#######
# import tf.transformations
# from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
#######

###
from joblib import Parallel, delayed
import multiprocessing
##

def cast_gym_to_gymnasium(space):
    def cast_to_box(space):
        return spaces.Box(space.low, space.high)

    def cast_to_discrete(space):
        return spaces.Discrete(space.n)

    def cast_to_multi_discrete(space):
        return spaces.MultiDiscrete(space.nvec)

    def cast_to_multi_binary(space):
        return spaces.MultiBinary(space.n)

    def cast_to_dict(space):
        return spaces.Dict(space)

    space_types = {
        "<class 'gym.spaces.box.Box'>": cast_to_box,
        "<class 'gym.spaces.box.Discrete'>": cast_to_discrete,
        "<class 'gym.spaces.box.MultiDiscrete'>": cast_to_multi_discrete,
        "<class 'gym.spaces.box.MultiBinary'>": cast_to_multi_binary,
        "<class 'gym.spaces.box.Dict'>": cast_to_dict,
    }

    for class_name, cast_fn in space_types.items():
        if str(space.__class__) == class_name:
            return cast_fn(space)
    else:
        raise ValueError(f"Error: Cannot cast tyoe {type(space)} to a gymnasium type.")

class ExpertDidntSucceed(Exception):
	  pass


class TfMatrix():
	def __init__(self, T):
		self.T=T;
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

	#Hopf fibration approach
	thrust=w_accel + np.array([[0.0], [0.0], [9.81]]); 
	thrust_normalized=thrust/np.linalg.norm(thrust);

	a=thrust_normalized[0];
	b=thrust_normalized[1];
	c=thrust_normalized[2];

	tmp=(1/math.sqrt(2*(1+c)));
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

	w_T_b = w_q_b.transformation_matrix;
	
	w_T_b[0:3,3]=w_pos.flatten()
	# print(w_T_b)

	return TfMatrix(w_T_b)

class Obstacle():
	def __init__(self, ctrl_pts, bbox_inflated):
		self.ctrl_pts=ctrl_pts
		self.bbox_inflated=bbox_inflated

def convertPPObstacle2Obstacle(ppobstacle): #pp stands for py_panther
	assert type(ppobstacle.ctrl_pts) is list

	ctrl_pts=np.array([[],[],[]]);
	for ctrl_pt in ppobstacle.ctrl_pts:
		ctrl_pts=np.concatenate((ctrl_pts, ctrl_pt.reshape(3,1)), axis=1)

	obstacle=Obstacle(ctrl_pts, ppobstacle.bbox_inflated.reshape(3,1));

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


def getObsAndGtermToCrossPath():

	# thetas=[-np.pi/4, np.pi/4]
	# theta=random.choice(thetas)
	theta=random.uniform(-np.pi, np.pi)
	radius_obstacle=random.uniform(0.0, 5.5)
	radius_gterm=random.uniform(0.0, 10.0) #radius_obstacle + random.uniform(2.0, 10.0)
	std_deg=30
	theta_g_term=theta + random.uniform(-std_deg*np.pi/180, std_deg*np.pi/180) 
	center=np.zeros((3,1))

	w_pos_obstacle = center + np.array([[radius_obstacle*math.cos(theta)],[radius_obstacle*math.sin(theta)],[1.0]])
	w_pos_g_term = center + np.array([[radius_gterm*math.cos(theta_g_term)],[radius_gterm*math.sin(theta_g_term)],[1.0]])

	# Use this to train static obstacles
	theta=random.uniform(-np.pi/2, np.pi/2)
	radius_obstacle=random.uniform(1.5, 4.5)
	radius_gterm=radius_obstacle + random.uniform(1.0, 6.0)
	std_deg=10#30
	theta_g_term=theta + random.uniform(-std_deg*np.pi/180, std_deg*np.pi/180) 
	center=np.zeros((3,1))

	w_pos_obstacle = center + np.array([[radius_obstacle*math.cos(theta)],[radius_obstacle*math.sin(theta)],[1.0]])
	w_pos_g_term = center + np.array([[radius_gterm*math.cos(theta_g_term)],[radius_gterm*math.sin(theta_g_term)],[random.uniform(1.0-1.5, 1.0+1.5)]])
	########


	return w_pos_obstacle, w_pos_g_term


class GTermManager():
	def __init__(self):
		self.newRandomPos();

	def newRandomPos(self):
		self.w_gterm=np.array([[random.uniform(-10.0, 10.0)],[random.uniform(-10.0, 10.0)],[random.uniform(1.0,1.0)]]);
		#self.w_gterm=np.array([[5.0],[0.0],[1.0]]);

	def newRandomPosFarFrom_w_Position(self, w_position):
		dist=0.0
		while dist<3.0: #Goal at least 3 meters away from the current position
			self.newRandomPos()
			dist=np.linalg.norm(self.w_gterm-w_position)

	def setPos(self, pos):
		self.w_gterm=pos

	def get_w_GTermPos(self):
		return self.w_gterm;

class Trefoil():
	def __init__(self, pos, scale, offset, slower):
		self.x=pos[0,0];
		self.y=pos[1,0];
		self.z=pos[2,0];
		self.scale_x=scale[0,0]
		self.scale_y=scale[1,0]
		self.scale_z=scale[2,0]
		self.offset=offset;
		self.slower=slower;

	def getPosT(self,t):
		#slower=1.0; #The higher, the slower the obstacles move" 
		tt=t/self.slower;

		x_trefoil=(self.scale_x/6.0)*(math.sin(tt+self.offset) + 2*math.sin(2*tt+self.offset)) + self.x
		y_trefoil=(self.scale_y/5.0)*(math.cos(tt+self.offset) - 2*math.cos(2*tt+self.offset)) + self.y
		z_trefoil=(self.scale_z/2.0)*(-math.sin(3*tt+self.offset)) + self.z

		# x_trefoil=self.x
		# y_trefoil=self.y
		# z_trefoil=self.z

		return np.array([[x_trefoil], [y_trefoil], [z_trefoil]])




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
		self.w_T_f= posAccelYaw2TfMatrix(self.w_pos, np.array([[0.0],[0.0], [0.0]]), w_yaw) #pos, accel, yaw
		ez=np.array([[0.0],[0.0],[1.0]]);
		np.testing.assert_allclose(self.w_T_f.T[0:3,2].reshape(3,1)-ez, 0, atol=1e-07)
		self.f_T_w= self.w_T_f.inv()
	def f_pos(self):
		return self.f_T_w*self.w_pos;
	def f_vel(self):
		f_vel=self.f_T_w.rot()@self.w_vel;
		# assert (np.linalg.norm(f_vel)-np.linalg.norm(self.w_vel)) == pytest.approx(0.0), f"f_vel={f_vel} (norm={np.linalg.norm(f_vel)}), w_vel={self.w_vel} (norm={np.linalg.norm(self.w_vel)}), f_R_w={self.f_T_w.rot()}, "
		return f_vel;
	def f_accel(self):
		self.f_T_w.debug();
		f_accel=self.f_T_w.rot()@self.w_accel;
		# assert (np.linalg.norm(f_accel)-np.linalg.norm(self.w_accel)) == pytest.approx(0.0), f"f_accel={f_accel} (norm={np.linalg.norm(f_accel)}), w_accel={self.w_accel} (norm={np.linalg.norm(self.w_accel)}), f_R_w={self.f_T_w.rot()}, " 
		return f_accel;
	def f_yaw(self):
		return np.array([[0.0]]);
	def print_w_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In w frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw, "+ \
		Fore.MAGENTA +f"dyaw: "+ \
		Fore.RED +f"{self.w_pos.T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.w_vel.T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.w_accel.T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.w_yaw}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot}"+Style.RESET_ALL)

	def print_f_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In f frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw, "+ \
		Fore.MAGENTA +f"dyaw: "+ \
		Fore.RED +f"{self.f_pos().T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.f_vel().T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.f_accel().T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.f_yaw()}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot}"+Style.RESET_ALL)

def generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg):
	
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	# print("tf*np.ones(deg)= ",tf*np.ones(deg))
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	result= np.concatenate((t0*np.ones(deg), \
							np.linspace(t0, tf, num=num_seg+1),\
							tf*np.ones(deg)))

	return 	result

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
		self.deg=deg;
		self.num_seg=num_seg;
		self.dim=dim;
		self.knots=generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg)

		###Debugging
		if (abs(tf-t0)<1e-5):
			print(f"t0={t0}, tf={tf}, deg={deg}, num_seg={num_seg}")
			print(f"self.knots={self.knots}")
		#######

		self.ctrl_pts=ctrl_pts;
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
		# print("The norm is zero, aborting!!")
		# exit()
		raise RuntimeError("The norm is zero")
	return v / norm

def computeTotalTime(init_state, final_state, par_vmax, par_amax, par_factor_alloc):
	# invsqrt3_vector=math.sqrt(3)*np.ones((3,1));
	total_time=par_factor_alloc*py_panther.getMinTimeDoubleIntegrator3DFromState(init_state, final_state, par_vmax, par_amax)
	return total_time

#Wraps an angle in [-pi, pi)
#Works also for np arrays
def wrapInmPiPi(data): #https://stackoverflow.com/a/15927914
	return (data + np.pi) % (2 * np.pi) - np.pi

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

# 	def f_obs_f_traj_2f_ppSolOrGuess(self, f_traj): #pp stands for py_panther
# 		zero_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))
# class State():
# 	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
# 		pos=np.zeros(3,1)
# 		vel=np.zeros(3,1)
# 		return self.f_trajAnd_w_State2w_ppSolOrGuess(f_traj, zero_state)

class ClosedFormYawSubstituter():
	def __init__(self):
		self.cy=py_panther.ClosedFormYawSolver();
		self.am=ActionManager();



	def substituteWithClosedFormYaw(self, f_action_n, w_init_state, w_obstacles):

		# print("In substituteWithClosedFormYaw")

		f_action=self.am.denormalizeAction(f_action_n)

		#Compute w_ppobstacles

		#####
		for i in range( np.shape(f_action)[0]): #For each row of action
			traj=f_action[i,:].reshape(1,-1);

			my_solOrGuess= self.am.f_trajAnd_w_State2w_ppSolOrGuess(traj,w_init_state);

			my_solOrGuess.qy=self.cy.getyCPsfrompCPSUsingClosedForm(my_solOrGuess.qp, my_solOrGuess.getTotalTime(), numpy3XmatrixToListOf3dVectors(w_obstacles[0].ctrl_pts),   w_init_state.w_yaw,   w_init_state.yaw_dot, 0.0)

			tmp=np.array(my_solOrGuess.qy[2:-1])
			f_action[i,self.am.traj_size_pos_ctrl_pts:self.am.traj_size_pos_ctrl_pts+self.am.traj_size_yaw_ctrl_pts]=tmp  - w_init_state.w_yaw*np.ones(tmp.shape)#+ my_solOrGuess.qy[0]
			
			# all_solOrGuess.append(my_solOrGuess)

		f_action_n=self.am.normalizeAction(f_action) #Needed because we have modified action in the previous loop

		return f_action_n



class StudentCaller():
	def __init__(self, policy_path):
		# self.student_policy=bc.reconstruct_policy(policy_path)
		self.student_policy=policy = th.load(policy_path, map_location=utils.get_device("auto")) #Same as doing bc.reconstruct_policy(policy_path)
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
	am=ActionManager();
	om=ObservationManager();
	par=getPANTHERparamsAsCppStruct();
	obsm=ObstaclesManager();


	def __init__(self):
		# self.par=getPANTHERparamsAsCppStruct();

		self.num_obstacles=CostComputer.obsm.getNumObs()

		

	def setUpSolverIpoptAndGetppSolOrGuess(self, f_obs_n, f_traj_n):

		#Denormalize observation and action
		f_obs = CostComputer.om.denormalizeObservation(f_obs_n);
		f_traj = CostComputer.am.denormalizeTraj(f_traj_n);

		#Set up SolverIpopt
		# print("\n========================")
		init_state=CostComputer.om.getInit_f_StateFromObservation(f_obs)
		final_state=CostComputer.om.getFinal_f_StateFromObservation(f_obs)
		total_time=computeTotalTime(init_state, final_state, CostComputer.par.v_max, CostComputer.par.a_max, CostComputer.par.factor_alloc)
		# print(f"init_state=")
		# init_state.printHorizontal();
		# print(f"final_state=")
		# final_state.printHorizontal();
		# print(f"total_time={total_time}")
		CostComputer.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
		CostComputer.my_SolverIpopt.setFocusOnObstacle(True);
		obstacles=CostComputer.om.getObstacles(f_obs)

		# print(f"obstacles=")

		# for obs in obstacles:
		# 	obs.printInfo()

		CostComputer.my_SolverIpopt.setObstaclesForOpt(obstacles);

		###############################
		f_state=CostComputer.om.get_f_StateFromf_obs(f_obs)
		f_ppSolOrGuess=CostComputer.am.f_trajAnd_f_State2f_ppSolOrGuess(f_traj, f_state)
		###############################

		return f_ppSolOrGuess;		


	def computeObstAvoidanceConstraintsViolation(self, f_obs_n, f_traj_n):

		#Denormalize observation and action
		f_obs = CostComputer.om.denormalizeObservation(f_obs_n);
		f_traj = CostComputer.am.denormalizeTraj(f_traj_n);

		total_time=CostComputer.am.getTotalTimeTraj(f_traj)

		###Debugging
		if(total_time<1e-5):
			print(f"total_time={total_time}")
			print(f"f_traj_n={f_traj_n}")
			print(f"f_traj={f_traj}")
		######

		f_state = CostComputer.om.get_f_StateFromf_obs(f_obs)
		f_posBS, f_yawBS = CostComputer.am.f_trajAnd_f_State2fBS(f_traj, f_state, no_deriv=True)

		violation=0


		for i in range(self.num_obstacles):
			f_posObs_ctrl_pts=listOf3dVectors2numpy3Xmatrix(CostComputer.om.getCtrlPtsObstacleI(f_obs, i))
			bbox=CostComputer.om.getBboxInflatedObstacleI(f_obs, i)
			# print(f"f_posObs_ctrl_pts={f_posObs_ctrl_pts}")
			# print(f"f_posBS.ctrl_pts={f_posBS.ctrl_pts}")

			# start=time.time();

			f_posObstBS = MyClampedUniformBSpline(0.0, CostComputer.par.fitter_total_time, CostComputer.par.fitter_deg_pos, 3, CostComputer.par.fitter_num_seg, f_posObs_ctrl_pts, True) 

			# print(f" compute MyClampedUniformBSpline creation took {(time.time() - start)*(1e3)} ms")


			# print("\n============")
			# start=time.time();

			#TODO: move num to a parameter
			for t in np.linspace(start=0.0, stop=total_time, num=15).tolist():

				obs = f_posObstBS.getPosT(t);
				drone = f_posBS.getPosT(t);

				obs_drone = drone - obs #position of the drone wrt the obstacle

				if(abs(obs_drone[0,0])<=bbox[0,0]/2 and abs(obs_drone[1,0])<=bbox[1,0]/2 and abs(obs_drone[2,0])<=bbox[2,0]/2):

					for i in range(3):
						obs_dronecoord=obs_drone[i,0]
						tmp = bbox[i,0]/2
						violation+= min(abs(tmp - obs_dronecoord), abs(obs_dronecoord - (-tmp)) )

					# print("THERE IS VIOLATION in obs avoid")
					# exit()

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




#You can check the previous function by using this Matlab script:
# clc;
# t0=0;
# dim_yaw=1;
# deg_yaw=2;
# dim_pos=3;
# deg_pos=3;
# num_seg=6;

# tf=0.07246680753723689;
# pos_ctrl_pts=[[0.2022673  0.20524327 0.21123073 0.24623035 0.80351345 0.50678383 0.08125478 0.08125478 0.08125478];
#  [0.57454454 0.57512383 0.5763065  0.34355742 0.63901906 0.5159091  0.01585657 0.01585657 0.01585657];
#  [0.97750177 0.98143953 0.98935799 0.4940338  0.13587985 0.3554246  0.97093003 0.97093003 0.97093003]];
# yaw_ctrl_pts=[0.56358052 0.565937   0.84083564 0.4594103  0.8658295  0.17317506 0.43348313 0.43348313];


# sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti); %spline yaw.
# sp.updateCPsWithSolution(pos_ctrl_pts)

# sp.getPosT(t0)
# sp.getVelT(t0)
# sp.getAccelT(t0)


# sy=MyClampedUniformSpline(t0,tf,deg_yaw, dim_yaw, num_seg, opti); %spline yaw.
# sy.updateCPsWithSolution(yaw_ctrl_pts)


# sy.getPosT(t0)
# sy.getVelT(t0)