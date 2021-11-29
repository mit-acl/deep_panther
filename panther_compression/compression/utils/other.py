import yaml
import os
import numpy as np
from pyquaternion import Quaternion
import math
from scipy.interpolate import BSpline
import py_panther
from colorama import init, Fore, Back, Style
from imitation.algorithms import bc
import random
import pytest
import time

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
	qabc=Quaternion(q_w, q_x, q_y, q_z)  #Constructor is Quaternion(w,x,y,z), see http://kieranwynn.github.io/pyquaternion/#object-initialisation


	q_w = math.cos(yaw/2.0);  #w
	q_x = 0;                  #x 
	q_y = 0;                  #y
	q_z = math.sin(yaw/2.0);  #z
	qpsi=Quaternion(q_w, q_x, q_y, q_z)  #Constructor is Quaternion(w,x,y,z)

	w_q_b=qabc * qpsi

	w_T_b = w_q_b.transformation_matrix;
	
	w_T_b[0:3,3]=w_pos.flatten()
	# print(w_T_b)

	return TfMatrix(w_T_b)

def getPANTHERparamsAsCppStruct():

	params_yaml=readPANTHERparams();

	params_yaml["b_T_c"]=np.array([[0, 0, 1, 0],
								  [-1, 0, 0, 0],
								  [0, -1, 0, 0],
								  [0, 0, 0, 1]])

	par=py_panther.parameters();

	for key in params_yaml:
		exec('%s = %s' % ('par.'+key, 'params_yaml["'+key+'"]')) #See https://stackoverflow.com/a/60487422/6057617 and https://www.pythonpool.com/python-string-to-variable-name/

	return par


def readPANTHERparams():

	params_yaml_1=[];
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../../../panther/param/panther.yaml', "r") as stream:
		try:
			params_yaml_1=yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	params_yaml_2=[];
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../../../panther/matlab/casadi_generated_files/params_casadi.yaml', "r") as stream:
		try:
			params_yaml_2=yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	# params_yaml = dict(params_yaml_1.items() + params_yaml_2.items()) #Doesn't work in Python 3
	params_yaml = {**params_yaml_1, **params_yaml_2}                        # NOTE: Python 3.5+ ONLY

	return params_yaml;

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


class ObstaclesManager():
	def __init__(self):
		self.num_obs=1;
		self.params=readPANTHERparams();
		# self.fitter_total_time=params["fitter_total_time"];
		self.fitter_num_seg=self.params["fitter_num_seg"];
		self.fitter_deg_pos=self.params["fitter_deg_pos"];
		self.newRandomPos();

	def newRandomPos(self):
		self.random_pos=np.array([[random.uniform(1.5, 2.5)],[random.uniform(-1.5, 1.5)],[random.uniform(0.0, 2.0)]]);
		# self.random_pos=np.array([[2.5],[0.0],[1.0]]);

	def getNumObs(self):
		return self.num_obs

	def getCPsPerObstacle(self):
		return self.fitter_num_seg + self.fitter_deg_pos

	def getSizeAllObstacles(self):
		#Size of the ctrl_pts + bbox
		return self.num_obs*(3*self.getCPsPerObstacle() + 3) 

	def getFutureWPosObstacles(self,t):
		w_obs=[];
		for i in range(self.num_obs):
			w_ctrl_pts_ob=np.array([[],[],[]]);
			for j in range(self.fitter_num_seg + self.fitter_deg_pos):
				w_ctrl_pts_ob=np.concatenate((w_ctrl_pts_ob, self.random_pos), axis=1)
				# w_ctrl_pts_ob.append(np.array([[2],[2],[2]]))

			# bbox_ob=np.array([[0.5],[0.5], [0.5]]);
			bbox_inflated=np.array([[0.8],[0.8], [0.8]])+2*self.params["drone_radius"]*np.ones((3,1));
			w_obs.append(Obstacle(w_ctrl_pts_ob, bbox_inflated))
		return w_obs;



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
		assert (np.linalg.norm(f_vel)-np.linalg.norm(self.w_vel)) == pytest.approx(0.0)
		return f_vel;
	def f_accel(self):
		self.f_T_w.debug();
		f_accel=self.f_T_w.rot()@self.w_accel;
		assert (np.linalg.norm(f_accel)-np.linalg.norm(self.w_accel)) == pytest.approx(0.0)
		return f_accel;
	def f_yaw(self):
		return 0.0;

def generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg):
	
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	# print("tf*np.ones(deg)= ",tf*np.ones(deg))
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	result= np.concatenate((t0*np.ones(deg), \
							np.linspace(t0, tf, num=num_seg+1),\
							tf*np.ones(deg)))

	return 	result

class MyClampedUniformBSpline():
	def __init__(self,t0, tf, deg, dim, num_seg, ctrl_pts):

		assert dim==ctrl_pts.shape[0]

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

		self.ctrl_pts=ctrl_pts;
		for i in range(dim):
			self.pos_bs.append( BSpline(self.knots, self.ctrl_pts[i,:], self.deg) )
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
		result1=self.ctrl_pts[0:3,-1].reshape(3,1)
		result2=self.getPosT(self.knots[-1])
		np.testing.assert_allclose(result1-result2, 0, atol=1e-07)
		return result1

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		print("The norm is zero, aborting")
		raise RuntimeError
	return v / norm

def isNormalized(observation_or_action_normalized):
	return np.logical_and(observation_or_action_normalized >= -1, observation_or_action_normalized <= 1).all()

def assertIsNormalized(observation_or_action_normalized):
	if not isNormalized(observation_or_action_normalized):
		print("normalized_value=\n")
		print(observation_or_action_normalized)
		# self.printObservation(observation_or_action_normalized)
		raise AssertionError()


class ObservationManager():
	def __init__(self):
		self.obsm=ObstaclesManager();
		#Observation =       [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
		#
		# Where f_ctrl_pts_oi = [cp0.transpose(), cp1.transpose(), ...]
		self.observation_size= 3 +  3 +   1    + 3   + self.obsm.getSizeAllObstacles();

		params=readPANTHERparams();
		self.v_max=np.array(params["v_max"]).reshape(3,1);
		self.a_max=np.array(params["a_max"]).reshape(3,1);
		self.j_max=np.array(params["j_max"]).reshape(3,1);
		self.ydot_max=params["ydot_max"];
		# self.max_dist2goal=params["max_dist2goal"];
		self.max_dist2obs=params["max_dist2obs"];
		self.max_side_bbox_obs=params["max_side_bbox_obs"];
		self.Ra=params["Ra"]
		ones13=np.ones((1,3));
		#Note that the sqrt(3) is needed because the expert/student plan in f_frame --> bouding ball around the box v_max, a_max,... 
		margin_v=100.0 #math.sqrt(3)
		margin_a=100.0 #math.sqrt(3)
		margin_ydot=100.0 #because the student sometimes may not satisfy that limit
		self.normalization_constant=np.concatenate((margin_v*self.v_max.T*ones13, margin_a*self.a_max.T*ones13, margin_ydot*self.ydot_max*np.ones((1,1)), self.Ra*ones13), axis=1)
		for i in range(self.obsm.getNumObs()):
			self.normalization_constant=np.concatenate((self.normalization_constant, self.max_dist2obs*np.ones((1,3*self.obsm.getCPsPerObstacle())), self.max_side_bbox_obs*ones13), axis=1)

		# assert print("Shape observation=", observation.shape==)

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



	def getIndexStartObstacleI(self,i):
		return 10+(self.obsm.getCPsPerObstacle()+3)*i

	def getCtrlPtsObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)
		ctrl_pts=[]; 
		num_cps_per_obstacle=self.obsm.getCPsPerObstacle();
		for j in range(num_cps_per_obstacle):
			index_start_cpoint=index_start_obstacle_i+3*j
			cpoint_j=obs[0,index_start_cpoint:index_start_cpoint+3].reshape(3,1)
			ctrl_pts.append(cpoint_j)

		return ctrl_pts

	def getBboxInflatedObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)

		tmp=index_start_obstacle_i+3*self.obsm.getCPsPerObstacle()

		bbox_inflated=obs[0,tmp:tmp+4].reshape(3,1)

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
		# print("obs is= ", obs)

		obstacles=[]
		for i in range(self.obsm.getNumObs()):

			ctrl_pts=self.getCtrlPtsObstacleI(obs,i) 
			bbox_inflated=self.getBboxInflatedObstacleI(obs,i)

			obstacle=py_panther.obstacleForOpt()

			obstacle.ctrl_pts=ctrl_pts
			obstacle.bbox_inflated=bbox_inflated

			obstacles.append(obstacle)

		return obstacles

	#Normalize in [-1,1]
	def normalizeObservation(self, observation):
		# print("Shape observation=", observation.shape)
		# print("Shape normalization_constant=", self.normalization_constant.shape)
		# print("obsm.getSizeAllObstacles()=", self.obsm.getSizeAllObstacles())

		observation_normalized=observation/self.normalization_constant;
		assertIsNormalized(observation_normalized)
		assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()
		return observation_normalized;

	def getNormalized_fObservationFromTime_w_stateAnd_w_gtermAnd_w_obstacles(self, time, w_state, w_gterm_pos, w_obstacles):
	    # w_obstacles=self.obsm.getFutureWPosObstacles(time) #Don't do this here
	    f_observation=self.construct_f_obsFrom_w_state_and_w_obs(w_state, w_obstacles, w_gterm_pos)
	    f_observationn=self.normalizeObservation(f_observation) #Observation normalized
	    return f_observationn

	def denormalizeObservation(self,observation_normalized):
		assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()

		assertIsNormalized(observation_normalized)
		# assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all(), f"observation_normalized= {observation_normalized}" 
		observation=observation_normalized*self.normalization_constant;
		return observation;

	# def denormalize(self, )

	def getObservationShape(self):
		return (1,self.observation_size)

	def getRandomObservation(self):
		random_observation=self.denormalizeObservation(self.getRandomNormalizedObservation())
		return random_observation

	def getRandomNormalizedObservation(self):
		random_normalized_observation=np.random.uniform(-1,1, size=self.getObservationShape())
		return random_normalized_observation

	def construct_f_obsFrom_w_state_and_w_obs(self,w_state, w_obstacles, w_gterm):

		f_gterm=w_state.f_T_w * w_gterm

		dist2gterm=np.linalg.norm(f_gterm);
		f_g= min(dist2gterm-1e-4, self.Ra)*normalize(f_gterm)
		# print("w_state.f_vel().flatten()= ", w_state.f_vel().flatten())
		# print("w_state.f_accel().flatten()= ", w_state.f_accel().flatten())
		# print("w_state.f_accel().flatten()= ", w_state.f_accel().flatten())
		observation=np.concatenate((w_state.f_vel().flatten(), w_state.f_accel().flatten(), w_state.yaw_dot.flatten(), f_g.flatten()));

		#Convert obs to f frame and append ethem to observation
		for w_obstacle in w_obstacles:
			assert type(w_obstacle.ctrl_pts).__module__ == np.__name__, "the ctrl_pts should be a numpy matrix, not a list"
			observation=np.concatenate((observation, (w_state.f_T_w*w_obstacle.ctrl_pts).flatten(order='F'), (w_obstacle.bbox_inflated).flatten()))


		observation=observation.reshape(self.getObservationShape())

		# print("observation= ", observation)

		assert observation.shape == self.getObservationShape()

		return observation;

def numpy3XmatrixToListOf3dVectors(data):
	data_list=[]
	for i in range(data.shape[1]):
		data_list.append(data[:,i])
	return data_list

class ActionManager():
	def __init__(self):
		params=readPANTHERparams();
		self.deg_pos=params["deg_pos"];
		self.deg_yaw=params["deg_yaw"];
		self.num_seg=params["num_seg"];

		# Define action and observation space

		# action = np.array([ctrl_points_pos   ctrl_points_yaw  total time])
		#
		# --> where ctrlpoints_pos has the ctrlpoints that are not determined by init pos/vel/accel, and final vel/accel
		#     i.e., len(ctrlpoints_pos)= (num_seg_pos + deg_pos - 1 + 1) - 3 - 2 = num_seg_pos + deg_pos - 5;
		#	  and ctrl_points_pos=[cp3.transpose(), cp4.transpose(),...]
		#
		# --> where ctrlpoints_yaw has the ctrlpoints that are not determined by init pos/vel, and final vel
		#     i.e., len(ctrlpoints_yaw)= (num_seg_yaw + deg_yaw - 1 + 1) - 2 - 1 = num_seg_yaw + deg_yaw - 3;

		self.action_size_pos_ctrl_pts = 3*(self.num_seg + self.deg_pos - 5);
		self.action_size_yaw_ctrl_pts = (self.num_seg + self.deg_yaw - 3);
		self.action_size = self.action_size_pos_ctrl_pts + self.action_size_yaw_ctrl_pts +1;
		self.Npos = self.num_seg + self.deg_pos-1;

		self.max_dist2BSPoscPoint=params["max_dist2BSPoscPoint"];
		self.max_yawcPoint=4*math.pi;
		self.fitter_total_time=params["fitter_total_time"];

		print("self.max_dist2BSPoscPoint= ", self.max_dist2BSPoscPoint)
		print("self.max_yawcPoint= ", self.max_yawcPoint)
		self.normalization_constant=np.concatenate((self.max_dist2BSPoscPoint*np.ones((1, self.action_size_pos_ctrl_pts)), \
													self.max_yawcPoint*np.ones((1, self.action_size_yaw_ctrl_pts))), axis=1)

	def assertAction(self,action):
		assert action.shape==self.getActionShape(), f"[Env] ERROR: action.shape={action.shape} but should be={self.getActionShape()}"
		assert not np.isnan(np.sum(action)), f"Action has nan"


	def getDummyOptimalNormalizedAction(self):
		action=self.getDummyOptimalAction();
		return self.normalizeAction(action)

	def getDummyOptimalAction(self):
		return 0.6*np.ones(self.getActionShape())

	def normalizeAction(self, action):
		action_normalized=np.empty(action.shape)
		action_normalized[0,0:-1]=action[0,0:-1]/self.normalization_constant #Elementwise division
		action_normalized[0,-1]=(2.0/self.fitter_total_time)*action[0,-1]-1 #Note that action[0,-1] is in [0, fitter_total_time]
		assert np.logical_and(action_normalized >= -1, action_normalized <= 1).all(), f"action_normalized={action_normalized}"
		return action_normalized;

	def getTotalTime(self,action):
		return action[0,-1]

	def denormalizeAction(self, action_normalized):
		assert np.logical_and(action_normalized >= -1, action_normalized <= 1).all(), f"action_normalized={action_normalized}"
		action=np.empty(action_normalized.shape)
		action[0,0:-1]=action_normalized[0,0:-1]*self.normalization_constant #Elementwise multiplication
		action[0,-1]=(self.fitter_total_time/2.0)*(action_normalized[0,-1]+1) #Note that action[0,-1] is in [0, fitter_total_time]
		return action

	def getActionShape(self):
		return (1,self.action_size)

	def getRandomAction(self):
		random_action=self.denormalizeAction(self.getRandomNormalizedAction())
		return random_action

	def getRandomNormalizedAction(self):
		random_normalized_action=np.random.uniform(-1,1, size=self.getActionShape())
		return random_normalized_action

	def getDegPos(self):
		return self.deg_pos;

	def getDegYaw(self):
		return self.deg_yaw;

	def f_actionAnd_w_State2_w_pos_ctrl_pts_and_knots(self, f_action, w_state):

		assert type(w_state)==State

		total_time=self.getTotalTime(f_action)

		knots_pos=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_pos, self.num_seg)

		f_pos_ctrl_pts = f_action[0,0:self.action_size_pos_ctrl_pts].reshape((3,-1), order='F')

		#Convert to w frame
		w_pos_ctrl_pts = w_state.w_T_f * f_pos_ctrl_pts;

		pf=w_pos_ctrl_pts[:,-1].reshape((3,1)); #Assumming final vel and accel=0 
		p0=w_state.w_pos
		v0=w_state.w_vel
		a0=w_state.w_accel

		p=self.deg_pos;

		t1 = knots_pos[1];
		t2 = knots_pos[2];
		tpP1 = knots_pos[p + 1];
		t1PpP1 = knots_pos[1 + p + 1];

		
		# // See Mathematica Notebook
		q0_pos = p0;
		q1_pos = p0 + (-t1 + tpP1) * v0 / p;
		q2_pos = (p * p * q1_pos - (t1PpP1 - t2) * (a0 * (t2 - tpP1) + v0) - p * (q1_pos + (-t1PpP1 + t2) * v0)) / ((-1 + p) * p);

		w_pos_ctrl_pts=np.concatenate((q0_pos, q1_pos, q2_pos, w_pos_ctrl_pts, pf, pf), axis=1) #Assumming final vel and accel=0

		return w_pos_ctrl_pts, knots_pos


	def f_actionAnd_w_State2_w_yaw_ctrl_pts_and_knots(self, f_action, w_state):
		total_time=self.getTotalTime(f_action)

		knots_yaw=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_yaw, self.num_seg)

		f_yaw_ctrl_pts=f_action[0,self.action_size_pos_ctrl_pts:-1].reshape((1,-1));

		w_yaw_ctrl_pts =  f_yaw_ctrl_pts + w_state.w_yaw*np.ones(f_yaw_ctrl_pts.shape);


		y0=w_state.w_yaw
		y_dot0=w_state.yaw_dot
		yf=w_yaw_ctrl_pts[0,-1].reshape((1,1)); #Assumming final vel =0

		p=self.deg_yaw;
		t1 = knots_yaw[1];
		tpP1 = knots_yaw[p + 1];
		q0_yaw = y0;
		q1_yaw = y0 + (-t1 + tpP1) * y_dot0 / p;

		w_yaw_ctrl_pts=np.concatenate((q0_yaw, q1_yaw, w_yaw_ctrl_pts, yf), axis=1) #Assumming final vel and accel=0

		return w_yaw_ctrl_pts, knots_yaw

	def f_actionAnd_w_State2w_ppSolOrGuess(self, f_action, w_state):
		w_p_ctrl_pts,knots_p=self.f_actionAnd_w_State2_w_pos_ctrl_pts_and_knots(f_action, w_state)
		w_y_ctrl_pts,knots_y=self.f_actionAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_action, w_state)

		#Create and fill solOrGuess object
		w_sol_or_guess=py_panther.solOrGuess();

		w_sol_or_guess.qp=numpy3XmatrixToListOf3dVectors(w_p_ctrl_pts)
		w_sol_or_guess.qy=numpy3XmatrixToListOf3dVectors(w_y_ctrl_pts)

		w_sol_or_guess.knots_p=knots_p
		w_sol_or_guess.knots_y=knots_y

		w_sol_or_guess.solver_succeeded=True
		w_sol_or_guess.cost=0.0 #TODO
		w_sol_or_guess.is_guess=False #TODO

		w_sol_or_guess.deg_p=self.deg_pos #TODO
		w_sol_or_guess.deg_y=self.deg_yaw #TODO

		return w_sol_or_guess



	def f_actionAnd_w_State2wBS(self, f_action, w_state):


		w_pos_ctrl_pts,_=self.f_actionAnd_w_State2_w_pos_ctrl_pts_and_knots(f_action, w_state)
		w_yaw_ctrl_pts,_=self.f_actionAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_action, w_state)
		total_time=f_action[0,-1]


		w_posBS = MyClampedUniformBSpline(0.0, total_time, self.deg_pos, 3, self.num_seg, w_pos_ctrl_pts) #def __init__():[BSpline(knots_pos, w_pos_ctrl_pts[0,:], self.deg_pos)

		w_yawBS =  MyClampedUniformBSpline(0.0, total_time, self.deg_yaw, 1, self.num_seg, w_yaw_ctrl_pts)
		
		return w_posBS, w_yawBS

	def solOrGuess2action(self, sol_or_guess):
		action=np.array([[]]);

		#Append position control points
		for i in range(3,len(sol_or_guess.qp)-2):
			# print(sol_or_guess.qp[i].reshape(1,3))
			action=np.concatenate((action, sol_or_guess.qp[i].reshape(1,3)), axis=1)

		#Append yaw control points
		for i in range(2,len(sol_or_guess.qy)-1):
			action=np.concatenate((action, np.array([[sol_or_guess.qy[i]]])), axis=1)

		#Append total time
		assert sol_or_guess.knots_p[-1]==sol_or_guess.knots_y[-1]
		assert sol_or_guess.knots_p[0]==0
		assert sol_or_guess.knots_y[0]==0
		action=np.concatenate((action, np.array([[sol_or_guess.knots_p[-1]]])), axis=1)


		assert action.shape==self.getActionShape()

		return action



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
        observation_normalized=self.om.normalizeObservation(observation)

        start = time.time()
        action_normalized,info = self.student_policy.predict(observation_normalized, deterministic=True) 
        end = time.time()
        print(f"Calling the student took {(end - start)*(1e3)} ms")

        action_normalized=action_normalized.reshape(self.am.getActionShape())

        action=self.am.denormalizeAction(action_normalized)

        print("action.shape= ", action.shape)
        print("action=", action)   


        assert self.am.getTotalTime(action)>0, "Time needs to be >0"

        # w_pos_ctrl_pts,_ = self.am.actionAndState2_w_pos_ctrl_pts_and_knots(action,w_init_state)

        # print("w_pos_ctrl_pts=", w_pos_ctrl_pts)

        my_solOrGuess= self.am.f_actionAnd_w_State2w_ppSolOrGuess(action,w_init_state);



        # py_panther.solOrGuessl


        return my_solOrGuess   

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
