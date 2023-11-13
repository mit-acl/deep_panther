import numpy as np
import math
import pyquaternion

import random

from gymnasium import spaces

import py_panther

from .State import State
from .Obstacle import Obstacle
from .TfMatrix import TfMatrix

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
        raise ValueError(f"Error: Cannot cast type {type(space)} to a gymnasium type.")

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



def generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg):
	
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	# print("tf*np.ones(deg)= ",tf*np.ones(deg))
	# print("t0*np.ones(deg)= ",t0*np.ones(deg))
	result= np.concatenate((t0*np.ones(deg), \
							np.linspace(t0, tf, num=num_seg+1),\
							tf*np.ones(deg)))

	return 	result

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

def convertPPState2State(ppstate):

	p=ppstate.pos.reshape(3,1)
	v=ppstate.vel.reshape(3,1)
	a=ppstate.accel.reshape(3,1)
	yaw=np.array([[ppstate.yaw]])
	dyaw=np.array([[ppstate.dyaw]])

	return State(p, v, a, yaw, dyaw)

def getZeroState():
	p0=np.array([[0.0],[0.0],[0.0]])
	v0=np.array([[0.0],[0.0],[0.0]])
	a0=np.array([[0.0],[0.0],[0.0]])
	y0=np.array([[0.0]])
	ydot0=np.array([[0.0]])
	zero_state= State(p0, v0, a0, y0, ydot0)

	return zero_state

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

class ExpertDidntSucceed(Exception):
	  pass

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