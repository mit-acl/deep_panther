import yaml
import os
import numpy as np
from pyquaternion import Quaternion
import math
from scipy.interpolate import BSpline
import py_panther

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
		self.bbox_inflated=np.array([[0.5],[0.5], [0.5]])

class ObstaclesManager():
	def __init__(self):
		self.num_obs=1;
		params=readPANTHERparams();
		# self.fitter_total_time=params["fitter_total_time"];
		self.fitter_num_seg=params["fitter_num_seg"];
		self.fitter_deg_pos=params["fitter_deg_pos"];

	def getNumObs(self):
		return self.num_obs

	def getCPsPerObstace(self):
		return self.fitter_num_seg + self.fitter_deg_pos

	def getSizeAllObstacles(self):
		#Size of the ctrl_pts + bbox
		return self.num_obs*(3*self.getCPsPerObstace() + 3) 

	def getFutureWPosObstacles(self,t):
		w_obs=[];
		for i in range(self.num_obs):
			w_ctrl_pts_ob=np.array([[],[],[]]);
			for j in range(self.fitter_num_seg + self.fitter_deg_pos):
				w_ctrl_pts_ob=np.concatenate((w_ctrl_pts_ob, np.array([[2],[2],[2]])), axis=1)
				# w_ctrl_pts_ob.append(np.array([[2],[2],[2]]))

			bbox_ob=np.array([[0.5],[0.5], [0.5]]);
			w_obs.append(Obstacle(w_ctrl_pts_ob, bbox_ob))
		return w_obs;



class State():
	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
		self.w_pos = w_pos
		self.w_vel = w_vel
		self.w_accel = w_accel
		self.w_yaw = w_yaw
		self.yaw_dot = yaw_dot
		self.w_T_f= posAccelYaw2TfMatrix(self.w_pos, self.w_accel, 0.0)
		self.f_T_w= self.w_T_f.inv()
	def f_pos(self):
		return self.f_T_w*self.w_pos;
	def f_vel(self):
		return self.f_T_w*self.w_vel;
	def f_accel(self):
		return self.f_T_w*self.w_accel;
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

class ObservationManager():
	def __init__(self):
		self.om=ObstaclesManager();
		#Observation =       [f_v, f_a, yaw_dot, f_g,  f_o1, bbox_o1, f_o2, bbox_o2 ,...]
		self.observation_size= 3 +  3 +   1    + 3   + self.om.getSizeAllObstacles();

		params=readPANTHERparams();
		self.v_max=params["v_max"];
		self.a_max=params["a_max"];
		self.j_max=params["j_max"];
		self.ydot_max=params["ydot_max"];
		self.max_dist2goal=params["max_dist2goal"];
		self.max_dist2obs=params["max_dist2obs"];
		self.max_side_bbox_obs=params["max_side_bbox_obs"];
		ones13=np.ones((1,3));
		self.normalization_constant=np.concatenate((self.v_max*ones13, self.a_max*ones13, self.ydot_max*np.ones((1,1)), self.max_dist2goal*ones13), axis=1)
		for i in range(self.om.getNumObs()):
			self.normalization_constant=np.concatenate((self.normalization_constant, self.max_dist2obs*np.ones((1,3*self.om.getCPsPerObstace())), self.max_side_bbox_obs*ones13), axis=1)

		# assert print("Shape observation=", observation.shape==)

	#Normalize in [-1,1]
	def normalizeObservation(self, observation):
		print("Shape observation=", observation.shape)
		print("Shape normalization_constant=", self.normalization_constant.shape)
		print("om.getSizeAllObstacles()=", self.om.getSizeAllObstacles())

		observation_normalized=observation/self.normalization_constant;
		assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()
		return observation_normalized;

	def denormalizeObservation(self,observation_normalized):
		assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all(), f"observation_normalized= {observation_normalized}" 
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

	def construct_f_obsFrom_w_state_and_w_obs(self,w_state, w_obs, w_goal):

		f_goal=w_state.f_T_w * w_goal
		# print("w_state.f_vel().flatten()= ", w_state.f_vel().flatten())
		# print("w_state.f_accel().flatten()= ", w_state.f_accel().flatten())
		# print("w_state.f_accel().flatten()= ", w_state.f_accel().flatten())
		observation=np.concatenate((w_state.f_vel().flatten(), w_state.f_accel().flatten(), w_state.yaw_dot.flatten(), f_goal.flatten()));

		#Convert obs to f frame and append ethem to observation
		for w_ob in w_obs:
			observation=np.concatenate((observation, (w_state.f_T_w*w_ob.ctrl_pts).flatten(), (w_ob.bbox_inflated).flatten()))


		observation=observation.reshape(self.getObservationShape())

		# print("observation= ", observation)

		assert observation.shape == self.getObservationShape()

		return observation;

class ActionManager():
	def __init__(self):
		params=readPANTHERparams();
		self.deg_pos=params["deg_pos"];
		self.deg_yaw=params["deg_yaw"];
		self.num_seg=params["num_seg"];

		# Define action and observation space

		# action = np.array([ctrlpoints_pos   ctrol_points_yaw  total time])
		# where ctrlpoints_pos has the ctrlpoints that are not determined by init pos/vel/accel, and final vel/accel
		# i.e., len(ctrlpoints_pos)= (num_seg_pos + deg_pos - 1 + 1) - 3 - 2 = num_seg_pos + deg_pos - 5;
		# where ctrlpoints_pos has the ctrlpoints that are not determined by init pos/vel, and final vel
		# i.e., len(ctrlpoints_yaw)= (num_seg_yaw + deg_yaw - 1 + 1) - 2 - 1 = num_seg_yaw + deg_yaw - 3;

		self.action_size_pos_ctrl_pts = 3*(self.num_seg + self.deg_pos - 5);
		self.action_size_yaw_ctrl_pts = (self.num_seg + self.deg_yaw - 3);
		self.action_size = self.action_size_pos_ctrl_pts + self.action_size_yaw_ctrl_pts +1;
		self.Npos = self.num_seg + self.deg_pos-1;

		self.max_dist2BSPoscPoint=params["max_dist2BSPoscPoint"];
		self.max_yawcPoint=2*math.pi;
		self.fitter_total_time=params["fitter_total_time"];

		print("self.max_dist2BSPoscPoint= ", self.max_dist2BSPoscPoint)
		print("self.max_yawcPoint= ", self.max_yawcPoint)
		self.normalization_constant=np.concatenate((self.max_dist2BSPoscPoint*np.ones((1, self.action_size_pos_ctrl_pts)), \
													self.max_yawcPoint*np.ones((1, self.action_size_yaw_ctrl_pts))), axis=1)

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

	def action2wBS(self, f_action, w_state):

		# print("f_action= ",f_action)
		# print("f_action.shape= ",f_action.shape)
		total_time=f_action[0,-1]

		knots_pos=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_pos, self.num_seg)
		knots_yaw=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_yaw, self.num_seg)

		f_pos_ctrl_pts = f_action[0,0:self.action_size_pos_ctrl_pts].reshape((3,-1))
		f_yaw_ctrl_pts=f_action[0,self.action_size_pos_ctrl_pts:-1].reshape((1,-1));

		# print("\n f_pos_ctrl_pts= \n", f_pos_ctrl_pts)

		#Convert to w frame
		# print("\n--------------------Going to convert values--------------------\n")
		w_pos_ctrl_pts = w_state.w_T_f * f_pos_ctrl_pts;
		w_yaw_ctrl_pts =  f_yaw_ctrl_pts + w_state.w_yaw*np.ones(f_yaw_ctrl_pts.shape);

		# print(w_pos_ctrl_pts)
		# print(w_yaw_ctrl_pts)


		pf=w_pos_ctrl_pts[:,-1].reshape((3,-1)); #Assumming final vel and accel=0 
		p0=w_state.w_pos
		v0=w_state.w_vel
		a0=w_state.w_accel
		y0=w_state.w_yaw
		y_dot0=w_state.yaw_dot

		p=self.deg_pos;

		t1 = knots_pos[1];
		t2 = knots_pos[2];
		tpP1 = knots_pos[p + 1];
		t1PpP1 = knots_pos[1 + p + 1];

		
		# // See Mathematica Notebook
		q0_pos = p0;
		q1_pos = p0 + (-t1 + tpP1) * v0 / p;
		q2_pos = (p * p * q1_pos - (t1PpP1 - t2) * (a0 * (t2 - tpP1) + v0) - p * (q1_pos + (-t1PpP1 + t2) * v0)) / ((-1 + p) * p);

		# print ("q0= \n", q0_pos)
		# print ("q1= \n", q1_pos)
		# print ("q2= \n", q2_pos)
		# print ("pos_ctrl_pts= \n", pos_ctrl_pts)
		# print ("pf= \n", pf)

		w_pos_ctrl_pts=np.concatenate((q0_pos, q1_pos, q2_pos, w_pos_ctrl_pts, pf, pf), axis=1) #Assumming final vel and accel=0

		###########FOR YAW
		# print("w_yaw_ctrl_pts=", w_yaw_ctrl_pts)
		yf=w_yaw_ctrl_pts[0,-1].reshape((1,1)); #Assumming final vel =0

		p=self.deg_yaw;
		t1 = knots_yaw[1];
		tpP1 = knots_yaw[p + 1];
		q0_yaw = y0;
		q1_yaw = y0 + (-t1 + tpP1) * y_dot0 / p;

		# print("q0_yaw= ",q0_yaw)
		# print("q1_yaw= ",q1_yaw)
		# print("w_yaw_ctrl_pts= ",w_yaw_ctrl_pts)
		# print("yf= ",yf)

		w_yaw_ctrl_pts=np.concatenate((q0_yaw, q1_yaw, w_yaw_ctrl_pts, yf), axis=1) #Assumming final vel and accel=0

		# print("\n knots_pos= \n", knots_pos)
		# print("\n w_pos_ctrl_pts= \n", w_pos_ctrl_pts)
		# print("\n self.deg_pos= ", self.deg_pos)
		# print("\n knots_yaw= \n", knots_yaw)

		#Generate the splines
		w_posBS = MyClampedUniformBSpline(0.0, total_time, self.deg_pos, 3, self.num_seg, w_pos_ctrl_pts) #def __init__():[BSpline(knots_pos, w_pos_ctrl_pts[0,:], self.deg_pos)

		w_yawBS =  MyClampedUniformBSpline(0.0, total_time, self.deg_yaw, 1, self.num_seg, w_yaw_ctrl_pts)
		
		return w_posBS, w_yawBS

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
