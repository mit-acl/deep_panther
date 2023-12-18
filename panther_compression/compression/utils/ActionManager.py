import numpy as np
import numpy.matlib
import py_panther

from colorama import Fore, Style

from .MyClampedUniformBSpline import MyClampedUniformBSpline
from .utils import generateKnotsForClampedUniformBspline, numpy3XmatrixToListOf3dVectors
from .State import State

from .yaml_utils import readPANTHERparams

class ActionManager():
	def __init__(self, dim=3):
		self.dim = dim
		params=readPANTHERparams();
		self.deg_pos=params["deg_pos"];
		self.deg_yaw=params["deg_yaw"];
		self.num_seg=params["num_seg"];
		self.use_closed_form_yaw_student=params["use_closed_form_yaw_student"];

		# Define action and observation space

		# action = np.array([ctrl_points_pos   ctrl_points_yaw  total time, prob that traj])
		#
		# --> where ctrlpoints_pos has the ctrlpoints that are not determined by init pos/vel/accel, and final vel/accel
		#     i.e., len(ctrlpoints_pos)= (num_seg_pos + deg_pos - 1 + 1) - 3 - 2 = num_seg_pos + deg_pos - 5;
		#	  and ctrl_points_pos=[cp3.transpose(), cp4.transpose(),...]
		#
		# --> where ctrlpoints_yaw has the ctrlpoints that are not determined by init pos/vel, and final vel
		#     i.e., len(ctrlpoints_yaw)= (num_seg_yaw + deg_yaw - 1 + 1) - 2 - 1 = num_seg_yaw + deg_yaw - 3;

		self.num_traj_per_action=params["num_of_trajs_per_replan"];

		self.total_num_pos_ctrl_pts = self.num_seg + self.deg_pos
		self.traj_size_pos_ctrl_pts = self.dim*(self.total_num_pos_ctrl_pts - 5);
		self.traj_size_yaw_ctrl_pts = (self.num_seg + self.deg_yaw - 3);
		# self.traj_size = self.traj_size_pos_ctrl_pts + self.traj_size_yaw_ctrl_pts + 1 + 1; # Last two numbers are time and prob that traj is real
		self.traj_size = self.traj_size_pos_ctrl_pts + self.traj_size_yaw_ctrl_pts + 1; # Last number is time
		self.action_size = self.num_traj_per_action*self.traj_size;
		self.Npos = self.num_seg + self.deg_pos-1;

		self.max_dist2BSPoscPoint=params["max_dist2BSPoscPoint"];
		self.max_yawcPoint=4e3*np.pi;
		self.fitter_total_time=params["fitter_total_time"];

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
			print(msg_before+"The observation is not normalized")
			print(f"NORMALIZED VALUE={action_normalized}")
			action=self.denormalizeAction(action_normalized)
			print(f"VALUE={action}")
			# self.printObservation(observation);
			raise AssertionError()

	def assertAction(self,action):
		assert action.shape==self.getActionShape(), f"[Env] ERROR: action.shape={action.shape} but should be={self.getActionShape()}"
		assert np.all(~np.isnan(action)), f"Action has nan"
		#assert not np.isnan(np.sum(action)), f"Action has nan"


	def getDummyOptimalNormalizedAction(self):
		action=self.getDummyOptimalAction();
		return self.normalizeAction(action)

	def getDummyOptimalAction(self):
		# return 0.6*np.ones(self.getActionShape())

		dummy=np.ones((self.num_traj_per_action,self.traj_size))

		for i in range((dummy.shape[0])):
			for j in range(0, self.traj_size_pos_ctrl_pts, self.dim):
				dummy[i,j]=i+j/10

		return dummy


	def getNanAction(self):
		return np.full(self.getActionShape(), np.nan)

	def isNanAction(self, act):
		return np.isnan(np.sum(act))

	def normalizeAction(self, action):
		action_normalized=np.empty(action.shape)
		action_normalized[:,0:-1]=action[:,0:-1]/self.normalization_constant #Elementwise division
		action_normalized[:,-1]=(2.0/self.fitter_total_time)*action[:,-1]-1 #Note that action[0,-1] is in [0, fitter_total_time]
		# action_normalized[:,-1]=(2.0/1.0)*action[:,-1]-1 #Note that action[0,-1] is in [0, 1]

		for index_traj in range(self.num_traj_per_action):
			time_normalized=self.getTotalTime(action_normalized, index_traj);
			slack=1-abs(time_normalized);
			if(slack<0):
				if abs(slack)<1e-5: #Can happen due to the tolerances in the optimization
					# print(f"Before= {action_normalized[0,-1]}")
					action_normalized[index_traj,-1]=np.clip(time_normalized, -1.0, 1.0) #Saturate within limits
					# print(f"After= {action_normalized[0,-1]}")
				else:
					assert False, f"time_normalized={time_normalized}"


		# assert np.logical_and(action_normalized >= -1, action_normalized <= 1).all(), f"action_normalized={action_normalized}, last element={action_normalized[0,-1]}"
		return action_normalized;

	# def getProb(self,action, index_traj):
	# 	return action[index_traj,-1]

	def getTotalTime(self,action, index_traj):
		return action[index_traj,-1]

	def getPosCtrlPts(self, action, index_traj):
		return action[index_traj,0:self.traj_size_pos_ctrl_pts].reshape((self.dim, -1), order='F')

	def getYawCtrlPts(self, action, index_traj):
		return action[index_traj,self.traj_size_pos_ctrl_pts:-1].reshape((1,-1));

	# def getProbTraj(self,traj):
	# 	return self.getProb(traj, 0)

	def getTotalTimeTraj(self,traj):
		return self.getTotalTime(traj, 0)

	def getPosCtrlPtsTraj(self, traj):
		return self.getPosCtrlPts(traj,0)

	def getYawCtrlPtsTraj(self, traj):
		return self.getYawCtrlPts(traj, 0);

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
		return self.deg_pos;

	def getDegYaw(self):
		return self.deg_yaw;

	def f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(self, f_traj, w_state):

		assert type(w_state)==State

		total_time=self.getTotalTimeTraj(f_traj)

		knots_pos=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_pos, self.num_seg)

		f_pos_ctrl_pts = self.getPosCtrlPtsTraj(f_traj)

		#Convert to w frame
		w_pos_ctrl_pts = w_state.w_T_f * f_pos_ctrl_pts;

		pf=w_pos_ctrl_pts[0:self.dim,-1].reshape((self.dim,1)); #Assumming final vel and accel=0 
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


	def f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(self, f_traj, w_state):
		total_time=self.getTotalTimeTraj(f_traj)

		knots_yaw=generateKnotsForClampedUniformBspline(0.0, total_time, self.deg_yaw, self.num_seg)

		f_yaw_ctrl_pts= self.getYawCtrlPtsTraj(f_traj)

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

	def f_trajAnd_w_State2w_ppSolOrGuess(self, f_traj, w_state): #pp stands for py_panther
		w_p_ctrl_pts,knots_p=self.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(f_traj, w_state)
		w_y_ctrl_pts,knots_y=self.f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_traj, w_state)

		#Create and fill solOrGuess object
		w_sol_or_guess=py_panther.solOrGuess();

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

	def getTrajFromAction(self, action, index_traj):
		return action[index_traj,:].reshape(1,-1)

	def f_trajAnd_w_State2wBS(self, f_traj, w_state, no_deriv=False):


		w_pos_ctrl_pts,_=self.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(f_traj, w_state)
		w_yaw_ctrl_pts,_=self.f_trajAnd_w_State2_w_yaw_ctrl_pts_and_knots(f_traj, w_state)
		total_time=self.getTotalTimeTraj(f_traj)


		# print(f"f_action={f_action}")
		# print(f"total_time={total_time}")
		w_posBS = MyClampedUniformBSpline(0.0, total_time, self.deg_pos, self.dim, self.num_seg, w_pos_ctrl_pts, no_deriv) #def __init__():[BSpline(knots_pos, w_pos_ctrl_pts[0,:], self.deg_pos)

		w_yawBS = MyClampedUniformBSpline(0.0, total_time, self.deg_yaw, 1, self.num_seg, w_yaw_ctrl_pts, no_deriv)
		
		return w_posBS, w_yawBS

	def f_trajAnd_f_State2fBS(self, f_traj, f_state, no_deriv=False):
		assert np.linalg.norm(f_state.w_pos)<1e-7, "The pos should be zero"
		assert np.linalg.norm(f_state.w_yaw)<1e-7, "The yaw should be zero"
		f_posBS, f_yawBS = self.f_trajAnd_w_State2wBS(f_traj, f_state, no_deriv)
		return f_posBS, f_yawBS

	def solOrGuess2traj(self, sol_or_guess):
		traj=np.array([[]]);

		#Append position control points
		for i in range(3,len(sol_or_guess.qp)-2):
			# print(sol_or_guess.qp[i].reshape(1,self.dim))
			traj=np.concatenate((traj, sol_or_guess.qp[i][0:self.dim].reshape(1,self.dim)), axis=1)

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
		action=np.empty((0,self.traj_size));
		for sol_or_guess in sols_or_guesses:
			traj=self.solOrGuess2traj(sol_or_guess)
			action=np.concatenate((action, traj), axis=0)

		assert action.shape==self.getActionShape()
		return action

	def guesses2action(self, guesses):
		action=np.empty((0,self.traj_size));
		for guess in guesses:
			traj=self.solOrGuess2traj(guess)
			action=np.concatenate((action, traj), axis=0)
		return action
