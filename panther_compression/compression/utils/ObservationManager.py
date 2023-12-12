import numpy as np
import py_panther

from .yaml_utils import readPANTHERparams
from .utils import normalize, wrapInmPiPi
from .State import State
from .ObstaclesManager import ObstaclesManager

from colorama import Fore, Style

class ObservationManager():
	def __init__(self, dim=3):
		self.dim = dim
		self.obsm=ObstaclesManager(dim=dim);
		#Observation =       [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
		#
		# Where f_ctrl_pts_oi = [cp0.transpose(), cp1.transpose(), ...]
		self.observation_size= self.dim + self.dim + 1 + self.dim + self.obsm.getSizeAllObstacles();

		params=readPANTHERparams();

		self.v_max=np.array(params["v_max"]).reshape(3,1);
		self.a_max=np.array(params["a_max"]).reshape(3,1);
		self.j_max=np.array(params["j_max"]).reshape(3,1);
		self.ydot_max=params["ydot_max"];
		# self.max_dist2goal=params["max_dist2goal"];
		self.max_dist2obs=params["max_dist2obs"];
		self.max_side_bbox_obs=params["max_side_bbox_obs"];
		self.Ra=params["Ra"]
		ones13=np.ones((1,self.dim));
		#Note that the sqrt(self.dim) is needed because the expert/student plan in f_frame --> bouding ball around the box v_max, a_max,... 
		margin_v=np.sqrt(self.dim)
		margin_a=np.sqrt(self.dim)
		margin_ydot=1.5 
		self.normalization_constant=np.concatenate(
			(
				margin_v*self.v_max[:self.dim, :].T*ones13,
				margin_a*self.a_max[:self.dim, :].T*ones13,
				margin_ydot*self.ydot_max*np.ones((1,1)),
				self.Ra*ones13
			), axis=1
		)
		for i in range(self.obsm.getNumObs()):
			self.normalization_constant=np.concatenate((self.normalization_constant, self.max_dist2obs*np.ones((1,self.dim*self.obsm.getCPsPerObstacle())), self.max_side_bbox_obs*ones13), axis=1)

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
		# print(observation_normalized.shape)
		assert observation_normalized.shape == self.getObservationShape()

		return np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()

	def assertObsIsNormalized(self, observation_normalized, msg_before=""):

		if not self.obsIsNormalized(observation_normalized):
			print(msg_before+"The observation is not normalized")
			print(f"NORMALIZED VALUE={observation_normalized}")
			observation=self.denormalizeObservation(observation_normalized)
			print(f"VALUE={observation}")
			self.printObservation(observation);
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
		pos=np.zeros((self.dim,1))
		vel=self.getf_v(f_obs)
		accel=self.getf_a(f_obs)
		yaw=np.zeros((1,1))
		yaw_dot=self.getyaw_dot(f_obs)
		state=State(pos, vel, accel, yaw, yaw_dot)
		return state


	def getIndexStartObstacleI(self,i):
		return 10+(self.obsm.getCPsPerObstacle()+self.dim)*i

	def getCtrlPtsObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)
		ctrl_pts=[]; 
		num_cps_per_obstacle=self.obsm.getCPsPerObstacle();
		for j in range(num_cps_per_obstacle):
			index_start_cpoint=index_start_obstacle_i+self.dim*j
			cpoint_j=obs[0,index_start_cpoint:index_start_cpoint+self.dim].reshape(self.dim,1)
			ctrl_pts.append(cpoint_j)

		return ctrl_pts

	def getBboxInflatedObstacleI(self,obs,i):
		index_start_obstacle_i=self.getIndexStartObstacleI(i)

		tmp=index_start_obstacle_i+self.dim*self.obsm.getCPsPerObstacle()

		bbox_inflated=obs[0,tmp:tmp+4].reshape(self.dim,1)

		return bbox_inflated

	def getf_v(self, obs):
		return obs[0,0:self.dim].reshape((self.dim,1)) #Column vector

	def getf_a(self, obs):
		return obs[0,self.dim:2*self.dim].reshape((self.dim,1)) #Column vector

	def getyaw_dot(self, obs):
		return obs[0,2*self.dim].reshape((1,1)) 

	def getf_g(self, obs):
		return obs[0,2*self.dim+1:3*self.dim+1].reshape((self.dim,1)) #Column vector

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

	def getInit_f_StateFromObservation(self, obs):
		init_state=py_panther.state();  #Everything initialized as zero
		init_state.pos= np.array([[0.0],[0.0],[0.0]]);#Because it's in f frame
		init_state.vel= self.getf_v(obs);
		init_state.accel= self.getf_a(obs);
		init_state.yaw= 0.0  #Because it's in f frame
		init_state.dyaw = self.getyaw_dot(obs);
		return init_state

	def getFinal_f_StateFromObservation(self, obs):
		final_state=py_panther.state();  #Everything initialized as zero
		final_state.pos= self.getf_g(obs);
		# final_state.vel= 
		# final_state.accel= 
		# final_state.yaw= 
		# final_state.dyaw = 
		return final_state

	def getNanObservation(self):
		return np.full(self.getObservationShape(), np.nan)

	def isNanObservation(self, obs):
		return np.isnan(np.sum(obs))

	#Normalize in [-1,1]
	def normalizeObservation(self, observation):
		# print("Shape observation=", observation.shape)
		# print("Shape normalization_constant=", self.normalization_constant.shape)
		# print("obsm.getSizeAllObstacles()=", self.obsm.getSizeAllObstacles())

		observation_normalized=observation/self.normalization_constant;
		# assertIsNormalized(observation_normalized)
		# assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()
		return observation_normalized;

	def denormalizeObservation(self,observation_normalized):
		# assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all()

		# assertIsNormalized(observation_normalized)
		# assert np.logical_and(observation_normalized >= -1, observation_normalized <= 1).all(), f"observation_normalized= {observation_normalized}" 
		observation=observation_normalized*self.normalization_constant;
		return observation;

	# def denormalize(self, )

	def getObservationSize(self):
		return self.observation_size

	def getObservationShape(self):
		return (1,self.observation_size)

	def getRandomObservation(self):
		random_observation=self.denormalizeObservation(self.getRandomNormalizedObservation())
		return random_observation

	def getRandomNormalizedObservation(self):
		random_normalized_observation=np.random.uniform(-1,1, size=self.getObservationShape())
		return random_normalized_observation

	def get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self,w_state, w_gterm_pos, w_obstacles):

		f_gterm_pos=w_state.f_T_w * w_gterm_pos

		dist2gterm=np.linalg.norm(f_gterm_pos);
		f_g= min(dist2gterm-1e-4, self.Ra)*normalize(f_gterm_pos)
		#f_g= self.Ra*normalize(f_gterm_pos)
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

	def getNormalized_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self, w_state, w_gterm_pos, w_obstacles):
		f_observation=self.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(w_state, w_gterm_pos, w_obstacles)
		f_observationn=self.normalizeObservation(f_observation) #Observation normalized
		return f_observationn
