import numpy as np
import random
import py_panther

from .yaml_utils import readPANTHERparams

class ObstaclesManager():

    #The reason to create this here (instead of in the constructor) is that C++ objects created with pybind11 cannot be pickled by default (pickled is needed when parallelizing)
    #See https://stackoverflow.com/a/68672/6057617
    #Other option would be to do this: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
    #Note that pickle is needed when saving the Student Policy (which has a ObservationManager, which has a ObstaclesManager, which has a Fitter )
	params=readPANTHERparams()
	fitter = py_panther.Fitter(params["fitter_num_samples"]);

	def __init__(self):
		self.num_obs=1;
		self.params=readPANTHERparams();
		# self.fitter_total_time=params["fitter_total_time"];
		self.fitter_num_seg=self.params["fitter_num_seg"];
		self.fitter_deg_pos=self.params["fitter_deg_pos"];
		self.fitter_total_time=self.params["fitter_total_time"];
		self.fitter_num_samples=self.params["fitter_num_samples"];
		# self.fitter = py_panther.Fitter(self.fitter_num_samples);
		self.newRandomPos();

	def newRandomPos(self):
		self.random_pos=np.array([[random.uniform(-4.0, 4.0)],[random.uniform(-4.0, 4.0)],[random.uniform(1.0,1.0)]]);
		self.random_offset=random.uniform(0.0, 10*np.pi)
		self.random_scale=np.array([[random.uniform(0.5, 4.0)],[random.uniform(0.5, 4.0)],[random.uniform(0.5, 4.0)]]);
		#self.random_pos=np.array([[2.5],[1.0],[1.0]]);

	def setPos(self, pos):
		self.random_pos=pos

	def getNumObs(self):
		return self.num_obs

	def getCPsPerObstacle(self):
		return self.fitter_num_seg + self.fitter_deg_pos

	def getSizeAllObstacles(self):
		#Size of the ctrl_pts + bbox
		return self.num_obs*(3*self.getCPsPerObstacle() + 3) 

	def getFutureWPosStaticObstacles(self):
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

	def getFutureWPosDynamicObstacles(self,t):
		w_obs=[];
		# trefoil=Trefoil(pos=self.random_pos, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset=0.0, slower=1.5);
		# novale=np.array([[4.0],[4.0],[1.0]]);
		# print(f"Using offset={self.random_offset}")
		# print(f"Using random_scale={self.random_scale}")

		###HACK TO GENERATE A STATIC OBSTACLE
		self.random_scale=np.zeros((3,1))
		####################################

		trefoil=Trefoil(pos=self.random_pos, scale=self.random_scale, offset=self.random_offset, slower=1.5);
		for i in range(self.num_obs):

			samples=[]
			for t_interm in np.linspace(t, t + self.fitter_total_time, num=self.fitter_num_samples):#.tolist():
				samples.append(trefoil.getPosT(t_interm))

			w_ctrl_pts_ob_list=ObstaclesManager.fitter.fit(samples)

			w_ctrl_pts_ob=listOf3dVectors2numpy3Xmatrix(w_ctrl_pts_ob_list)

			bbox_inflated=np.array([[0.8],[0.8], [0.8]])+2*self.params["drone_radius"]*np.ones((3,1));
			w_obs.append(Obstacle(w_ctrl_pts_ob, bbox_inflated))
		return w_obs;
