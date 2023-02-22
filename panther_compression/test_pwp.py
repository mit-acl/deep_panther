# benchmark studies for yaw comparison

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from imitation.data import rollout, types
from compression.policies.ExpertPolicy import ExpertPolicy
from imitation.util import util
from compression.utils.other import ObservationManager, ActionManager, CostComputer, State, GTermManager
import rosbag
import rospy
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray
from compression.utils.other import ActionManager, ObservationManager, GTermManager, State, ObstaclesManager, getPANTHERparamsAsCppStruct, computeTotalTime, getObsAndGtermToCrossPath, posAccelYaw2TfMatrix
from compression.utils.other import TfMatrix2RosQuatAndVector3, TfMatrix2RosPose
from compression.utils.other import CostComputer
from compression.utils.other import MyClampedUniformBSpline
from compression.utils.other import listOf3dVectors2numpy3Xmatrix
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path
from os.path import exists
from mpl_toolkits.mplot3d import Axes3D
import math
import yaml
import rospkg

if __name__ == "__main__":

	### params for venv
	ENV_NAME = 	"my-environment-v1" # you shouldn't change the name
	num_envs = 	1
	seed = 	1
	###

	### policies
	expert_policy = ExpertPolicy()
	###

	### get vectorized environment
	# venv = util.make_env(env_name=ENV_NAME, n_envs=num_envs, seed=seed, parallel=False)
	venv = gym.make(ENV_NAME)
	venv.seed(seed)
	###

	### get observation and actions
	obstacle_position = np.array([[2.5],[0.0],[1.0]])
	goal_position = np.array([[10.0],[0.0],[1.0]])
	venv.changeConstantObstacleAndGtermPos(obstacle_position, goal_position)
	obs = venv.reset()
	acts_n = expert_policy.predict(obs, deterministic=False)
	###

	### get BS for expert (ref: MyEnvironment.py)
	acts_n_expert=acts_n[0].reshape(venv.am.getActionShape()) 
	venv.am.assertAction(acts_n_expert)
	venv.am.assertActionIsNormalized(acts_n_expert)
	acts_expert = venv.am.denormalizeAction(acts_n_expert)

	# print("(make sure all the trajs are the same) acts_expert: ", acts_expert)

	traj_expert=venv.am.getTrajFromAction(acts_expert, 0)
	p0=np.array([[0.0],[0.0],[1.0]])
	v0=np.array([[0.0],[0.0],[0.0]])
	a0=np.array([[0.0],[0.0],[0.0]])
	y0=np.array([[0.0]])
	ydot0=np.array([[0.0]])
	w_posBS_expert, w_yawBS_expert= venv.am.f_trajAnd_w_State2wBS(traj_expert, State(p0, v0, a0, y0, ydot0))
	w_pos_ctrl_pts, knots_pos = venv.am.f_trajAnd_w_State2_w_pos_ctrl_pts_and_knots(traj_expert, State(p0, v0, a0, y0, ydot0))

	print(knots_pos)
	print(knots_pos.shape)
	sys.exit(0)	


	### get BS
	acts_n_clf = venv.cfys.substituteWithClosedFormYaw(acts_n_expert, venv.w_state, venv.w_obstacles) #acts_n, w_init_state, w_obstacles
	venv.am.assertAction(acts_n_clf)
	venv.am.assertActionIsNormalized(acts_n_clf)
	acts_clf = venv.am.denormalizeAction(acts_n_clf)
	traj_clf=venv.am.getTrajFromAction(acts_clf, 0)
	w_posBS_clf, w_yawBS_clf= venv.am.f_trajAnd_w_State2wBS(traj_clf, State(p0, v0, a0, y0, ydot0))

	### plot

	## pos
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	xx = np.linspace(w_posBS_expert.getT0(), w_posBS_expert.getTf(), 100)
	ax.set_title('3d pos')

	# trajs
	ax.plot(w_posBS_expert.pos_bs[0](xx), w_posBS_expert.pos_bs[1](xx), w_posBS_expert.pos_bs[2](xx), lw=4, alpha=0.7, label='traj')
	# box
	f_obs = venv.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(venv.w_state, venv.gm.get_w_GTermPos(), venv.w_obstacles)
	obstacles=venv.om.getObstacles(f_obs)
	x,y,z = get_cube()
	x *= obstacles[0].bbox_inflated[0]
	y *= obstacles[0].bbox_inflated[1]
	z *= obstacles[0].bbox_inflated[2]
	x += obstacle_position[0]
	y += obstacle_position[1]
	z += obstacle_position[2]
	ax.plot_surface(x, y, z, alpha=0.2, color='brown')
	# yaw vector
	num_vectors = 10
	xx2 = np.linspace(w_posBS_expert.getT0(), w_posBS_expert.getTf(), num_vectors)
	initialized = False # for quiver label
	for xx2_i in xx2:
		# for expert
		xs_expert = w_posBS_expert.pos_bs[0](xx2_i)
		ys_expert = w_posBS_expert.pos_bs[1](xx2_i)
		zs_expert = w_posBS_expert.pos_bs[2](xx2_i)
		yaw_expert = w_yawBS_expert.pos_bs[0](xx2_i)
		xe_expert = 1.0 * np.cos(yaw_expert)
		ye_expert = 1.0 * np.sin(yaw_expert)
		ze_expert = 0
		# for clf
		xs_clf = w_posBS_clf.pos_bs[0](xx2_i)
		ys_clf = w_posBS_clf.pos_bs[1](xx2_i)
		zs_clf = w_posBS_clf.pos_bs[2](xx2_i)
		yaw_clf = w_yawBS_clf.pos_bs[0](xx2_i)
		xe_clf = 1.0 * np.cos(yaw_clf)
		ye_clf = 1.0 * np.sin(yaw_clf)
		ze_clf = 0
		# for nlp
		xs_nlp = w_posBS_nlp.pos_bs[0](xx2_i)
		ys_nlp = w_posBS_nlp.pos_bs[1](xx2_i)
		zs_nlp = w_posBS_nlp.pos_bs[2](xx2_i)
		yaw_nlp = w_yawBS_nlp.pos_bs[0](xx2_i)
		xe_nlp = 1.0 * np.cos(yaw_nlp)
		ye_nlp = 1.0 * np.sin(yaw_nlp)
		ze_nlp = 0
		if not initialized:
			ax.quiver(xs_expert, ys_expert, zs_expert, xe_expert, ye_expert, ze_expert, color='red', label='expert')
			ax.quiver(xs_clf, ys_clf, zs_clf, xe_clf, ye_clf, ze_clf, color='blue', label='clf')
			ax.quiver(xs_nlp, ys_nlp, zs_nlp, xe_nlp, ye_nlp, ze_nlp, color='green', label='nlp')
			initialized = True
		else:
			ax.quiver(xs_expert, ys_expert, zs_expert, xe_expert, ye_expert, ze_expert, color='red')
			ax.quiver(xs_clf, ys_clf, zs_clf, xe_clf, ye_clf, ze_clf, color='blue')
			ax.quiver(xs_nlp, ys_nlp, zs_nlp, xe_nlp, ye_nlp, ze_nlp, color='green')
	ax.grid(True)
	ax.legend(loc='best')
	ax.set_xlim(-2, 7)
	ax.set_ylim(-2, 7)
	ax.set_zlim(-2, 7)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_aspect('equal')
	fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_pos.png')
	# plt.show()

	sys.exit(0)