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
from compression.utils.other import ClosedFormYawSubstituter, NLPYawSubstituter
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path
from os.path import exists
from mpl_toolkits.mplot3d import Axes3D
import math
import yaml
import rospkg

# not sure but traj is short...
def save_in_bag(venv, acts):
	
	bag_name = '/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.bag'	
	# option = 'a' if exists(bag_name) else 'w' 
	option = 'a'
	bag = rosbag.Bag(bag_name, option)
	w_posBS_list=[]
	w_yawBS_list=[]
	
	w_state = venv.w_state
	f_obs = venv.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(venv.w_state, venv.gm.get_w_GTermPos(), venv.w_obstacles)

	for i in range(np.shape(acts)[0]): #For each row of action   
		f_traj=venv.am.getTrajFromAction(acts, i)
		w_posBS, w_yawBS= venv.am.f_trajAnd_w_State2wBS(f_traj, venv.w_state)
		w_posBS_list.append(w_posBS)
		w_yawBS_list.append(w_yawBS)

	time_now=rospy.Time.from_sec(venv.time_rosbag) #rospy.Time.from_sec(time.time());
	venv.time_rosbag=venv.time_rosbag+0.1;

	# with rosbag.Bag(name_file, option) as bag:
	f_v=venv.om.getf_v(f_obs)
	f_a=venv.om.getf_a(f_obs)
	yaw_dot=venv.om.getyaw_dot(f_obs)
	f_g=venv.om.getf_g(f_obs)
	obstacles=venv.om.getObstacles(f_obs)
	point_msg=PointStamped()

	point_msg.header.frame_id = "f";
	point_msg.header.stamp = time_now;
	point_msg.point.x=f_g[0,0]
	point_msg.point.y=f_g[1,0]
	point_msg.point.z=f_g[2,0]

	marker_array_msg=MarkerArray()

	for i in range(len(obstacles)):

		t0=venv.time
		tf=venv.time + np.max(acts[:,-1]) #venv.am.getTotalTime()#venv.par.fitter_total_time
		bspline_obs_i=MyClampedUniformBSpline(t0=t0, tf=tf, deg=venv.par.deg_pos, \
			                           dim=3, num_seg=venv.par.num_seg, ctrl_pts=listOf3dVectors2numpy3Xmatrix(obstacles[i].ctrl_pts) )

		id_sample=0
		num_samples=20
		for t_interm in np.linspace(t0, tf, num=num_samples).tolist():
			marker_msg=Marker();
			marker_msg.header.frame_id = "f";
			marker_msg.header.stamp = time_now;
			marker_msg.ns = "ns";
			marker_msg.id = id_sample;
			marker_msg.type = marker_msg.CUBE;
			marker_msg.action = marker_msg.ADD;
			pos=bspline_obs_i.getPosT(t_interm)
			marker_msg.pose.position.x = pos[0];
			marker_msg.pose.position.y = pos[1];
			marker_msg.pose.position.z = pos[2];
			marker_msg.pose.orientation.x = 0.0;
			marker_msg.pose.orientation.y = 0.0;
			marker_msg.pose.orientation.z = 0.0;
			marker_msg.pose.orientation.w = 1.0;
			marker_msg.scale.x = obstacles[0].bbox_inflated[0];
			marker_msg.scale.y = obstacles[0].bbox_inflated[1];
			marker_msg.scale.z = obstacles[0].bbox_inflated[2];
			marker_msg.color.a = 1.0*(num_samples-id_sample)/num_samples; 
			marker_msg.color.r = 0.0;
			marker_msg.color.g = 1.0;
			marker_msg.color.b = 0.0;
			marker_array_msg.markers.append(marker_msg)
			id_sample=id_sample+1

	tf_stamped=TransformStamped();
	tf_stamped.header.frame_id="world"
	tf_stamped.header.stamp=time_now
	tf_stamped.child_frame_id="f"
	rotation_ros, translation_ros=TfMatrix2RosQuatAndVector3(w_state.w_T_f)
	tf_stamped.transform.translation=translation_ros
	tf_stamped.transform.rotation=rotation_ros

	tf_msg=TFMessage()
	tf_msg.transforms.append(tf_stamped)
	for i in range(len(w_posBS_list)):

		w_posBS=w_posBS_list[i]
		w_yawBS=w_yawBS_list[i]

		t0=w_posBS.getT0();
		tf=w_posBS.getTf();

		traj_msg=Path()
		traj_msg.header.frame_id="world"
		traj_msg.header.stamp=time_now

		for t_i in np.arange(t0, tf, (tf-t0)/10.0):
		  pose_stamped=PoseStamped();
		  pose_stamped.header.frame_id="world"
		  pose_stamped.header.stamp=time_now
		  tf_matrix=posAccelYaw2TfMatrix(w_posBS.getPosT(t_i),w_posBS.getAccelT(t_i),w_yawBS.getPosT(t_i))
		  pose_stamped.pose=TfMatrix2RosPose(tf_matrix)
		  traj_msg.poses.append(pose_stamped)

		bag.write('/path'+str(i), traj_msg, time_now)

	bag.write('/g', point_msg, time_now)
	bag.write('/obs', marker_array_msg, time_now)
	bag.write('/tf', tf_msg, time_now)
	bag.close()

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

	### get BS for CLF
	acts_n_clf = venv.cfys.substituteWithClosedFormYaw(acts_n_expert, venv.w_state, venv.w_obstacles) #acts_n, w_init_state, w_obstacles
	venv.am.assertAction(acts_n_clf)
	venv.am.assertActionIsNormalized(acts_n_clf)
	acts_clf = venv.am.denormalizeAction(acts_n_clf)
	traj_clf=venv.am.getTrajFromAction(acts_clf, 0)
	w_posBS_clf, w_yawBS_clf= venv.am.f_trajAnd_w_State2wBS(traj_clf, State(p0, v0, a0, y0, ydot0))

	### get B-spline for NLP
	## get ydot_max
	rospack = rospkg.RosPack()
	with open(rospack.get_path('panther')+'/param/panther.yaml', 'rb') as f:
	    conf = yaml.safe_load(f.read())    # load the config file
	yaw_dot_max = conf['ydot_max']
	acts_n_nlp = venv.nlpys.substituteWithNLPYaw(acts_n_expert, venv.w_state, venv.w_obstacles, yaw_dot_max) #acts_n, w_init_state, w_obstacles
	venv.am.assertAction(acts_n_nlp)
	venv.am.assertActionIsNormalized(acts_n_nlp)
	acts_nlp = venv.am.denormalizeAction(acts_n_nlp)
	traj_nlp = venv.am.getTrajFromAction(acts_nlp, 0)
	w_posBS_nlp, w_yawBS_nlp= venv.am.f_trajAnd_w_State2wBS(traj_nlp, State(p0, v0, a0, y0, ydot0))

	### plot

	def get_cube():   
		phi = np.arange(1,10,2)*np.pi/4
		Phi, Theta = np.meshgrid(phi, phi)

		x = np.cos(Phi)*np.sin(Theta)
		y = np.sin(Phi)*np.sin(Theta)
		z = np.cos(Theta)/np.sqrt(2)
		return x,y,z

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

	## yaw
	fig, ax = plt.subplots(2,1)
	fig.tight_layout(pad=2.0)
	xx = np.linspace(w_yawBS_expert.getT0(), w_yawBS_expert.getTf(), 100)
	ax[0].set_title('Yaw comparison')
	ax[0].plot(xx, w_yawBS_expert.pos_bs[0](xx), 'r-', lw=4, alpha=0.7, label='expert yaw')
	ax[0].plot(xx, w_yawBS_clf.pos_bs[0](xx), 'b-', lw=4, alpha=0.7, label='clf yaw')
	ax[0].plot(xx, w_yawBS_nlp.pos_bs[0](xx), 'g-', lw=4, alpha=0.7, label='nlp yaw')
	ax[0].grid(True)
	ax[0].legend(loc='best')

	ax[1].set_title('Derivative of Yaw comparison')
	ax[1].plot(xx, w_yawBS_expert.vel_bs[0](xx), 'r-', lw=4, alpha=0.7, label='expert dyaw')
	ax[1].plot(xx, w_yawBS_clf.vel_bs[0](xx), 'b-', lw=4, alpha=0.7, label='clf dyaw')
	ax[1].plot(xx, w_yawBS_nlp.vel_bs[0](xx), 'g-', lw=4, alpha=0.7, label='nlp dyaw')
	ax[1].grid(True)
	ax[1].legend(loc='best')
	fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.png')
	# plt.show()

	## check constraint violation
	for xx_i in xx:
		if abs(w_yawBS_expert.vel_bs[0](xx_i)) > yaw_dot_max:
			print('expert violates ydot constraint')
		if abs(w_yawBS_clf.vel_bs[0](xx_i)) > yaw_dot_max:
			print('clf violates ydot constraint') 
		if abs(w_yawBS_nlp.vel_bs[0](xx_i)) > yaw_dot_max:
			print('nlp violates ydot constraint') 

	### save in bag
	# save_in_bag(venv, acts_expert)

	sys.exit(0)