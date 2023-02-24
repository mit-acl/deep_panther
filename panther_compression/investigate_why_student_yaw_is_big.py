# benchmark studies for yaw comparison

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from imitation.data import rollout, types
from compression.policies.ExpertPolicy import ExpertPolicy
from compression.policies.StudentPolicy import StudentPolicy
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
from imitation.algorithms import bc
import torch as th
from numpy import load
from scipy.optimize import linear_sum_assignment
from compression.utils.other import getPANTHERparamsAsCppStruct

# not sure but traj is short..

if __name__ == "__main__":

	##
	## params
	##

	yaw_scaling = getPANTHERparamsAsCppStruct().yaw_scaling
	use_demo_obs_acts = False
	calculate_loss = False
	ENV_NAME = 	"my-environment-v1" # you shouldn't change the name
	num_envs = 	1
	seed = 	1

	##
	## get demonstration data
	##

	rospack = rospkg.RosPack()
	path_panther=rospack.get_path('panther')
	if use_demo_obs_acts:
		data = load(path_panther[:-8]+'/panther_compression/evals/tmp_dagger/2/demos/round-004/dagger-demo-20230223_212805_2379ee.npz')
		lst = data.files

	##
	## get vectorized environment
	##

	venv = gym.make(ENV_NAME)
	venv.seed(seed)

	##
	## policies
	##

	expert_policy = ExpertPolicy()
	path_file=path_panther[:-8]+"/panther_compression/evals/tmp_dagger/2/intermediate_policy_round0_log6.pt"
	student_policy = bc.reconstruct_policy(path_file) #got this from test_infeasibility_ipopt.py

	##
	## get observation and actions
	##
	
	## note that demo's obs and acts are tensor and visualization is not supported yet in this file (you can see loss tho)
	if use_demo_obs_acts:
		obs = th.as_tensor(data["obs"]).detach()
		acts_expert = th.as_tensor(data["acts"]).detach()
	else:
		obstacle_position = np.array([[2.5],[0.0],[1.0]])
		goal_position = np.array([[5.0],[0.0],[1.0]])
		venv.changeConstantObstacleAndGtermPos(obstacle_position, goal_position)
		obs = venv.reset()

	##
	## initial condition
	##

	p0=np.array([[0.0],[0.0],[1.0]])
	v0=np.array([[0.0],[0.0],[0.0]])
	a0=np.array([[0.0],[0.0],[0.0]])
	y0=np.array([[0.0]])
	ydot0=np.array([[0.0]])

	##
	## get BS for expert (ref: MyEnvironment.py)
	##

	acts_n_expert_np = expert_policy.predict(obs, deterministic=False)
	acts_n_expert_np = acts_n_expert_np[0].reshape(venv.am.getActionShape())
	venv.am.assertAction(acts_n_expert_np)
	venv.am.assertActionIsNormalized(acts_n_expert_np)
	acts_expert_np = venv.am.denormalizeAction(acts_n_expert_np)
	# traj_expert = venv.am.getTrajFromAction(np.array([acts_expert.detach().numpy()[0,1]]), 0)
	traj_expert = venv.am.getTrajFromAction(acts_expert_np, 0)
	w_posBS_expert, w_yawBS_expert= venv.am.f_trajAnd_w_State2wBS(traj_expert, State(p0, v0, a0, y0, ydot0))

	##
	## get BS for student
	##

	acts_n = student_policy.predict(obs, deterministic=False)
	# acts_student = student_policy.forward(obs[0], deterministic=True)
	# acts_student = acts_student.detach().numpy()[0]
	# scale back down the acts_student
	# acts_n = student_policy.forward(obs[0], deterministic=True)
	acts_n_student=(acts_n[0].reshape(venv.am.getActionShape())) 
	venv.am.assertAction(acts_n_student)
	venv.am.assertActionIsNormalized(acts_n_student)
	acts_student = venv.am.denormalizeAction(acts_n_student)

	# scale back down the acts_student
	acts_student[:,venv.am.traj_size_pos_ctrl_pts:venv.am.traj_size_pos_ctrl_pts+venv.am.traj_size_yaw_ctrl_pts] = acts_student[:,venv.am.traj_size_pos_ctrl_pts:venv.am.traj_size_pos_ctrl_pts+venv.am.traj_size_yaw_ctrl_pts]/yaw_scaling
	traj_student = venv.am.getTrajFromAction(acts_student, 0)
	w_posBS_student, w_yawBS_student= venv.am.f_trajAnd_w_State2wBS(traj_student, State(p0, v0, a0, y0, ydot0))

	##
	## get ydot_max
	##

	rospack = rospkg.RosPack()
	with open(rospack.get_path('panther')+'/param/panther.yaml', 'rb') as f:
	    conf = yaml.safe_load(f.read())    # load the config file
	yaw_dot_max = conf['ydot_max']

	##
	## calculate loss (ref:: bc.py)
	##

	## note that this is only for tensor obs acts (in case use_demos_obs_acts == True)
	if (calculate_loss):
		num_of_traj_per_action=list(acts_expert_np.shape)[1] #acts_expert_np.shape is [batch size, num_traj_action, size_traj]
		num_of_elements_per_traj=list(acts_expert_np.shape)[2] #acts_expert_np.shape is [batch size, num_traj_action, size_traj]
		batch_size=list(acts_expert_np.shape)[0]

		# not sure why but list(acts_expert_np.shape)[0] is different from list(acts_student.shape)[0]
		acts_student = acts_student[:batch_size,:,:]

		distance_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action); 
		distance_pos_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action); 
		distance_yaw_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action); 
		distance_time_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action); 
		distance_pos_matrix_within_expert= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action); 

		for i in range(num_of_traj_per_action):
			for j in range(num_of_traj_per_action):
				expert_i = acts_expert_np[:,i,:].float() #All the elements
				student_j = acts_student[:,j,:].float() #All the elements

				expert_pos_i = acts_expert_np[:,i,0:venv.am.traj_size_pos_ctrl_pts].float()
				student_pos_j = acts_student[:,j,0:venv.am.traj_size_pos_ctrl_pts].float()

				expert_yaw_i = acts_expert_np[:,i,venv.am.traj_size_pos_ctrl_pts:(venv.am.traj_size_pos_ctrl_pts+venv.am.traj_size_yaw_ctrl_pts)].float()
				student_yaw_j = acts_student[:,j,venv.am.traj_size_pos_ctrl_pts:(venv.am.traj_size_pos_ctrl_pts+venv.am.traj_size_yaw_ctrl_pts)].float()

				expert_time_i = acts_expert_np[:,i,-1:].float(); #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false
				student_time_j = acts_student[:,j,-1:].float() #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false

				distance_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_i, student_j), dim=1)
				distance_pos_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, student_pos_j), dim=1)
				distance_yaw_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_yaw_i, student_yaw_j), dim=1)
				distance_time_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_time_i, student_time_j), dim=1)

				#This is simply to delete the trajs from the expert that are repeated
				expert_pos_j = acts_expert_np[:,j,0:venv.am.traj_size_pos_ctrl_pts].float();
				distance_pos_matrix_within_expert[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, expert_pos_j), dim=1)

		is_repeated=th.zeros(batch_size, num_of_traj_per_action, dtype=th.bool)

		for index_batch in range(batch_size):         
			# cost_matrix_numpy=distance_pos_matrix_numpy[index_batch,:,:];
			cost_matrix=distance_pos_matrix[index_batch,:,:];
			map2RealRows=np.array(range(num_of_traj_per_action))
			map2RealCols=np.array(range(num_of_traj_per_action))

			rows_to_delete=[]
			for i in range(num_of_traj_per_action): #for each row (expert traj)
			    # expert_prob=th.round(acts[index_batch, i, -1]) #this should be either 1 or -1
			    # if(expert_prob==-1): 
			    if(is_repeated[index_batch,i]==True): 
			        #Delete that row
			        rows_to_delete.append(i)

			# print(f"Deleting index_batch={index_batch}, rows_to_delete={rows_to_delete}")
			# cost_matrix_numpy=np.delete(cost_matrix_numpy, rows_to_delete, axis=0)
			cost_matrix=cost_matrix[is_repeated[index_batch,:]==False]   #np.delete(cost_matrix_numpy, rows_to_delete, axis=0)
			cost_matrix_numpy=cost_matrix.cpu().detach().numpy()

		map2RealRows=np.delete(map2RealRows, rows_to_delete, axis=0)
		A_matrix=th.ones(batch_size, num_of_traj_per_action, num_of_traj_per_action)
		row_indexes, col_indexes = linear_sum_assignment(cost_matrix_numpy)
		for row_index, col_index in zip(row_indexes, col_indexes):
		    A_matrix[index_batch, map2RealRows[row_index], map2RealCols[col_index]]=1
		num_nonzero_A=th.count_nonzero(A_matrix)
		pos_loss=th.sum(A_matrix*distance_pos_matrix)/num_nonzero_A
		yaw_loss=th.sum(A_matrix*distance_yaw_matrix)/num_nonzero_A
		time_loss=th.sum(A_matrix*distance_time_matrix)/num_nonzero_A
		print("distance_pos_matrix", distance_pos_matrix)
		print("distance_yaw_matrix", distance_yaw_matrix)
		print("pos_loss ", pos_loss)
		print("yaw_loss ", yaw_loss)
		print("time_loss ", time_loss)


	##
	## plot
	##

	def get_cube():   
		phi = np.arange(1,10,2)*np.pi/4
		Phi, Theta = np.meshgrid(phi, phi)

		x = np.cos(Phi)*np.sin(Theta)
		y = np.sin(Phi)*np.sin(Theta)
		z = np.cos(Theta)/np.sqrt(2)
		return x,y,z

	##
	## pos
	##

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	xx = np.linspace(w_posBS_expert.getT0(), w_posBS_expert.getTf(), 100)
	ax.set_title('3d pos')

	##
	## trajs
	##

	ax.plot(w_posBS_expert.pos_bs[0](xx), w_posBS_expert.pos_bs[1](xx), w_posBS_expert.pos_bs[2](xx), lw=4, alpha=0.7, label='expert traj')
	ax.plot(w_posBS_student.pos_bs[0](xx), w_posBS_student.pos_bs[1](xx), w_posBS_student.pos_bs[2](xx), lw=4, alpha=0.7, label='student traj')
	
	##
	## box
	##

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
	
	##
	## yaw vector
	##

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
		# for student
		xs_student = w_posBS_student.pos_bs[0](xx2_i)
		ys_student = w_posBS_student.pos_bs[1](xx2_i)
		zs_student = w_posBS_student.pos_bs[2](xx2_i)
		yaw_student = w_yawBS_student.pos_bs[0](xx2_i)
		xe_student = 1.0 * np.cos(yaw_student)
		ye_student = 1.0 * np.sin(yaw_student)
		ze_student = 0
		if not initialized:
			ax.quiver(xs_expert, ys_expert, zs_expert, xe_expert, ye_expert, ze_expert, color='red', label='expert')
			ax.quiver(xs_student, ys_student, zs_student, xe_student, ye_student, ze_student, color='blue', label='student')
			initialized = True
		else:
			ax.quiver(xs_expert, ys_expert, zs_expert, xe_expert, ye_expert, ze_expert, color='red')
			ax.quiver(xs_student, ys_student, zs_student, xe_student, ye_student, ze_student, color='blue')
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

	##
	## yaw
	##
	
	fig, ax = plt.subplots(2,1)
	fig.tight_layout(pad=2.0)
	xx = np.linspace(w_yawBS_expert.getT0(), w_yawBS_expert.getTf(), 100)
	ax[0].set_title('Yaw comparison')
	ax[0].plot(xx, w_yawBS_expert.pos_bs[0](xx), 'r-', lw=4, alpha=0.7, label='expert yaw')
	ax[0].plot(xx, w_yawBS_student.pos_bs[0](xx), 'b-', lw=4, alpha=0.7, label='student yaw')
	ax[0].grid(True)
	ax[0].legend(loc='best')

	ax[1].set_title('Derivative of Yaw comparison')
	ax[1].plot(xx, w_yawBS_expert.vel_bs[0](xx), 'r-', lw=4, alpha=0.7, label='expert dyaw')
	ax[1].plot(xx, w_yawBS_student.vel_bs[0](xx), 'b-', lw=4, alpha=0.7, label='student dyaw')
	ax[1].grid(True)
	ax[1].legend(loc='best')
	fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.png')
	plt.show()

	##
	## check constraint violation
	##

	exp_violated = False
	stu_violated = False
	for xx_i in xx:
		if abs(w_yawBS_expert.vel_bs[0](xx_i)) > yaw_dot_max and not exp_violated:
			print('expert violates ydot constraint')
			exp_violated = True
		if abs(w_yawBS_student.vel_bs[0](xx_i)) > yaw_dot_max and not stu_violated:
			print('student violates ydot constraint') 
			stu_violated = True

	sys.exit(0)