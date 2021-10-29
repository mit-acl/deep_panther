import numpy as np
import py_panther
import yaml
import os


if __name__ == '__main__':


	my_solOrGuess=py_panther.solOrGuess();
	v1 = np.array([[1], [3], [5]])
	v2 = np.array([[2], [4], [6]])
	my_solOrGuess.qp=[v1,v2]
	my_solOrGuess.printInfo()

	params_yaml_1=[];
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../param/panther.yaml', "r") as stream:
	    try:
	        params_yaml_1=yaml.safe_load(stream)
	    except yaml.YAMLError as exc:
	        print(exc)

	params_yaml_2=[];
	with open(os.path.dirname(os.path.abspath(__file__)) + '/../matlab/casadi_generated_files/params_casadi.yaml', "r") as stream:
	    try:
	        params_yaml_2=yaml.safe_load(stream)
	    except yaml.YAMLError as exc:
	        print(exc)
	params_yaml = dict(params_yaml_1.items() + params_yaml_2.items())


	# print a



	params_yaml["b_T_c"]=np.array([[0, 0, 1, 0],
    	                          [-1, 0, 0, 0],
    	                          [0, -1, 0, 0],
    	                          [0, 0, 0, 1]])

	# my_SolverIpopt=py_panther.SolverIpopt();
	par_=py_panther.parameters();

	par_.use_ff =                           params_yaml["use_ff"]
	par_.visual =                           params_yaml["visual"]
	par_.color_type =                       params_yaml["color_type"]
	par_.n_agents =                         params_yaml["n_agents"]
	par_.num_of_trajs_per_replan =          params_yaml["num_of_trajs_per_replan"]
	par_.dc =                               params_yaml["dc"]
	par_.goal_radius =                      params_yaml["goal_radius"]
	par_.drone_radius =                     params_yaml["drone_radius"]
	par_.Ra =                               params_yaml["Ra"]
	par_.impose_FOV_in_trajCB =             params_yaml["impose_FOV_in_trajCB"]
	par_.stop_time_when_replanning =        params_yaml["stop_time_when_replanning"]
	par_.replanning_trigger_time =          params_yaml["replanning_trigger_time"]
	par_.replanning_lookahead_time =        params_yaml["replanning_lookahead_time"]
	par_.max_runtime_octopus_search =       params_yaml["max_runtime_octopus_search"]
	par_.fov_x_deg =                        params_yaml["fov_x_deg"]
	par_.fov_y_deg =                        params_yaml["fov_y_deg"]
	par_.fov_depth =                        params_yaml["fov_depth"]
	par_.angle_deg_focus_front =            params_yaml["angle_deg_focus_front"]
	par_.x_min =                            params_yaml["x_min"]
	par_.x_max =                            params_yaml["x_max"]
	par_.y_min =                            params_yaml["y_min"]
	par_.y_max =                            params_yaml["y_max"]
	par_.z_min =                            params_yaml["z_min"]
	par_.z_max =                            params_yaml["z_max"]
	par_.ydot_max =                         params_yaml["ydot_max"]
	par_.v_max =                            params_yaml["v_max"]
	par_.a_max = 							params_yaml["a_max"]
	par_.j_max = 							params_yaml["j_max"]
	par_.factor_alpha =                     params_yaml["factor_alpha"]
	par_.max_seconds_keeping_traj =         params_yaml["max_seconds_keeping_traj"]
	par_.a_star_samp_x =                    params_yaml["a_star_samp_x"]
	par_.a_star_samp_y =                    params_yaml["a_star_samp_y"]
	par_.a_star_samp_z =                    params_yaml["a_star_samp_z"]
	par_.a_star_fraction_voxel_size =       params_yaml["a_star_fraction_voxel_size"]
	par_.a_star_bias =                      params_yaml["a_star_bias"]
	par_.res_plot_traj =                    params_yaml["res_plot_traj"]
	par_.factor_alloc =                     params_yaml["factor_alloc"]
	par_.alpha_shrink =                     params_yaml["alpha_shrink"]
	par_.norminv_prob =                     params_yaml["norminv_prob"]
	par_.disc_pts_per_interval_oct_search = params_yaml["disc_pts_per_interval_oct_search"]
	par_.c_smooth_yaw_search =              params_yaml["c_smooth_yaw_search"]
	par_.c_visibility_yaw_search =          params_yaml["c_visibility_yaw_search"]
	par_.c_maxydot_yaw_search =             params_yaml["c_maxydot_yaw_search"]
	par_.c_pos_smooth =                     params_yaml["c_pos_smooth"]
	par_.c_yaw_smooth =                     params_yaml["c_yaw_smooth"]
	par_.c_fov =                            params_yaml["c_fov"]
	par_.c_final_pos =                      params_yaml["c_final_pos"]
	par_.c_final_yaw =                      params_yaml["c_final_yaw"]
	par_.c_total_time =                     params_yaml["c_total_time"]
	par_.print_graph_yaw_info =             params_yaml["print_graph_yaw_info"]
	par_.fitter_total_time =                params_yaml["fitter_total_time"]
	par_.mode =                             params_yaml["mode"]
	par_.b_T_c =                            params_yaml["b_T_c"]
	par_.basis =                            params_yaml["basis"]
	par_.num_max_of_obst =                  params_yaml["num_max_of_obst"]
	par_.num_seg =                          params_yaml["num_seg"]
	par_.deg_pos =                          params_yaml["deg_pos"]
	par_.deg_yaw =                          params_yaml["deg_yaw"]
	par_.num_of_yaw_per_layer =             params_yaml["num_of_yaw_per_layer"]
	par_.fitter_num_samples =               params_yaml["fitter_num_samples"]
	par_.fitter_num_seg =                   params_yaml["fitter_num_seg"]
	par_.fitter_deg_pos =                   params_yaml["fitter_deg_pos"]
	par_.sampler_num_samples =              params_yaml["sampler_num_samples"]


	my_SolverIpopt=py_panther.SolverIpopt(par_);

	init_state=py_panther.state();
	init_state.pos=np.array([[-10], [0], [0]]);
	init_state.vel=np.array([[0], [0], [0]]);
	init_state.accel=np.array([[0], [0], [0]]);

	final_state=py_panther.state();
	final_state.pos=np.array([[10], [0], [0]]);
	final_state.vel=np.array([[0], [0], [0]]);
	final_state.accel=np.array([[0], [0], [0]]);


	my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, 2.0);
	my_SolverIpopt.setFocusOnObstacle(True);

	obstacle=py_panther.obstacleForOpt()
	obstacle.bbox_inflated=np.array([[0.5],[0.5],[0.5]])

	ctrl_pts=[]; #np.array([[],[],[]])
	for i in range(int(par_.fitter_num_seg + par_.fitter_deg_pos)):
		print i
		ctrl_pts.append(np.array([[0],[0],[0]]))

	obstacle.ctrl_pts = ctrl_pts

	obstacles=[obstacle];

	my_SolverIpopt.setObstaclesForOpt(obstacles);

	my_SolverIpopt.optimize();
	

	