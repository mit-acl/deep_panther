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
	# params_yaml = dict(params_yaml_1.items() + params_yaml_2.items()) #Doesn't work in Python 3
	params_yaml = {**params_yaml_1, **params_yaml_2}                        # NOTE: Python 3.5+ ONLY


	# print a



	params_yaml["b_T_c"]=np.array([[0, 0, 1, 0],
    	                          [-1, 0, 0, 0],
    	                          [0, -1, 0, 0],
    	                          [0, 0, 0, 1]])

	# my_SolverIpopt=py_panther.SolverIpopt();
	par_=py_panther.parameters();

	for key in params_yaml:
		exec('%s = %s' % ('par_.'+key, 'params_yaml["'+key+'"]')) #See https://stackoverflow.com/a/60487422/6057617 and https://www.pythonpool.com/python-string-to-variable-name/
		#par_.XXX = params_yaml["XXX"]

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
		print ("i=",i)
		ctrl_pts.append(np.array([[0],[0],[0]]))

	obstacle.ctrl_pts = ctrl_pts

	obstacles=[obstacle];

	my_SolverIpopt.setObstaclesForOpt(obstacles);

	my_SolverIpopt.optimize();