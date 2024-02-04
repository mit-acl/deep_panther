import os
import yaml
import numpy as np

import py_panther

def readPANTHERparams(additional_config=None):

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

	# check if some additional config is defined
	if additional_config is not None:
		params_yaml = {**params_yaml, **additional_config}

	return params_yaml;


def getPANTHERparamsAsCppStruct(additional_config=None):

	params_yaml=readPANTHERparams(additional_config=additional_config);

	params_yaml["b_T_c"]=np.array([[0, 0, 1, 0],
								  [-1, 0, 0, 0],
								  [0, -1, 0, 0],
								  [0, 0, 0, 1]])

	par=py_panther.parameters();

	for key in params_yaml:
		exec('%s = %s' % ('par.'+key, 'params_yaml["'+key+'"]')) #See https://stackoverflow.com/a/60487422/6057617 and https://www.pythonpool.com/python-string-to-variable-name/

	return par

