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
from compression.utils.other import ActionManager, ObservationManager, GTermManager, State, ObstaclesManager, getPANTHERparamsAsCppStruct, computeTotalTime, posAccelYaw2TfMatrix
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

import sys
import numpy as np
import time
from statistics import mean
import copy
from random import random, shuffle
from compression.utils.other import ActionManager, ObservationManager, ObstaclesManager, getPANTHERparamsAsCppStruct, ExpertDidntSucceed, computeTotalTime
from colorama import init, Fore, Back, Style
import py_panther
import math 
from gym import spaces
from joblib import Parallel, delayed
import multiprocessing


def printFailedOpt(info):
  print(" Called optimizer--> "+Style.BRIGHT+Fore.RED +"Failed"+ Style.RESET_ALL+". "+ info)

def printSucessOpt(info):
  print(" Called optimizer--> "+Style.BRIGHT+Fore.GREEN +"Success"+ Style.RESET_ALL+". "+ info)


##
## params
##

ENV_NAME = 	"my-environment-v1" # you shouldn't change the name
num_envs = 	1
seed = 	1

##
## get vectorized environment
##

venv = gym.make(ENV_NAME)
venv.seed(seed)

##
## initial condition
##

p0=np.array([[0.0],[0.0],[1.0]])
v0=np.array([[0.0],[0.0],[0.0]])
a0=np.array([[0.0],[0.0],[0.0]])
y0=np.array([[0.0]])
ydot0=np.array([[0.0]])

##
## get b-spline
##

primer_branch_traj_expert = np.array([[ 
  0.91224966,  0.82266163,  1.32147417,  2.32024804,  1.57076722,  1.9657202,
  3.76792436,  1.23108137,  0.98295584,  5.10536001,  0.12682231, -0.65702177,
  -0.20569983, -0.4017783,  -0.73733611, -0.78718934, -1.0430407,   3.47650815]])
w_posBS_expert, w_yawBS_expert= venv.am.f_trajAnd_w_State2wBS(primer_branch_traj_expert, State(p0, v0, a0, y0, ydot0))

f_obs = np.array([[ 
   0.,          0.,          0.,          0.,          0.,          0.,
   0.,          5.20167424,  0.,         -0.75670669,  4.98871496, -0.70308872,
   0.1861533,   5.30512708, -0.70930747, -0.38321416,  5.93380803, -0.35927004,
  -1.42349827,  5.93663367,  0.52480393,  0.64198267,  5.10172051,  0.5930105,
   1.69905263,  4.43149702, -0.43995811, -0.72002117,  4.64061886, -1.54891718,
  -1.0398974,   5.33794646, -1.54461168,  1.40823608,  5.47928502, -0.80551708,
   1.23867821,  5.39583596, -0.44840409,  0.72501169,  1.4,         1.4,
   1.        
  ]])

f_obs_n = venv.om.normalizeObservation(f_obs)

##
## compare it to master's trajectory
##

par_v_max = [2.5, 2.5, 2.5]
par_a_max = [5.5, 5.5, 5.5]
par_factor_alloc = 1.0

my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct())

init_state=venv.om.getInit_f_StateFromObservation(f_obs);        
final_state=venv.om.getFinal_f_StateFromObservation(f_obs);        
total_time=computeTotalTime(init_state, final_state, par_v_max, par_a_max, par_factor_alloc)
ExpertPolicy.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time)
ExpertPolicy.my_SolverIpopt.setFocusOnObstacle(True)
obstacles=venv.om.getObstacles(f_obs)
ExpertPolicy.my_SolverIpopt.setObstaclesForOpt(obstacles)
succeed=ExpertPolicy.my_SolverIpopt.optimize(True)

info=ExpertPolicy.my_SolverIpopt.getInfoLastOpt()

##
## Print results
##
if not succeed:
    printFailedOpt(info)
else:
    printSucessOpt(info)

best_solutions=ExpertPolicy.my_SolverIpopt.getBestSolutions()
action=venv.am.solsOrGuesses2action(best_solutions)
action_normalized=venv.am.normalizeAction(action)
venv.am.assertAction(action_normalized)

index_smallest_augmented_cost=venv.cost_computer.getIndexBestTraj(f_obs_n, action_normalized)
f_traj=venv.am.getTrajFromAction(action, index_smallest_augmented_cost)

print("master_branch_traj_expert", f_traj)

# master_branch_traj_expert [[ 1.41127246e+00  4.94472252e-03 -1.27355942e-01  2.47619666e+00
#    1.02452388e-02 -3.66909040e-01  3.53830662e+00  9.06570997e-03
#   -6.15215741e-01  4.95261100e+00  6.43649333e-04 -7.55919312e-01
#   -1.30022058e-01 -9.23626866e-02 -2.67463558e-01 -5.36647659e-02
#   -6.44980348e-01  3.07028260e+00]]


exit()

for i in range(1):
    f_posObs_ctrl_pts = listOf3dVectors2numpy3Xmatrix(CostComputer.om.getCtrlPtsObstacleI(f_obs, i))
    inflated_bbox = CostComputer.om.getBboxInflatedObstacleI(f_obs, i)
    f_posObstBS = MyClampedUniformBSpline(0.0, 6.0, 3, 3, 7, f_posObs_ctrl_pts, True)


##
## pot
##

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xx_exp = np.linspace(w_posBS_expert.getT0(), w_posBS_expert.getTf(), 100)
xx_stu = []
ax.set_title('3d pos')
ax.plot(w_posBS_expert.pos_bs[0](xx_exp), w_posBS_expert.pos_bs[1](xx_exp), w_posBS_expert.pos_bs[2](xx_exp), lw=4, alpha=0.7, label='expert traj')
ax.plot(f_posObstBS.pos_bs[0](xx_exp), f_posObstBS.pos_bs[1](xx_exp), f_posObstBS.pos_bs[2](xx_exp), lw=4, alpha=0.7, label='obstacle')
ax.grid(True)
ax.legend(loc='best')
ax.set_xlim(-2, 7)
ax.set_ylim(-2, 7)
ax.set_zlim(-2, 7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_aspect('equal')
plt.show()
