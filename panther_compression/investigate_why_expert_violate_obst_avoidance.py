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

traj_expert = np.array([[ 1.90339031,  2.18211999, -1.33990154,  3.6838341,   3.29023144, -1.93637244,
   5.20031786,  2.19353318, -0.53288827,  5.63669461,  0.01027185,  1.61969445,
  -0.61044954, -1.37475992, -2.1390703,  -2.90338068, -3.66769104,  4.58586231]])
w_posBS_expert, w_yawBS_expert= venv.am.f_trajAnd_w_State2wBS(traj_expert, State(p0, v0, a0, y0, ydot0))

f_obs = np.array([[ 0.,          0.,          0.,          0.,          0.,          0.,
   0.,          5.61144409,  0.,          1.64846655,  2.85507983,  0.88096795,
  -2.46870724,  3.17149195,  0.87474921, -3.0380747,   3.8001729,   1.22478663,
  -4.07835881,  3.80299854,  2.10886061, -2.01287786,  2.96808538,  2.17706717,
  -0.95580791,  2.29786189,  1.14409856, -3.37488171,  2.50698373,  0.03513949,
  -3.69475794,  3.20431133,  0.039445,   -1.24662446,  3.34564989,  0.7785396,
  -1.41618232,  3.26220083,  1.13565258, -1.92984885,  1.42816106,  1.34055598,
   1.36956824,  2.85507983,  0.88096795, -2.46870724,  3.17149195,  0.87474921,
  -3.0380747,   3.8001729,   1.22478663, -4.07835881,  3.80299854,  2.10886061,
  -2.01287786,  2.96808538,  2.17706717, -0.95580791,  2.29786189,  1.14409856,
  -3.37488171,  2.50698373,  0.03513949, -3.69475794,  3.20431133,  0.039445,
  -1.24662446,  3.34564989,  0.7785396,  -1.41618232,  3.26220083,  1.13565258,
  -1.92984885,  1.42816106 , 1.34055598,  1.36956824]]
)

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
