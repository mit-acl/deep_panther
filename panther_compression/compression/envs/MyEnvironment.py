import gym
import sys
import numpy as np
import copy
from gym import spaces
from compression.utils.other import ActionManager, ObservationManager, GTermManager, State, ObstaclesManager, getPANTHERparamsAsCppStruct, computeTotalTime, posAccelYaw2TfMatrix
from compression.utils.other import TfMatrix2RosQuatAndVector3, TfMatrix2RosPose
from compression.utils.other import CostComputer
from compression.utils.other import MyClampedUniformBSpline
from compression.utils.other import listOf3dVectors2numpy3Xmatrix
from compression.utils.other import ClosedFormYawSubstituter
from colorama import init, Fore, Back, Style
import py_panther
##### For rosbag logging
import rosbag
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
import time
import rospy
from os.path import exists
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PointStamped
#####
from imitation.util import logger, util

import uuid

class MyEnvironment(gym.Env):
  """
    Custom Environment that follows gym interface
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__() #Equivalently, we could also write super(MyEnvironment, self).__init__()    , see 2nd paragraph of https://stackoverflow.com/a/576183

    print("Creating new environment!")
    
    self.verbose = False
    self.len_episode = 50     # Time steps [-] # is overwritten in policy_compression_train.py
    self.am=ActionManager()
    self.om=ObservationManager()
    self.obsm=ObstaclesManager()
    self.gm=GTermManager()
    self.cfys=ClosedFormYawSubstituter()
    self.cost_computer=CostComputer()
    self.action_shape=self.am.getActionShape()
    self.observation_shape=self.om.getObservationShape()
    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)
    self.color=Style.BRIGHT+Fore.YELLOW
    self.name=""
    self.par=getPANTHERparamsAsCppStruct()
    self.training_dt = self.par.training_dt
    self.use_clipping = self.par.use_clipping
    self.constant_obstacle_pos=None
    self.constant_gterm_pos=None
    self.record_bag=False
    self.time_rosbag=0
    self.name_bag=None
    self.force_done=False
    self.id=0
    self.num_goal_reached = 0
    self.time = 0.0
    self.prev_dist_current2goal = 0.0
    # self.my_SolverIpopt=py_panther.SolverIpopt(self.par)
    # self.reset()

  def __del__(self):
    # if(self.record_bag==True):
    #   self.bag.close();
    pass

  def changeConstantObstacleAndGtermPos(self, obstacle_pos, gterm_pos):
    self.constant_obstacle_pos=obstacle_pos
    self.constant_gterm_pos=gterm_pos

  def setID(self, data):
    self.id=data
    self.name=self.color+"  [Env "+str(self.id)+"]"+Style.RESET_ALL

  def startRecordBag(self):
    self.record_bag=True
    self.name_bag="training"+str(self.id)+".bag"

    # name_bag="training_"+str(uuid.uuid1())+".bag"
    # if(exists(name_bag)):
    #     option='a'
    # else:
    #     option='w'
    # print(f"option= {option}, name_bag={name_bag}")
    # self.bag=rosbag.Bag(name_bag, option)

  def printwithName(self,data):
    print(self.name+data)

  def printwithNameAndColor(self,data):
    print(self.name+self.color+data+Style.RESET_ALL)

  # def printFailedOpt(self):
  #     print(self.name+" Called optimizer--> "+Style.BRIGHT+Fore.RED +"Failed"+ Style.RESET_ALL)

  def seed(self, seed=None):
    """Set seed function in this environment and calls
    the openAi gym seed function"""
    np.random.seed(seed)
    super().seed(seed)

  def get_len_ep(self):
    return self.len_episode
  
  def set_len_ep(self, len_ep):
    assert len_ep > 0, "Episode len > 0!"
    self.len_episode = len_ep
    # self.printwithName(f"Ep. len updated to {self.len_episode } [steps].")
    self.reset()

  def step(self, f_action_n):

    """
      Take normalized actions and update the environments
    """

    ##
    ## Check for normalization
    ##

    if(self.am.isNanAction(f_action_n) or self.am.actionIsNormalized(f_action_n)==False):
      print(f"Nan action! Previous dist to goal: {self.prev_dist_current2goal}")
      return self.om.getNanObservation(), 0.0, True, {} #This line is added to make generate_trajectories() of rollout.py work when the expert fails 

    f_action_n=f_action_n.reshape(self.action_shape)

    ##
    ## USE CLOSED FORM FOR YAW
    ##

    if(self.par.use_closed_form_yaw_student==True):
      f_action_n=self.cfys.substituteWithClosedFormYaw(f_action_n, self.w_state, self.w_obstacles) #f_action_n, w_init_state, w_obstacles

    ##
    ## Yaw scaling correctoin
    ## to have bigger loss, we magnified yaw actions (look at bc.py's expert_yaw_i assignment)
    ## before convert actions to B-spline, we need to revert this change
    ##
    
    f_action_n[:,self.am.traj_size_pos_ctrl_pts:self.am.traj_size_pos_ctrl_pts+self.am.traj_size_yaw_ctrl_pts] = f_action_n[:,self.am.traj_size_pos_ctrl_pts:self.am.traj_size_pos_ctrl_pts+self.am.traj_size_yaw_ctrl_pts]/self.par.yaw_scaling

    ##
    ## Check for normalization
    ##

    f_action = self.am.denormalizeAction(f_action_n)

    ##
    ##  if total time that student produced is 0.0, BS function gives you an error so let's just make it bigger than 0
    ##

    # for i, (f_act, f_act_n) in enumerate(zip(f_action, f_action_n)):
    #   if f_act[-1] <= 0.0:
    #     print(f"total time {f_act[-1]}, so increase it up to 1e-5")
    #     f_act[-1] = 1e-10

    ##
    ## Choose the trajectory with smallest cost:
    ##

    index_smallest_augmented_cost=self.cost_computer.getIndexBestTraj(self.previous_f_obs_n, f_action_n)

    # self.printwithNameAndColor(f"Choosing traj_{index_smallest_augmented_cost}")
    f_traj=self.am.getTrajFromAction(f_action, index_smallest_augmented_cost)
    f_traj_n=self.am.getTrajFromAction(f_action_n, index_smallest_augmented_cost)
    w_posBS, w_yawBS= self.am.f_trajAnd_w_State2wBS(f_traj, self.w_state)

    ###
    ### MOVE THE ENVIRONMENT
    ###

    ##
    ## Update state
    ##

    self.w_state = State(w_posBS.getPosT(self.training_dt), w_posBS.getVelT(self.training_dt), w_posBS.getAccelT(self.training_dt), \
                        w_yawBS.getPosT(self.training_dt), w_yawBS.getVelT(self.training_dt))

    ##
    ## Update time and timestep
    ##

    self.time = self.time + self.training_dt
    self.timestep = self.timestep + 1

    ##
    ## Construct Obstacles
    ##

    # static or dynamic obstacles
    if self.par.use_dynamic_obst_in_training:
      self.w_obstacles=self.obsm.getFutureWPosDynamicObstacles(self.time)
    else:
      self.w_obstacles=self.obsm.getFutureWPosStaticObstacles()

    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), self.w_obstacles)
    f_observation_n=self.om.normalizeObservation(f_observation)

    ##
    ## Calculate distance
    ##

    dist_current_2gterm=np.linalg.norm(self.w_state.w_pos-self.gm.get_w_GTermPos()) #From the current position to the goal
    dist_endtraj_2gterm=np.linalg.norm(w_posBS.getLastPos()-self.gm.get_w_GTermPos()) #From the end of the current traj to the goal
    self.prev_dist_current2goal = dist_current_2gterm
    # self.printwithName(f"Timestep={self.timestep}, dist_current_2gterm={round(dist_current_2gterm,2)}, w_state.w_pos={self.w_state.w_pos.T}")

    goal_reached = (dist_current_2gterm<self.par.goal_seen_radius and dist_endtraj_2gterm<self.par.goal_radius)
    if(goal_reached):
      self.num_goal_reached += 1
      print(Style.BRIGHT + Fore.GREEN + "Goal reached!" + Style.RESET_ALL)
    self.printwithNameAndColor(f"Timestep={self.timestep}, dist2gterm={round(dist_current_2gterm,2)}, total # of goal reached: {self.num_goal_reached}")

    ##
    ## clip the observation
    ##

    if self.par.use_clipping:
      f_observation_n = np.clip(f_observation_n, -1, 1)

    ##
    ## Check for violation and terminate condition
    ##
    
    info = {}
    is_normalized, which_dyn_limit_violated = self.om.obsIsNormalizedWithVerbose(f_observation_n)
    if is_normalized == False:
      self.printwithName(Style.BRIGHT+Fore.RED +"f_observation_n is not normalized (i.e., constraints are not satisfied). Terminating" + Style.RESET_ALL)
      done = True
      info["obs_constraint_violation"] = True
    elif ( (self.timestep >= self.len_episode) or goal_reached or self.force_done):
      done = True
      info["obs_constraint_violation"] = False
      self.printwithNameAndColor(f"Done, self.timestep={self.timestep}, goal_reached={goal_reached}, force_done={self.force_done}")
    else:
      done=False

    ##
    ## Reward
    ##

    cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = self.cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(self.previous_f_obs_n, f_traj_n)

    info["obst_avoidance_violation"] = True if obst_avoidance_violation > 1e-3 else False
    info["trans_dyn_lim_violation"] = True if any(f in which_dyn_limit_violated for f in ('f_v', 'f_a')) else False
    info["yaw_dyn_lim_violation"] = True if 'yaw_dot' in which_dyn_limit_violated else False

    reward=-augmented_cost

    self.previous_f_observation=f_observation
    self.previous_f_obs_n=f_observation_n

    return f_observation_n, reward, done, info

  def reset(self):

    self.printwithNameAndColor("Resetting environment")
    
    ##
    ## Agent state  
    ##

    self.time=0.0
    self.timestep = 0
    self.force_done=False

    p0=np.array([[0.0],[0.0],[1.0]])
    v0=np.array([[0.0],[0.0],[0.0]])
    a0=np.array([[0.0],[0.0],[0.0]])
    y0=np.array([[0.0]])
    ydot0=np.array([[0.0]])
    self.w_state=State(p0, v0, a0, y0, ydot0)

    ##
    ## Obstacles state
    ##

    if(isinstance(self.constant_obstacle_pos, type(None)) and isinstance(self.constant_gterm_pos, type(None))):
      if np.random.uniform(0, 1) < 1 - self.par.prob_choose_cross:
        self.gm.newRandomPosFarFrom_w_Position(self.w_state.w_pos)
        # self.gm.newRandomPos()
      else:
        w_pos_obstacle, w_pos_g_term = self.obsm.getObsAndGtermToCrossPath()
        self.obsm.setPos(w_pos_obstacle)
        self.gm.setPos(w_pos_g_term)  
    else:
      self.obsm.setPos(self.constant_obstacle_pos)
      self.gm.setPos(self.constant_gterm_pos)

    if self.par.use_dynamic_obst_in_training:
      self.w_obstacles=self.obsm.getFutureWPosDynamicObstacles(self.time)
    else:
      self.w_obstacles=self.obsm.getFutureWPosStaticObstacles()

    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), self.w_obstacles)
    f_observation_n=self.om.normalizeObservation(f_observation)
    self.previous_f_observation=self.om.denormalizeObservation(f_observation_n)
    self.previous_f_obs_n=f_observation_n
    return f_observation_n

  def render(self, mode='human'):
    raise NotImplementedError()
    return
  
  def close (self):
    raise NotImplementedError()
    return
  
  def forceDone(self):
    self.force_done=True

  def get_best_action(self, f_obs, f_acts): #not used
    """ 
    get the best trajectory index from f_obs and f_acts
    this is called for benchmarking
    """
    f_obs_n = self.om.normalizeObservation(f_obs)
    f_acts_n = self.am.normalizeAction(f_acts)

    print("f_obs_n: ", f_obs_n.shape)
    print("f_acts_n: ", f_acts_n.shape)

    return f_acts[self.cost_computer.getIndexBestTraj(f_obs_n, f_acts_n),:]

  def saveInBag(self, f_action_n):
      if(self.record_bag==False or np.isnan(f_action_n).any()):
        return

      ##
      ## When using multiple environments, opening bag in constructor and closing it in _del leads to unindexed bags
      ##

      if(exists(self.name_bag)):
          option='a'
      else:
          option='w'
      self.bag=rosbag.Bag(self.name_bag, option)

      f_obs=self.previous_f_observation
      w_state = self.w_state

      f_action= self.am.denormalizeAction(f_action_n)

      w_posBS_list=[]
      w_yawBS_list=[]

      for i in range( np.shape(f_action)[0] ): #For each row of action   
         f_traj = self.am.getTrajFromAction(f_action, i)
         w_posBS, w_yawBS = self.am.f_trajAnd_w_State2wBS(f_traj, w_state)
         w_posBS_list.append(w_posBS)
         w_yawBS_list.append(w_yawBS)

      time_now=rospy.Time.from_sec(self.time_rosbag)#rospy.Time.from_sec(time.time());
      self.time_rosbag=self.time_rosbag+0.1

      # with rosbag.Bag(name_file, option) as bag:
      f_v=self.om.getf_v(f_obs)
      f_a=self.om.getf_a(f_obs)
      yaw_dot=self.om.getyaw_dot(f_obs)

      ##
      ## get g_term
      ##

      f_g=self.om.getf_g(f_obs)
      point_msg=PointStamped()

      point_msg.header.frame_id = "world"
      point_msg.header.stamp = time_now
      w_g = w_state.w_T_f * f_g
      point_msg.point.x=w_g[0,0]
      point_msg.point.y=w_g[1,0]
      point_msg.point.z=w_g[2,0]
      self.bag.write('/g', point_msg, time_now)

      ##
      ## get obstacles
      ##

      obstacles=self.om.getObstacles(f_obs)
      for i in range(len(obstacles)):

        marker_array_msg=MarkerArray()
        t0=self.time
        tf=self.time + np.max(f_action[:,-1]) #self.am.getTotalTime()#self.par.fitter_total_time

        bspline_obs_i=MyClampedUniformBSpline(t0=t0, tf=tf, deg=self.par.deg_pos, \
                                        dim=3, num_seg=self.par.num_seg, ctrl_pts=listOf3dVectors2numpy3Xmatrix(obstacles[i].ctrl_pts) )

        id_sample=0
        num_samples=20
        for t_interm in np.linspace(t0, tf, num=num_samples).tolist():

          marker_msg=Marker()
          marker_msg.header.frame_id = "world"
          marker_msg.header.stamp = time_now
          marker_msg.ns = "ns"
          marker_msg.id = id_sample
          marker_msg.type = marker_msg.CUBE
          marker_msg.action = marker_msg.ADD
          pos=bspline_obs_i.getPosT(t_interm)
          w_pos_obst = w_state.w_T_f * pos
          marker_msg.pose.position.x = w_pos_obst[0]
          marker_msg.pose.position.y = w_pos_obst[1]
          marker_msg.pose.position.z = w_pos_obst[2]
          # marker_msg.pose.position.x = pos[0]
          # marker_msg.pose.position.y = pos[1]
          # marker_msg.pose.position.z = pos[2]
          marker_msg.pose.orientation.x = 0.0
          marker_msg.pose.orientation.y = 0.0
          marker_msg.pose.orientation.z = 0.0
          marker_msg.pose.orientation.w = 1.0
          marker_msg.scale.x = obstacles[i].bbox_inflated[0]
          marker_msg.scale.y = obstacles[i].bbox_inflated[1]
          marker_msg.scale.z = obstacles[i].bbox_inflated[2]
          # marker_msg.color.a = 1.0*(num_samples-id_sample)/num_samples
          marker_msg.color.a = 1.0
          marker_msg.color.r = 0.0
          marker_msg.color.g = 1.0
          marker_msg.color.b = 0.0

          marker_array_msg.markers.append(marker_msg)

          id_sample=id_sample+1

        self.bag.write(f'/obs{i}', marker_array_msg, time_now)

      ##
      ## tf
      ##

      tf_stamped=TransformStamped()
      tf_stamped.header.frame_id="world"
      tf_stamped.child_frame_id="f"
      tf_stamped.header.stamp=time_now
      rotation_ros, translation_ros=TfMatrix2RosQuatAndVector3(w_state.w_T_f)
      tf_stamped.transform.translation=translation_ros
      tf_stamped.transform.rotation=rotation_ros
      tf_msg=TFMessage()
      tf_msg.transforms.append(tf_stamped)
      self.bag.write('/tf', tf_msg, time_now)

      ##
      ## get trajectory
      ##

      for i in range(len(w_posBS_list)):

        w_posBS=w_posBS_list[i]
        w_yawBS=w_yawBS_list[i]

        pos_t0=w_posBS.getT0()
        pos_tf=w_posBS.getTf()
        yaw_t0=w_yawBS.getT0()
        yaw_tf=w_yawBS.getTf()

        t0 = max(pos_t0, yaw_t0)
        tf = min(pos_tf, yaw_tf)

        traj_msg=Path()
        traj_msg.header.frame_id="world"
        traj_msg.header.stamp=time_now

        for t_i in np.arange(t0, tf, (tf-t0)/10.0):
          pose_stamped=PoseStamped()
          pose_stamped.header.frame_id="world"
          pose_stamped.header.stamp=time_now

          tf_matrix=posAccelYaw2TfMatrix(w_posBS.getPosT(t_i),w_posBS.getAccelT(t_i),w_yawBS.getPosT(t_i))

          pose_stamped.pose=TfMatrix2RosPose(tf_matrix)

          traj_msg.poses.append(pose_stamped)

        self.bag.write('/path'+str(i), traj_msg, time_now)


      #When using multiple environments, opening bag in constructor and closing it in _del leads to unindexed bags
      self.bag.close()
