import gymnasium as gym
import sys
import numpy as np
import copy

from gymnasium import spaces

from compression.utils.utils import GTermManager, computeTotalTime, getObsAndGtermToCrossPath, posAccelYaw2TfMatrix, listOf3dVectors2numpy3Xmatrix
from compression.utils.State import State
from compression.utils.yaml_utils import getPANTHERparamsAsCppStruct
from compression.utils.ActionManager import ActionManager
from compression.utils.ObservationManager import ObservationManager
from compression.utils.ObstaclesManager import ObstaclesManager
from compression.utils.ros_utils import TfMatrix2RosQuatAndVector3, TfMatrix2RosPose
from compression.utils.CostComputer import CostComputer, ClosedFormYawSubstituter
from compression.utils.MyClampedUniformBSpline import MyClampedUniformBSpline

from colorama import init, Fore, Back, Style
import py_panther

##### For rosbag logging
import rosbag
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
import time
import rospy
from os.path import exists
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PointStamped
#############################################

import uuid


class MyEnvironment(gym.Env):
  """
    Custom Environment that follows gym interface
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, dim=3, num_obs=1):
    super().__init__() #Equivalently, we could also write super(MyEnvironment, self).__init__()    , see 2nd paragraph of https://stackoverflow.com/a/576183

    self.verbose = False

    self.dim = dim
    self.len_episode = 200     # Time steps [-]
    self.am=ActionManager(dim=dim);
    self.om=ObservationManager(dim=dim, num_obs=num_obs);
    self.obsm=ObstaclesManager(dim=dim, num_obs=num_obs);
    self.gm=GTermManager();
    self.cfys=ClosedFormYawSubstituter(dim=dim);


    self.cost_computer=CostComputer(dim=dim, num_obs=num_obs)

    self.action_shape=self.am.getActionShape();
    self.observation_shape=self.om.getObservationShape();

    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)

    self.dt=0.5; #Timestep in seconds
    self.time=0.0;

    self.color=Style.BRIGHT+Fore.YELLOW
    
    self.name=""
    # print (self.params)

    # print("self.am.getActionShape()= ", self.am.getActionShape())


    ######
    self.par=getPANTHERparamsAsCppStruct();
    # self.my_SolverIpopt=py_panther.SolverIpopt(self.par);
    #######

    self.constant_obstacle_pos=None
    self.constant_gterm_pos=None

    self.record_bag=False
    self.time_rosbag=0;
    self.name_bag=None

    self.force_done=False

    self.id=0

    self.seed = None

    self.reset()

  def __del__(self):
    # if(self.record_bag==True):
    #   self.bag.close();
    pass

  def changeConstantObstacleAndGtermPos(self, obstacle_pos, gterm_pos):
    self.constant_obstacle_pos=obstacle_pos
    self.constant_gterm_pos=gterm_pos

  def setID(self, data):
    self.id=data;
    self.name=self.color+"  [Env "+str(self.id)+"]"+Style.RESET_ALL

  def startRecordBag(self):
    self.record_bag=True
    self.name_bag="training"+str(self.id)+".bag";

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
    self.seed = seed
    if seed is not None:
      np.random.seed(seed)
      self.reset(seed=seed)

  def get_len_ep(self):
    return self.len_episode
  
  def set_len_ep(self, len_ep):
    assert len_ep > 0, "Episode len > 0!"
    self.len_episode = len_ep
    # self.printwithName(f"Ep. len updated to {self.len_episode } [steps].")
    if self.seed is not None:
      self.reset(seed=self.seed)

  def step(self, f_action_n):
    # self.printwithName(f"Received actionN={f_action_n}")

    if(self.am.isNanAction(f_action_n)):
      #f_observationn, reward, done, info
      return self.om.getNanObservation(), 0.0, True, {} #This line is added to make generate_trajectories() of rollout.py work when the expert fails 

    f_action_n=f_action_n.reshape(self.action_shape) 

    self.am.assertAction(f_action_n)
    self.am.assertActionIsNormalized(f_action_n)


    #################################
    #### USE CLOSED FORM FOR YAW #####
    if(self.par.use_closed_form_yaw_student==True):
      f_action_n=self.cfys.substituteWithClosedFormYaw(f_action_n, self.w_state, self.w_obstacles) #f_action_n, w_init_state, w_obstacles
    ##################################

    # self.printwithName(f"Received actionN={f_action_n}")
    f_action= self.am.denormalizeAction(f_action_n);
    # self.printwithName(f"Received action size={action.shape}")

    # self.printwithName(f"Timestep={self.timestep}")
    # self.printwithName(f"w_state.w_pos={self.w_state.w_pos.T}")

    # self.am.printAction(f_action)
    ####################################



    #Choose the trajectory with smallest cost:
    index_smallest_augmented_cost=self.cost_computer.getIndexBestTraj(self.previous_f_obs_n, f_action_n)

    self.printwithNameAndColor(f"Choosing traj_{index_smallest_augmented_cost}")
    f_traj=self.am.getTrajFromAction(f_action, index_smallest_augmented_cost)
    f_traj=np.nan_to_num(f_traj)
    f_traj_n=self.am.getTrajFromAction(f_action_n, index_smallest_augmented_cost)
    f_traj_n=np.nan_to_num(f_traj_n)
    w_posBS, w_yawBS= self.am.f_trajAnd_w_State2wBS(f_traj, self.w_state)

    ####################################
    ####### MOVE THE ENVIRONMENT #######
    ####################################


    # if(dist2goal<0.5):
    #   # actual_dt=self.am.getTotalTime(f_action) #Advance to the end of that trajectory
    #   self.gm.newRandomPosFarFrom_w_Position(self.w_state.w_pos);   #DONT DO THIS HERE!(the observation will change with no correlation with the action sent)
    #   self.printwithNameAndColor(f"New goal at {self.gm.get_w_GTermPos()}") 
    # # else:
    # #   actual_dt=self.dt
    actual_dt=self.dt


    # self.printwithName("w and f state BEFORE UPDATE")
    # self.w_state.print_w_frameHorizontal(self.name);
    # # print(f"Matrix w_T_f=\n{self.w_state.w_T_f.T}")
    # self.w_state.print_f_frameHorizontal(self.name);

    #Update state
    self.w_state= State(w_posBS.getPosT(actual_dt), w_posBS.getVelT(actual_dt), w_posBS.getAccelT(actual_dt), \
                        w_yawBS.getPosT(actual_dt), w_yawBS.getVelT(actual_dt), dim=self.dim);

    # self.printwithName("w and f state AFTER UPDATE")
    # self.w_state.print_w_frameHorizontal(self.name);
    # # print(f"Matrix w_T_f=\n{self.w_state.w_T_f.T}")
    # self.w_state.print_f_frameHorizontal(self.name);

    #Update time and timestep
    self.time = self.time + actual_dt;
    self.timestep = self.timestep + 1

    ####################################
    ####### CONSTRUCT OBS        #######
    ####################################

    # w_obstacles=self.obsm.getFutureWPosStaticObstacles()
    self.w_obstacles=self.obsm.getFutureWPosDynamicObstacles(self.time)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), self.w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);

    w_g = self.gm.get_w_GTermPos()[:self.dim]
    dist_current_2gterm=np.linalg.norm(self.w_state.w_pos-w_g) #From the current position to the goal
    dist_endtraj_2gterm=np.linalg.norm(w_posBS.getLastPos()-w_g) #From the end of the current traj to the goal


    goal_reached=(dist_current_2gterm<2.0 and dist_endtraj_2gterm<self.par.goal_radius) 

    self.printwithName(f"Timestep={self.timestep}, dist_current_2gterm={dist_current_2gterm}, w_state.w_pos={self.w_state.w_pos.T}")

    # if(goal_reached):
    #   self.printwithNameAndColor("Goal reached!")

    
    info = {}
    if(self.om.obsIsNormalized(f_observationn)==False):
      # self.printwithName(f"f_observationn={f_observationn} is not normalized (i.e., constraints are not satisfied). Terminating")
      # self.printwithName(f"f_observation={f_observation} is not normalized (i.e., constraints are not satisfied). Terminating")
      # exit();
      self.printwithName(Style.BRIGHT+Fore.RED +"f_observationn is not normalized (i.e., constraints are not satisfied). Terminating" + Style.RESET_ALL)
      # self.om.printObservation(f_observation)
    
      # print(f"[Env] Terminated due to constraint violation: obs: {self.x}, act: {u}, steps: {self.timestep}")
      done = True
      info["constraint_violation"] = True
    elif ( (self.timestep >= self.len_episode) or goal_reached or self.force_done):
      done = True
      info["constraint_violation"] = False
      self.printwithNameAndColor(f"Done, self.timestep={self.timestep}, goal_reached={goal_reached}, force_done={self.force_done}")

    else:
      done=False

    self.printwithNameAndColor(f"done={done}")


    #####################
    # init_state=self.om.getInit_f_StateFromObservation(self.previous_f_observation)
    # final_state=self.om.getFinal_f_StateFromObservation(self.previous_f_observation)
    # total_time=computeTotalTime(init_state, final_state, self.par.v_max, self.par.a_max, self.par.factor_alloc)
    # self.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
    # self.my_SolverIpopt.setFocusOnObstacle(True);
    # self.my_SolverIpopt.setObstaclesForOpt(self.om.getObstacles(self.previous_f_observation));
    # cost=self.my_SolverIpopt.computeCost(self.am.f_traj2f_ppSolOrGuess(f_traj)) #TODO: this cost does not take into accout the constraints right now
    # cost=self.cost_computer.computeCost(self.previous_f_obs_n, f_traj_n)
    # constraints_violation=self.cost_computer.computeConstraintsViolation(self.previous_f_obs_n, f_traj_n)
    # augmented_cost=self.cost_computer.computeAugmentedCost(self.previous_f_obs_n, f_traj_n)
    # self.printwithNameAndColor(f"augmented cost={augmented_cost}")

    # print(f"constraints_violation={constraints_violation}")
    # reward=-augmented_cost;
    reward=0.0;
    # # print(reward)
    ###################

    # self.printwithName("THIS IS THE OBSERVATION:")
    # self.om.printObservation(f_observationn)
    # np.set_printoptions(precision=2)
    # print("w_obstacles[0].ctrl_pts=", w_obstacles[0].ctrl_pts)
    # print("w_obstacles[0].bbox_inflated=", w_obstacles[0].bbox_inflated.T)
    # print("self.gm.get_w_GTermPos()=", self.gm.get_w_GTermPos().T)
    # print("observation=", f_observationn)

    # print("w_yawBS.getPosT(self.dt)= ", w_yawBS.getPosT(self.dt))
        # self.printwithName(f"w_obstacles={w_obstacles[0].ctrl_pts}")
    # print("w_posBS.getAccelT(self.dt)= ", w_posBS.getAccelT(self.dt).T)
    # self.printwithName(f"returning reward={reward}")
    # self.printwithName(f"returning obsN={observation}")
    # self.printwithName(f"returning obs size={observation.shape}")

    self.previous_f_observation=f_observation
    self.previous_f_obs_n=f_observationn

    return f_observationn, reward, done, False, info

  def reset(self, seed = None, options = None):
    self.printwithNameAndColor("Resetting environment")

    super().reset(seed=seed, options=options)

    self.time=0.0
    self.timestep = 0
    self.force_done=False

    p0=np.zeros((2, 1))
    if self.dim == 3:
      p0=np.array([[0.0],[0.0],[1.0]])
    v0=np.zeros((self.dim, 1))
    a0=np.zeros((self.dim, 1))
    y0=np.array([[0.0]])
    ydot0=np.array([[0.0]])
    self.w_state=State(p0, v0, a0, y0, ydot0, dim=self.dim)

    if(isinstance(self.constant_obstacle_pos, type(None)) and isinstance(self.constant_gterm_pos, type(None))):

      self.obsm.newRandomPos();

      prob_choose_cross=1.0;
      if np.random.uniform(0, 1) < 1 - prob_choose_cross:
        self.gm.newRandomPosFarFrom_w_Position(self.w_state.w_pos);
        # self.gm.newRandomPos();
      else:
        w_pos_obstacle, w_pos_g_term = getObsAndGtermToCrossPath();
        self.obsm.setPos(w_pos_obstacle)
        self.gm.setPos(w_pos_g_term);        
        # self.printwithNameAndColor("Using cross!")

    else:
      self.obsm.setPos(self.constant_obstacle_pos)
      self.gm.setPos(self.constant_gterm_pos);


    
    # observation = self.om.getRandomNormalizedObservation()
    # w_obstacles=self.obsm.getFutureWPosStaticObstacles()
    self.w_obstacles=self.obsm.getFutureWPosDynamicObstacles(self.time)
    # print("w_obstacles[0].ctrl_pts=", w_obstacles[0].ctrl_pts)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), self.w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);

    # self.printwithName("THIS IS THE OBSERVATION:")
    # self.om.printObservation(f_observation)

    self.previous_f_observation=self.om.denormalizeObservation(f_observationn)
    self.previous_f_obs_n=f_observationn
    # assert observation.shape == self.observation_shape
    # self.printwithName(f"returning obs={observation}")
    return f_observationn, {}

 
  def render(self, mode='human'):
    raise NotImplementedError()
    return
  
  def close (self):
    raise NotImplementedError()
    return
  def forceDone(self):
    self.force_done=True

  def saveInBag(self, f_action_n):
      if(self.record_bag==False or np.isnan(f_action_n).any()):
        return

      #When using multiple environments, opening bag in constructor and closing it in _del leads to unindexed bags
      if(exists(self.name_bag)):
          option='a'
      else:
          option='w'
      self.bag=rosbag.Bag(self.name_bag, option)
      ######################

      #####
      f_obs=self.previous_f_observation
      w_state=self.w_state
      ########

      f_action= self.am.denormalizeAction(f_action_n);

      w_posBS_list=[]
      w_yawBS_list=[]

      for i in range( np.shape(f_action)[0]): #For each row of action   
         f_traj=self.am.getTrajFromAction(f_action, i)
         w_posBS, w_yawBS= self.am.f_trajAnd_w_State2wBS(f_traj, self.w_state)
         w_posBS_list.append(w_posBS)
         w_yawBS_list.append(w_yawBS)

      time_now=rospy.Time.from_sec(self.time_rosbag)#rospy.Time.from_sec(time.time());
      self.time_rosbag=self.time_rosbag+0.1;

      # with rosbag.Bag(name_file, option) as bag:
      f_v=self.om.getf_v(f_obs)
      f_a=self.om.getf_a(f_obs)
      yaw_dot=self.om.getyaw_dot(f_obs)
      f_g=self.om.getf_g(f_obs)
      obstacles=self.om.getObstacles(f_obs)
      point_msg=PointStamped()

      point_msg.header.frame_id = "f";
      point_msg.header.stamp = time_now;
      point_msg.point.x=f_g[0,0]
      point_msg.point.y=f_g[1,0]
      point_msg.point.z=f_g[2,0]

      marker_array_msg=MarkerArray()

      for i in range(len(obstacles)):

        t0=self.time
        tf=self.time + np.max(f_action[:,-1]) #self.am.getTotalTime()#self.par.fitter_total_time

        bspline_obs_i=MyClampedUniformBSpline(t0=t0, tf=tf, deg=self.par.deg_pos, \
                                        dim=3, num_seg=self.par.num_seg, ctrl_pts=listOf3dVectors2numpy3Xmatrix(obstacles[i].ctrl_pts) )

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
          tf_matrix=posAccelYaw2TfMatrix(w_posBS.getPosT(t_i),w_posBS.getAccelT(t_i),w_yawBS.getPosT(t_i), dim=self.dim)

          pose_stamped.pose=TfMatrix2RosPose(tf_matrix)

          traj_msg.poses.append(pose_stamped)

        self.bag.write('/path'+str(i), traj_msg, time_now)


      self.bag.write('/g', point_msg, time_now)
      self.bag.write('/obs', marker_array_msg, time_now)
      self.bag.write('/tf', tf_msg, time_now)

      #When using multiple environments, opening bag in constructor and closing it in _del leads to unindexed bags
      self.bag.close();
      ########################




    #Debugging
    # print("\n\n\n\n\n")

    # p0=np.array([[0],[0],[0]]);
    # v0=np.array([[0],[0],[0]]);
    # a0=np.array([[2],[0],[0]]);
    # y0=np.array([[0]]);
    # y_dot0=np.array([[0]]);
    # self.w_state= State(p0, v0, a0, y0, y_dot0)
    # w_posBS, w_yawBS= self.am.f_actionAnd_w_State2wBS(f_action, self.w_state)

    # p0=np.array([[0],[0],[0]]);
    # v0=np.array([[0],[0],[0]]);
    # a0=np.array([[0],[0],[0]]);
    # y0=np.array([[0]]);
    # y_dot0=np.array([[0]]);
    # my_state_zero=State(p0, v0, a0, y0, y_dot0)
    # f_posBS, f_yawBS= self.am.f_actionAnd_w_State2wBS(f_action, my_state_zero)

    # f_accelBuena=f_posBS.getAccelT(self.dt);
    # w_posdtBuena=(self.w_state.w_T_f*f_posBS.getPosT(self.dt));
    # # print("[BUENA] f_state.accel= ", f_accelBuena.T)
    # # print("[BUENA] w_state.accel= ", (np.linalg.inv(self.w_state.f_T_w.rot())@f_accelBuena).T)
    # print("[BUENA] f_posBS.pos(self.dt)= ", (f_posBS.getPosT(self.dt)).T)
    # print("[BUENA] w_posdt= ", w_posdtBuena.T)


    # # print("w_state.accel= ",self.w_state.w_accel.T)
    # print("w_T_f=\n", self.w_state.w_T_f.T)

    # # print("f_state.accel2= ", self.w_state.f_accel().T)
    # # print("w_state.accel= ", w_posBS.getAccelT(self.dt).T)
    # w_posdt=w_posBS.getPosT(self.dt);
    # print("w_posdt= ", w_posdt.T)

    # np.set_printoptions(edgeitems=30, linewidth=100000, 
    # formatter=dict(float=lambda x: "%.3g" % x))
    # print("w_posBS.ctrl_pts= \n", w_posBS.ctrl_pts)
    # print("f_posBS.ctrl_pts= \n", f_posBS.ctrl_pts)

    # assert np.linalg.norm(w_posdtBuena-w_posdt)<1e-6 
    #####