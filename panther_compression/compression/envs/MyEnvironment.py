import gym
import sys
import numpy as np
import copy
from gym import spaces
from compression.utils.other import ActionManager, ObservationManager, GTermManager, State, ObstaclesManager, getPANTHERparamsAsCppStruct, computeTotalTime, getObsAndGtermToCrossPath, posAccelYaw2TfMatrix
from colorama import init, Fore, Back, Style
import py_panther
##### For rosbag logging
import rosbag
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
import time
import rospy
from os.path import exists
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
import tf.transformations as tr
#############################################

class MyEnvironment(gym.Env):
  """
    Custom Environment that follows gym interface
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__() #Equivalently, we could also write super(MyEnvironment, self).__init__()    , see 2nd paragraph of https://stackoverflow.com/a/576183

    self.verbose = False

    # Load quadcopter simulator.
    # self.mpc_state_size = 8
    # self.mpc_act_size = 3
    # self.act_size = self.mpc_act_size

    self.len_episode = 200     # Time steps [-]

    self.am=ActionManager();
    self.om=ObservationManager();
    self.obsm=ObstaclesManager();
    self.gm=GTermManager();

    self.action_shape=self.am.getActionShape();
    self.observation_shape=self.om.getObservationShape();

    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)

    self.dt=0.5; #Timestep in seconds
    self.time=0.0;

    self.color=Style.BRIGHT+Fore.YELLOW
    
    self.name=self.color+"  [Env]"+Style.RESET_ALL
    # print (self.params)

    # print("self.am.getActionShape()= ", self.am.getActionShape())


    ######
    self.par=getPANTHERparamsAsCppStruct();
    self.my_SolverIpopt=py_panther.SolverIpopt(self.par);
    #######
    print("Calling reset")

    self.constant_obstacle_pos=None
    self.constant_gterm_pos=None

    self.record_bag=False

    self.reset()

  def __del__(self):
    if(self.record_bag==True):
      self.bag.close();
    # pass

  def changeConstantObstacleAndGtermPos(self, obstacle_pos, gterm_pos):
    self.constant_obstacle_pos=obstacle_pos
    self.constant_gterm_pos=gterm_pos

  def startRecordBag(self, name_bag):
    self.record_bag=True

    if(exists(name_bag)):
        option='a'
    else:
        option='w'
    print("option= ", option)
    self.bag=rosbag.Bag(name_bag, option)

  def printwithName(self,data):
    print(self.name+data)

  def printwithNameAndColor(self,data):
    print(self.name+self.color+data+Style.RESET_ALL)


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

  def step(self, f_action_normalized):
    f_action_normalized=f_action_normalized.reshape(self.action_shape) 

    self.am.assertAction(f_action_normalized)
    self.am.assertActionIsNormalized(f_action_normalized)

    # self.printwithName(f"Received actionN={f_action_normalized}")
    f_action= self.am.denormalizeAction(f_action_normalized);
    # self.printwithName(f"Received action size={action.shape}")

    # self.printwithName(f"Timestep={self.timestep}")
    # self.printwithName(f"w_state.w_pos={self.w_state.w_pos.T}")

    # self.am.printAction(f_action)
    ####################################


    w_posBS, w_yawBS= self.am.f_actionAnd_w_State2wBS(f_action, self.w_state)

    ######################################
    if(self.record_bag==True):
      self.saveInBag(self.previous_f_observation, w_posBS, w_yawBS, self.w_state)
    ######################################

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
                        w_yawBS.getPosT(actual_dt), w_yawBS.getVelT(actual_dt));

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

    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);


    dist_current_2gterm=np.linalg.norm(self.w_state.w_pos-self.gm.get_w_GTermPos()) #From the current position to the goal
    dist_endtraj_2gterm=np.linalg.norm(w_posBS.getLastPos()-self.gm.get_w_GTermPos()) #From the end of the current traj to the goal


    goal_reached=(dist_current_2gterm<4.0 and dist_endtraj_2gterm<self.par.goal_radius) 

    self.printwithName(f"Timestep={self.timestep}, dist_current_2gterm={dist_current_2gterm}, w_state.w_pos={self.w_state.w_pos.T}")

    if(goal_reached):
      self.printwithNameAndColor("Goal reached!")

    
    info = {}
    if(self.om.obsIsNormalized(f_observationn)==False):
      # self.printwithName(f"f_observationn={f_observationn} is not normalized (i.e., constraints are not satisfied). Terminating")
      # self.printwithName(f"f_observation={f_observation} is not normalized (i.e., constraints are not satisfied). Terminating")
      # self.om.printObservation(f_observation)
      # exit();
      # self.printwithName(f"f_observationn is not normalized (i.e., constraints are not satisfied). Terminating")
      # print(f"[Env] Terminated due to constraint violation: obs: {self.x}, act: {u}, steps: {self.timestep}")
      done = True
      info["constraint_violation"] = True
    elif ( (self.timestep >= self.len_episode) or goal_reached ):
      done = True
      info["constraint_violation"] = False
    else:
      done=False


    #####################
    init_state=self.om.getInit_f_StateFromObservation(self.previous_f_observation)
    final_state=self.om.getFinal_f_StateFromObservation(self.previous_f_observation)
    total_time=computeTotalTime(init_state, final_state, self.par.v_max, self.par.a_max, self.par.factor_alloc)
    self.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
    self.my_SolverIpopt.setFocusOnObstacle(True);
    self.my_SolverIpopt.setObstaclesForOpt(self.om.getObstacles(self.previous_f_observation));
    cost=self.my_SolverIpopt.computeCost(self.am.f_action2f_ppSolOrGuess(f_action))
    reward=-cost;
    # print(reward)
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

    return f_observationn, reward, done, info

  def reset(self):
    self.printwithNameAndColor("Resetting environment")

    self.time=0.0
    self.timestep = 0

    p0=np.array([[0.0],[0.0],[1.0]])
    v0=np.array([[0.0],[0.0],[0.0]])
    a0=np.array([[0.0],[0.0],[0.0]])
    y0=np.array([[0.0]])
    ydot0=np.array([[0.0]])
    self.w_state=State(p0, v0, a0, y0, ydot0)

    if(isinstance(self.constant_obstacle_pos, type(None)) and isinstance(self.constant_gterm_pos, type(None))):

      if np.random.uniform(0, 1) > 0.5:
        self.obsm.newRandomPos();
        self.gm.newRandomPosFarFrom_w_Position(self.w_state.w_pos);
      else:
        w_pos_obstacle, w_pos_g_term = getObsAndGtermToCrossPath();
        self.obsm.setPos(w_pos_obstacle)
        self.gm.setPos(w_pos_g_term);        
        self.printwithNameAndColor("Using cross!")

    else:
      self.obsm.setPos(self.constant_obstacle_pos)
      self.gm.setPos(self.constant_gterm_pos);


    
    # observation = self.om.getRandomNormalizedObservation()
    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)
    # print("w_obstacles[0].ctrl_pts=", w_obstacles[0].ctrl_pts)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);

    # self.printwithName("THIS IS THE OBSERVATION:")
    # self.om.printObservation(f_observation)

    self.previous_f_observation=self.om.denormalizeObservation(f_observationn)
    # assert observation.shape == self.observation_shape
    # self.printwithName(f"returning obs={observation}")
    return f_observationn

 
  def render(self, mode='human'):
    raise NotImplementedError()
    return
  
  def close (self):
    raise NotImplementedError()
    return

  def TfMatrix2RosQuatAndVector3(self, tf_matrix):

      translation_ros=Vector3();
      rotation_ros=Quaternion();

      translation=tf_matrix.translation();
      translation_ros.x=translation[0];
      translation_ros.y=translation[1];
      translation_ros.z=translation[2];
      # q=tr.quaternion_from_matrix(w_state.w_T_f.T)
      quaternion=tr.quaternion_from_matrix(tf_matrix.T) #See https://github.com/ros/geometry/issues/64
      rotation_ros.x=quaternion[0] #See order at http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
      rotation_ros.y=quaternion[1]
      rotation_ros.z=quaternion[2]
      rotation_ros.w=quaternion[3]

      return rotation_ros, translation_ros


  def TfMatrix2RosPose(self, tf_matrix):

      rotation_ros, translation_ros=self.TfMatrix2RosQuatAndVector3(tf_matrix);

      pose_ros=Pose();
      pose_ros.position.x=translation_ros.x
      pose_ros.position.y=translation_ros.y
      pose_ros.position.z=translation_ros.z

      pose_ros.orientation=rotation_ros

      return pose_ros


  def saveInBag(self, f_obs, w_posBS, w_yawBS, w_state):

      time_now=rospy.Time.from_sec(time.time());

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

      marker_msg=Marker();
      marker_msg.header.frame_id = "f";
      marker_msg.header.stamp = time_now;
      marker_msg.ns = "ns";
      marker_msg.id = 0;
      marker_msg.type = marker_msg.CUBE;
      marker_msg.action = marker_msg.ADD;
      marker_msg.pose.position.x = obstacles[0].ctrl_pts[0][0];
      marker_msg.pose.position.y = obstacles[0].ctrl_pts[0][1];
      marker_msg.pose.position.z = obstacles[0].ctrl_pts[0][2];
      marker_msg.pose.orientation.x = 0.0;
      marker_msg.pose.orientation.y = 0.0;
      marker_msg.pose.orientation.z = 0.0;
      marker_msg.pose.orientation.w = 1.0;
      marker_msg.scale.x = obstacles[0].bbox_inflated[0];
      marker_msg.scale.y = obstacles[0].bbox_inflated[1];
      marker_msg.scale.z = obstacles[0].bbox_inflated[2];
      marker_msg.color.a = 1.0; 
      marker_msg.color.r = 0.0;
      marker_msg.color.g = 1.0;
      marker_msg.color.b = 0.0;

      obstacles[0].ctrl_pts[0][1]


      tf_stamped=TransformStamped();
      tf_stamped.header.frame_id="world"
      tf_stamped.header.stamp=time_now
      tf_stamped.child_frame_id="f"
      rotation_ros, translation_ros=self.TfMatrix2RosQuatAndVector3(w_state.w_T_f)
      tf_stamped.transform.translation=translation_ros
      tf_stamped.transform.rotation=rotation_ros

      tf_msg=TFMessage()
      tf_msg.transforms.append(tf_stamped)


      traj_msg=Path()
      traj_msg.header.frame_id="world"
      traj_msg.header.stamp=time_now

      t0=w_posBS.getT0();
      tf=w_posBS.getTf();

      for t_i in np.arange(t0, tf, (tf-t0)/10.0):
        pose_stamped=PoseStamped();
        pose_stamped.header.frame_id="world"
        pose_stamped.header.stamp=time_now
        tf_matrix=posAccelYaw2TfMatrix(w_posBS.getPosT(t_i),w_posBS.getAccelT(t_i),w_yawBS.getPosT(t_i))

        pose_stamped.pose=self.TfMatrix2RosPose(tf_matrix)

        traj_msg.poses.append(pose_stamped)



      self.bag.write('/path', traj_msg, time_now)
      self.bag.write('/g', point_msg, time_now)
      self.bag.write('/obs', marker_msg, time_now)
      self.bag.write('/tf', tf_msg, time_now)




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