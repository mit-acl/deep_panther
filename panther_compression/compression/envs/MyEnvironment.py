import gym
import sys
import numpy as np
import copy
from gym import spaces
from compression.utils.other import ActionManager, ObservationManager, GTermManager, State, ObstaclesManager, getPANTHERparamsAsCppStruct, computeTotalTime
from colorama import init, Fore, Back, Style
import py_panther

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

    self.reset()

  def __del__(self):
    # self.eng.quit()
    pass

  def changeConstantObstacleAndGtermPos(self, obstacle_pos, gterm_pos):
    self.constant_obstacle_pos=obstacle_pos
    self.constant_gterm_pos=gterm_pos

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

    # self.printwithName(f"Received actionN={f_action_normalized}")
    f_action= self.am.denormalizeAction(f_action_normalized);
    # self.printwithName(f"Received action size={action.shape}")

    # self.printwithName(f"Timestep={self.timestep}")
    # self.printwithName(f"w_state.w_pos={self.w_state.w_pos.T}")

    ####################################


    w_posBS, w_yawBS= self.am.f_actionAnd_w_State2wBS(f_action, self.w_state)

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



    #Update state
    self.w_state= State(w_posBS.getPosT(actual_dt), w_posBS.getVelT(actual_dt), w_posBS.getAccelT(actual_dt), \
                        w_yawBS.getPosT(actual_dt), w_yawBS.getVelT(actual_dt));


    #Update time and timestep
    self.time = self.time + actual_dt;
    self.timestep = self.timestep + 1

    ####################################
    ####### CONSTRUCT OBS        #######
    ####################################

    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);


    dist_current_2goal=np.linalg.norm(self.w_state.w_pos-self.gm.get_w_GTermPos()) #From the current position to the goal
    dist_endtraj_2goal=np.linalg.norm(w_posBS.getLastPos()-self.gm.get_w_GTermPos()) #From the end of the current traj to the goal


    goal_reached=(dist_current_2goal<4.0 and dist_endtraj_2goal<0.5) 

    self.printwithName(f"Timestep={self.timestep}, dist_current_2goal={dist_current_2goal}, w_state.w_pos={self.w_state.w_pos.T}")

    if(goal_reached):
      self.printwithNameAndColor("Goal reached!")

    
    info = {}
    if(self.om.obsIsNormalized(f_observationn)==False):
      # self.printwithName(f"f_observationn={f_observationn} is not normalized (i.e., constraints are not satisfied). Terminating")
      # self.printwithName(f"f_observation={f_observation} is not normalized (i.e., constraints are not satisfied). Terminating")
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
    print("In reset base")
    self.printwithNameAndColor("Resetting environment")

    self.time=0.0
    self.timestep = 0
    self.w_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))

    if(isinstance(self.constant_obstacle_pos, type(None))):
      self.obsm.newRandomPos();
    else:
      self.obsm.setPos(self.constant_obstacle_pos)

    if(isinstance(self.constant_gterm_pos, type(None))):
      self.gm.newRandomPosFarFrom_w_Position(self.w_state.w_pos);
    else:
      self.gm.setPos(self.constant_gterm_pos);

    
    # observation = self.om.getRandomNormalizedObservation()
    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)
    f_observation=self.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(self.w_state, self.gm.get_w_GTermPos(), w_obstacles);
    f_observationn=self.om.normalizeObservation(f_observation);


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