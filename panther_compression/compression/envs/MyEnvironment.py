import gym
import sys
import numpy as np
import copy
from gym import spaces
from compression.utils.other import ActionManager, ObservationManager, State, ObstaclesManager, isNormalized
from colorama import init, Fore, Back, Style

class MyEnvironment(gym.Env):
  """
    Custom Environment that follows gym interface
    Provides a nonlinear quadcopter gym environment by
    wrapping the mav_sim simulator. 
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(MyEnvironment, self).__init__()

    self.verbose = False

    # Load quadcopter simulator.
    # self.mpc_state_size = 8
    # self.mpc_act_size = 3
    # self.act_size = self.mpc_act_size

    self.len_episode = 200     # Time steps [-]

    self.am=ActionManager();
    self.om=ObservationManager();
    self.obsm=ObstaclesManager();

    self.action_shape=self.am.getActionShape();
    self.observation_shape=self.om.getObservationShape();

    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)
    self.w_gterm_pos=np.array([[8], [0.0], [0.0]])

    self.dt=0.5; #Timestep in seconds
    self.time=0.0;
    
    self.name=Style.BRIGHT+Fore.GREEN+"[Env]"+Style.RESET_ALL
    # print (self.params)

    # print("self.am.getActionShape()= ", self.am.getActionShape())


    self.reset()

  def __del__(self):
    # self.eng.quit()
    pass

  def printwithName(self,data):
    print(self.name+data)

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


    # print("w_posBS.getAccelT(self.dt)= ", w_posBS.getAccelT(self.dt).T)

    #Update state
    self.w_state= State(w_posBS.getPosT(self.dt), w_posBS.getVelT(self.dt), w_posBS.getAccelT(self.dt), \
                        w_yawBS.getPosT(self.dt), w_yawBS.getVelT(self.dt));


    # print("w_yawBS.getPosT(self.dt)= ", w_yawBS.getPosT(self.dt))

    #Update time
    self.time = self.time + self.dt;


    ##### Construct observation
    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)

    # self.printwithName(f"w_obstacles={w_obstacles[0].ctrl_pts}")

    f_observationn=self.om.getNormalized_fObservationFromTime_w_stateAnd_w_gtermAnd_w_obstacles(self.time, self.w_state, self.w_gterm_pos, w_obstacles);


    dist2goal=np.linalg.norm(self.w_state.w_pos-self.w_gterm_pos)
    # dist2goal=np.linalg.norm(w_posBS.getLastPos()-self.w_gterm_pos) #From the end of the current traj to the goal
    self.printwithName(f"Timestep={self.timestep}, dist2goal={dist2goal}, w_state.w_pos={self.w_state.w_pos.T}")


    self.timestep = self.timestep + 1
    info = {}
    if(isNormalized(f_observationn)==False):
      self.printwithName(f"f_observationn={f_observationn} is not normalized (i.e., constraints are not satisfied). Terminating")
      # print(f"[Env] Terminated due to constraint violation: obs: {self.x}, act: {u}, steps: {self.timestep}")
      done = True
      info["constraint_violation"] = True
    elif ( (self.timestep >= self.len_episode) or (dist2goal<0.5) ):
      done = True
      info["constraint_violation"] = False
    else:
      done=False


    reward=0.0

    # self.printwithName(f"returning reward={reward}")
    # self.printwithName(f"returning obsN={observation}")
    # self.printwithName(f"returning obs size={observation.shape}")
    return f_observationn, reward, done, info

  def reset(self):
    self.printwithName("Resetting environment")

    self.time=0.0
    self.timestep = 0
    self.w_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))
    self.obsm.newRandomPos();
    # observation = self.om.getRandomNormalizedObservation()
    w_obstacles=self.obsm.getFutureWPosObstacles(self.time)
    f_observationn=self.om.getNormalized_fObservationFromTime_w_stateAnd_w_gtermAnd_w_obstacles(self.time, self.w_state, self.w_gterm_pos, w_obstacles);
    
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