import gym
import sys
import numpy as np
import copy
from gym import spaces
from compression.utils.other import ActionManager, ObservationManager, State, ObstaclesManager
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

    self.len_episode = 200     # Time steps [-] # TODO: Andrea: load from params or set from outside. 

    self.am=ActionManager();
    self.om=ObservationManager();
    self.obsm=ObstaclesManager();

    self.action_shape=self.am.getActionShape();
    self.observation_shape=self.om.getObservationShape();

    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=self.action_shape)
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=self.observation_shape)
    self.w_goal=np.array([[10], [0.0], [0.0]])
    # self.max_act = 12.0 # Todo: make sure this is the same value as used in the MPC
    # self.max_obs = 10.0 # Todo: make sure this is the same as in the MPC

    self.dt=0.2; #Timestep in seconds
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

  def step(self, action):
    action=action.reshape(self.action_shape) 
    assert not np.isnan(np.sum(action)), "Received invalid command! u contains nan"

    self.printwithName(f"Received actionN={action}")

    action= self.am.denormalizeAction(action);

    # print ("action.shape= ", action.shape)
    # print ("self.action_shape= ",self.action_shape)

    # self.printwithName(f"Received action size={action.shape}")

    assert action.shape==self.action_shape, f"[Env] ERROR: action.shape={action.shape} but self.action_shape={self.action_shape}"

      
    # print(f"[Env] Timestep={self.timestep}")

    ####################################

    w_posBS, w_yawBS= self.am.action2wBS(action, self.w_state)


    #Update state
    self.w_state= State(w_posBS.getPosT(self.dt), w_posBS.getVelT(self.dt), w_posBS.getAccelT(self.dt), \
                        w_yawBS.getPosT(self.dt), w_yawBS.getVelT(self.dt));

    #Update time
    self.time = self.time + self.dt;

    #Construct observation
    w_obs=self.obsm.getFutureWPosObstacles(self.time)
    f_obs=self.om.construct_f_obsFrom_w_state_and_w_obs(self.w_state, w_obs, self.w_goal)


    ####################################

    reward=0.0

    info = {}
    self.timestep = self.timestep + 1
    if self.timestep >= self.len_episode:
      done = True
      info["constraint_violations"] = False
    # elif violation_bfr or violation_after: #or u_violation
    #   done = True
    #   info["constraint_violation"] = True
      #print(f"[Env] Terminated due to constraint violation: obs: {self.x}, act: {u}, steps: {self.timestep}")
    else:
      done = False
    
    observation = self.om.getRandomNormalizedObservation()#np.random.rand(1,self.mpc_state_size)

    # normalized_obs = observation.reshape(-1,)/self.max_obs
    self.printwithName(f"returning obsN={observation}")
    # self.printwithName(f"returning obs size={observation.shape}")
    return observation, reward, done, info

  def reset(self):
    self.time=0.0
    self.timestep = 0
    self.w_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))
    observation = self.om.getRandomNormalizedObservation()
    
    assert observation.shape == self.observation_shape
    # self.printwithName(f"returning obs={observation}")
    return observation

  # def get_random_init_state(self):
  #   # TODO: pass initial set and randomly sample from there.
  #   # TODO: load initial state from params. 
  #   pos = np.zeros(3)
  #   pos[0:2] = np.random.uniform(0.1, -0.1, 2)
  #   pos[2] = np.random.uniform(-0.1, 0.1) #np.random.uniform(0.9, 1.1)
  #   vel = np.random.uniform(-0.1, 0.1, 3)
  #   rp = np.random.uniform(-0.1, 0.1, 2)
  #   return np.concatenate((pos, vel, rp), axis=None).reshape(self.mpc_state_size,)
  
  def render(self, mode='human'):
    raise NotImplementedError()
    return
  
  def close (self):
    raise NotImplementedError()
    return
