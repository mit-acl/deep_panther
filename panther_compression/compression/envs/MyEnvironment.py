import gym
import sys
import numpy as np
import copy
from gym import spaces
# from compression.experts.TubeMpc import LocalFeedbackController
 

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
    self.mpc_state_size = 8
    self.mpc_act_size = 3
    self.act_size = self.mpc_act_size

    self.len_episode = 200     # Time steps [-] # TODO: Andrea: load from params or set from outside. 

    # Define action and observation space
    self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=(self.mpc_act_size,))
    self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape=(self.mpc_state_size,))
    self.max_act = 12.0 # Todo: make sure this is the same value as used in the MPC
    self.max_obs = 10.0 # Todo: make sure this is the same as in the MPC

    self.reset()

  def __del__(self):
    # self.eng.quit()
    pass

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
    print(f"[GymEnv]: Ep. len updated to {self.len_episode } [steps].")
    self.reset()

  def step(self, u):
    assert not np.isnan(np.sum(u)), "Received invalid command! u contains nan"
    u = np.array(u*self.max_act).reshape(self.mpc_act_size,) # make sure is always dim 1
      
    # print(f"[Env] Timestep={self.timestep}")

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
    
    observation = np.random.rand(1,self.mpc_state_size)
  
    normalized_obs = observation.reshape(-1,)/self.max_obs
    assert normalized_obs.shape == (self.mpc_state_size, )
    return normalized_obs, reward, done, info

  def reset(self):
    self.timestep = 0
    self.x = self.get_random_init_state()
    # ref_traj = self.planner.reset()
    mpc_state= self.get_random_init_state();
    #assert np.allclose(mpc_state, self.x), f"Unable to set desired random initial state. Desired: {self.x}, mav_sim: {mpc_state}."
    observation = mpc_state; #np.concatenate((mpc_state), axis=0)
    normalized_obs = observation.reshape(-1,)/self.max_obs  # reward, done, info can't be included
    
    assert normalized_obs.shape == (self.mpc_state_size, ) 
    return normalized_obs

  def get_random_init_state(self):
    # TODO: pass initial set and randomly sample from there.
    # TODO: load initial state from params. 
    pos = np.zeros(3)
    pos[0:2] = np.random.uniform(0.1, -0.1, 2)
    pos[2] = np.random.uniform(-0.1, 0.1) #np.random.uniform(0.9, 1.1)
    vel = np.random.uniform(-0.1, 0.1, 3)
    rp = np.random.uniform(-0.1, 0.1, 2)
    return np.concatenate((pos, vel, rp), axis=None).reshape(self.mpc_state_size,)
  
  def render(self, mode='human'):
    raise NotImplementedError()
    return
  
  def close (self):
    raise NotImplementedError()
    return
