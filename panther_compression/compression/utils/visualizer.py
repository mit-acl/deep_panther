import pathlib

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(axs, obs_hist, act_hist, ref_hist, sampling_time_s=0.1):
    # Constraints
    #p_min = -3.0
    #p_max = 3.0
    #v_min = -1.5
    #v_max = 1.5
    T_max = len(obs_hist)
    axis_name = ["x", "y", "z"]
    time = [i*sampling_time_s for i in range(T_max)]
    for ax in range(3):
        # position
        axs[0, ax].plot(time, [state[ax] for state in obs_hist])
        axs[0, ax].plot(time, [ref[ax] for ref in ref_hist], '--', label='ref')
        #axs[0, ax].hlines(y=[p_min, p_max], xmin=0, xmax=T_max, linestyles='dashed', color='red')
        axs[0, ax].set_ylabel(f'Pos-{axis_name[ax]}(m)')
        #axs[0, ax].legend()
            
        # velocity
        axs[1, ax].plot(time, [state[ax+3] for state in obs_hist])
        axs[1, ax].plot(time, [ref[ax+3] for ref in ref_hist], '--', label='ref')
        #axs[1, ax].hlines(y=[v_min, v_max], xmin=0, xmax=T_max, linestyles='dashed', color='red')
        axs[1, ax].set_ylabel(f'Vel-{axis_name[ax]} (m/s)')
        #axs[1, ax].legend()

    # Roll, pitch, thrust 
    axs[2, 0].plot(time, [state[6]/np.pi*180. for state in obs_hist])
    axs[2, 0].plot(time[:-1], [cmd[0]/np.pi*180. for cmd in act_hist], '--', label='cmd')
    axs[2, 0].set_ylabel("Attitude Roll (deg)")
    axs[2, 0].set_xlabel('Time (s)')
    #axs[2, 0].legend()
    
    axs[2, 1].plot(time, [state[7]/np.pi*180. for state in obs_hist])
    axs[2, 1].plot(time[:-1], [cmd[1]/np.pi*180. for cmd in act_hist], '--', label='cmd')
    axs[2, 1].set_ylabel("Attitude Pitch (deg)")
    axs[2, 1].set_xlabel('Time (s)')
    #axs[2, 1].legend()

    axs[2, 2].plot(time[:-1], [cmd[2]/9.80655 for cmd in act_hist], label='cmd_thrust')
    axs[2, 2].set_ylabel('Thrust-acc ($g$)')
    axs[2, 2].set_xlabel('Time (s)')

def plot_reward_and_q(ax, rwrd_hist, q_hist, sampling_time_s=0.1):
    T_max = len(rwrd_hist)
    time = [i*sampling_time_s for i in range(T_max)]
    ax[0].plot(time, rwrd_hist)
    ax[0].set_ylabel('Reward')
    ax[0].set_xlabel('Time (s)')
    
    ax[1].plot(time, q_hist)
    ax[1].set_ylabel('Q-function')
    ax[1].set_xlabel('Time (s)')
    
def plot_statespace_trajectories(ax, obs_hist):
    state = np.reshape(obs_hist, (len(obs_hist), 2)) #[pos, vel]
    ax.plot(state[:, 0], state[:,1])
    ax.set_xlabel('x_1 (pos, m)')
    ax.set_ylabel('x_2 (vel, m/s)')
    
def plot_3d_trajectory(axs_p, obs_hist, ref_hist, label=None):
    x = [state[0] for state in obs_hist]
    y = [state[1] for state in obs_hist]
    z = [state[2] for state in obs_hist]
    x_ref = [state[0] for state in ref_hist]
    y_ref = [state[1] for state in ref_hist]
    z_ref = [state[2] for state in ref_hist]
    #z_range = np.ptp(z+z_ref)
    #if np.abs(np.diff(z_range)) < 0.1:
    #    z_range = np.array([z_range[0] - 0.1, z_range[1] + 0.1])  
    axs_p.set_box_aspect((np.ptp(x+x_ref), np.ptp(y+y_ref), np.ptp(z+z_ref)))  # aspect ratio is 1:1:1 in data space
    axs_p.plot3D(x, y, z, label=label)
    axs_p.plot3D(x_ref, y_ref, z_ref, '--', label='ref')
    axs_p.set_xlabel('x (m)')
    axs_p.set_ylabel('y (m)')
    axs_p.set_zlabel('z (m)')
    axs_p.set_zlim3d(1, 1.4)
    axs_p.legend()