import pandas as pd
from imitation.data import types, rollout
from imitation.data.rollout import generate_trajectories, generate_trajectories_for_benchmark, rollout_stats

def compute_success_rate(trajectories, eval_episodes, venv):
    nominal_ep_len = venv.get_attr("len_episode")[0]
    success = 0
    for traj in trajectories:
        if len(traj.obs) == int(nominal_ep_len + 1):
            success += 1
    return float(success)/float(eval_episodes)

def compute_success(trajectories, venv):
    try:
        nominal_ep_len = venv.get_attr("len_episode")[0]
    except: 
        nominal_ep_len = venv.len_episode
    success = []
    for traj in trajectories:
        if len(traj.obs) == int(nominal_ep_len + 1):
            success.append(True)
        else: 
            success.append(False)
    return success

def evaluate_policy(policy, venv, log_path, eval_episodes = 50):
    # venv.env_method(env_dist_f_call, (disturbance))
    trajectories = generate_trajectories(
        policy,
        venv,
        deterministic_policy=True,
        total_demos_per_round=eval_episodes
    )
    stats, descriptors = rollout_stats(trajectories)
    descriptors["success"] = compute_success(trajectories, venv)
    stats["success_rate"] = float(sum(descriptors["success"]))/float(stats['n_traj'])
    
    logs = pd.DataFrame(descriptors)
    # logs = logs.assign(disturbance = disturbance)
    # save logs 
    if log_path is not None:
        logs.to_pickle(log_path+".pkl")
    return stats

def evaluate_policy_for_benchmark(policy, venv, log_path, eval_episodes = 50):
    # venv.env_method(env_dist_f_call, (disturbance))
    trajectories, total_obs_avoidance_failure, total_trans_dyn_limit_failure, total_yaw_dyn_limit_failure, total_failure, num_demos, mean_computation_time \
     = generate_trajectories_for_benchmark(
        policy,
        venv,
        deterministic_policy=True,
        total_demos_per_round=eval_episodes
    )
    stats, descriptors = rollout_stats(trajectories)
    stats["obs_avoidance_failure_rate"] = float(total_obs_avoidance_failure) / float(num_demos) * 100
    stats["trans_dyn_limit_failure_rate"] = float(total_trans_dyn_limit_failure) / float(num_demos) * 100
    stats["yaw_dyn_limit_failure_rate"] = float(total_yaw_dyn_limit_failure) / float(num_demos) * 100
    stats["success_rate"] = (1 - float(total_failure)/float(num_demos)) * 100
    stats["mean_computation_time"] = mean_computation_time * 1000 # ms
    
    logs = pd.DataFrame(descriptors)
    # logs = logs.assign(disturbance = disturbance)
    # save logs 
    if log_path is not None:
        logs.to_pickle(log_path+".pkl")
    return stats